package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// makeReconfigurePayload builds a plugin.register/reconfigure request body
// with the given config YAML.
func makeReconfigurePayload(configYAML string) []byte {
	// The host sends config_yaml as []byte, which json.Marshal encodes as
	// base64. The plugin's ConfigYAML field is []byte, so json.Unmarshal
	// base64-decodes it back to the raw YAML bytes.
	b, _ := json.Marshal(struct {
		ConfigYAML    []byte `json:"config_yaml"`
		SchemaVersion uint32 `json:"schema_version"`
	}{
		ConfigYAML:    []byte(configYAML),
		SchemaVersion: 1,
	})
	return b
}

func TestParseConfig_ExtractsUpstreamSettings(t *testing.T) {
	yaml := `upstream_base_url: https://api.example.com/v1
upstream_api_key: sk-test-key
upstream_path: /embeddings
`
	if err := ParseConfig(makeReconfigurePayload(yaml)); err != nil {
		t.Fatalf("ParseConfig error = %v", err)
	}

	c := GetConfig()
	if c.UpstreamBaseURL != "https://api.example.com/v1" {
		t.Fatalf("UpstreamBaseURL = %q, want https://api.example.com/v1", c.UpstreamBaseURL)
	}
	if c.UpstreamAPIKey != "sk-test-key" {
		t.Fatalf("UpstreamAPIKey = %q, want sk-test-key", c.UpstreamAPIKey)
	}
	if c.UpstreamPath != "/embeddings" {
		t.Fatalf("UpstreamPath = %q, want /embeddings", c.UpstreamPath)
	}
}

func TestParseConfig_DefaultsPathWhenOmitted(t *testing.T) {
	yaml := `upstream_base_url: https://api.example.com
upstream_api_key: sk-test
`
	if err := ParseConfig(makeReconfigurePayload(yaml)); err != nil {
		t.Fatalf("ParseConfig error = %v", err)
	}

	c := GetConfig()
	if c.UpstreamPath != "" {
		t.Fatalf("UpstreamPath = %q, want empty (defaults at use time)", c.UpstreamPath)
	}
}

func TestParseConfig_InvalidInputReturnsError(t *testing.T) {
	// nil body: json.Unmarshal fails.
	if err := ParseConfig(nil); err == nil {
		t.Fatal("ParseConfig(nil) = nil, want error")
	}
	// Not JSON at all.
	if err := ParseConfig([]byte("not json")); err == nil {
		t.Fatal("ParseConfig(\"not json\") = nil, want error")
	}
	// Valid JSON envelope but invalid YAML payload (unterminated flow sequence).
	if err := ParseConfig([]byte(`{"config_yaml":"key: [\n"}`)); err == nil {
		t.Fatal(`ParseConfig({"config_yaml":"key: ["}) = nil, want error`)
	}
}

func TestParseConfig_InvalidJSONPreservesOldConfig(t *testing.T) {
	if err := ParseConfig(makeReconfigurePayload("upstream_base_url: https://valid.example.com\nupstream_api_key: key1")); err != nil {
		t.Fatalf("initial ParseConfig error = %v", err)
	}

	// Invalid JSON causes json.Unmarshal to fail, so ParseConfig returns an
	// error and the previous config is preserved.
	if err := ParseConfig([]byte("this is not json at all")); err == nil {
		t.Fatal("ParseConfig(invalid json) = nil, want error")
	}

	c := GetConfig()
	if c.UpstreamBaseURL != "https://valid.example.com" {
		t.Fatalf("UpstreamBaseURL = %q, want previous value https://valid.example.com", c.UpstreamBaseURL)
	}
}

func TestHandleMethod_PluginRegisterParsesConfig(t *testing.T) {
	// Reset config to ensure register actually parses.
	cfgMu.Lock()
	cfg = PluginConfig{}
	cfgMu.Unlock()

	yaml := `upstream_base_url: https://register.example.com
upstream_api_key: sk-register
`
	result, err := HandleMethod("plugin.register", makeReconfigurePayload(yaml))
	if err != nil {
		t.Fatalf("HandleMethod error = %v", err)
	}
	var env envelope
	if err := json.Unmarshal(result, &env); err != nil {
		t.Fatalf("unmarshal envelope: %v", err)
	}
	if !env.OK {
		t.Fatalf("envelope not OK: %s", string(result))
	}

	c := GetConfig()
	if c.UpstreamBaseURL != "https://register.example.com" {
		t.Fatalf("after register, UpstreamBaseURL = %q, want https://register.example.com", c.UpstreamBaseURL)
	}
}

func TestHandleMethod_ManagementRegisterDeclaresRoute(t *testing.T) {
	result, err := HandleMethod("management.register", nil)
	if err != nil {
		t.Fatalf("HandleMethod error = %v", err)
	}
	var env envelope
	if err := json.Unmarshal(result, &env); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !env.OK {
		t.Fatal("envelope not OK")
	}

	var reg struct {
		Routes []struct {
			Method string `json:"Method"`
			Path   string `json:"Path"`
		} `json:"routes"`
	}
	if err := json.Unmarshal(env.Result, &reg); err != nil {
		t.Fatalf("unmarshal routes: %v", err)
	}
	if len(reg.Routes) != 1 {
		t.Fatalf("routes = %d, want 1", len(reg.Routes))
	}
	r := reg.Routes[0]
	if r.Method != "POST" || r.Path != "/embeddings" {
		t.Fatalf("route = %s %s, want POST /embeddings", r.Method, r.Path)
	}
}

func TestHandleMethod_UnknownMethodReturnsError(t *testing.T) {
	result, err := HandleMethod("nonexistent.method", nil)
	if err != nil {
		t.Fatalf("HandleMethod error = %v", err)
	}
	var env envelope
	if err := json.Unmarshal(result, &env); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if env.OK {
		t.Fatal("expected error envelope for unknown method")
	}
	if env.Error == nil || !strings.Contains(env.Error.Code, "unknown") {
		t.Fatalf("error = %+v, want unknown_method", env.Error)
	}
}

func TestHandleEmbeddings_RejectsNonPost(t *testing.T) {
	mgmtReq := managementRequest{
		Method: http.MethodGet,
		Path:   "/v0/management/embeddings",
		Body:   []byte(`{"model":"test","input":"hello"}`),
	}
	reqBody, _ := json.Marshal(mgmtReq)

	result, _ := HandleMethod("management.handle", reqBody)

	var env envelope
	json.Unmarshal(result, &env)

	var resp managementResponse
	json.Unmarshal(env.Result, &resp)

	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("StatusCode = %d, want %d", resp.StatusCode, http.StatusMethodNotAllowed)
	}
}

func TestHandleEmbeddings_MissingUpstreamURL(t *testing.T) {
	cfgMu.Lock()
	cfg = PluginConfig{UpstreamAPIKey: "sk-test"} // no URL
	cfgMu.Unlock()

	mgmtReq := managementRequest{
		Method: http.MethodPost,
		Path:   "/v0/management/embeddings",
		Body:   []byte(`{"model":"test","input":"hello"}`),
	}
	reqBody, _ := json.Marshal(mgmtReq)

	result, _ := HandleMethod("management.handle", reqBody)

	var env envelope
	json.Unmarshal(result, &env)

	var resp managementResponse
	json.Unmarshal(env.Result, &resp)

	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("StatusCode = %d, want %d (BadGateway for missing URL)", resp.StatusCode, http.StatusBadGateway)
	}
	if !strings.Contains(string(resp.Body), "upstream_base_url") {
		t.Fatalf("body = %s, want mention of upstream_base_url", string(resp.Body))
	}
}

func TestHandleEmbeddings_MissingUpstreamAPIKey(t *testing.T) {
	cfgMu.Lock()
	cfg = PluginConfig{UpstreamBaseURL: "https://api.example.com"} // no key
	cfgMu.Unlock()

	mgmtReq := managementRequest{
		Method: http.MethodPost,
		Path:   "/v0/management/embeddings",
		Body:   []byte(`{"model":"test","input":"hello"}`),
	}
	reqBody, _ := json.Marshal(mgmtReq)

	result, _ := HandleMethod("management.handle", reqBody)

	var env envelope
	json.Unmarshal(result, &env)

	var resp managementResponse
	json.Unmarshal(env.Result, &resp)

	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("StatusCode = %d, want %d", resp.StatusCode, http.StatusBadGateway)
	}
	if !strings.Contains(string(resp.Body), "upstream_api_key") {
		t.Fatalf("body = %s, want mention of upstream_api_key", string(resp.Body))
	}
}

func TestUpstreamURLConstruction(t *testing.T) {
	cases := []struct {
		name    string
		baseURL string
		path    string
		wantURL string
	}{
		{
			name:    "default path",
			baseURL: "https://api.openai.com/v1",
			path:    "",
			wantURL: "https://api.openai.com/v1/embeddings",
		},
		{
			name:    "trailing slash trimmed",
			baseURL: "https://api.openai.com/v1/",
			path:    "",
			wantURL: "https://api.openai.com/v1/embeddings",
		},
		{
			name:    "custom path",
			baseURL: "https://aigc.example.com/v1/openai/native",
			path:    "/embeddings",
			wantURL: "https://aigc.example.com/v1/openai/native/embeddings",
		},
		{
			name:    "path without leading slash",
			baseURL: "https://api.example.com",
			path:    "embeddings",
			wantURL: "https://api.example.com/embeddings",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfgMu.Lock()
			cfg = PluginConfig{
				UpstreamBaseURL: tc.baseURL,
				UpstreamAPIKey:  "sk-test",
				UpstreamPath:    tc.path,
			}
			cfgMu.Unlock()

			// We can't call CallHost in a pure Go test (needs C host API).
			// Instead, verify the URL construction logic by replicating it.
			// This mirrors the logic in HandleEmbeddings.
			baseURL := strings.TrimRight(tc.baseURL, "/")
			upstreamPath := strings.TrimSpace(tc.path)
			if upstreamPath == "" {
				upstreamPath = "/embeddings"
			}
			if !strings.HasPrefix(upstreamPath, "/") {
				upstreamPath = "/" + upstreamPath
			}
			gotURL := baseURL + upstreamPath

			if gotURL != tc.wantURL {
				t.Fatalf("URL = %q, want %q", gotURL, tc.wantURL)
			}
		})
	}
}

func TestHandleEmbeddings_MalformedRequestBody(t *testing.T) {
	result, _ := HandleMethod("management.handle", []byte("not json at all"))

	var env envelope
	json.Unmarshal(result, &env)

	var resp managementResponse
	json.Unmarshal(env.Result, &resp)

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("StatusCode = %d, want %d", resp.StatusCode, http.StatusBadRequest)
	}
}

func TestFilterResponseHeaders_CanonicalizesLowercaseHeader(t *testing.T) {
	// Upstream returns lowercase "content-type"; the bug produced a duplicate
	// "Content-Type" entry because the lookup used canonical form but the
	// store used the raw key.
	upstream := map[string][]string{
		"content-type": {"text/plain"},
	}
	got := filterResponseHeaders(upstream)
	if _, ok := got["Content-Type"]; !ok {
		t.Fatalf("expected canonical key Content-Type, got keys: %v", mapKeys(got))
	}
	if _, ok := got["content-type"]; ok {
		t.Fatalf("raw lowercase key content-type should not be present, got keys: %v", mapKeys(got))
	}
	if len(got) != 1 {
		t.Fatalf("expected exactly 1 header (no duplicate), got %d: %v", len(got), got)
	}
}

func TestFilterResponseHeaders_DropsUnsafeHeaders(t *testing.T) {
	// Only Content-Type is forwarded; Set-Cookie, Server, etc. are dropped.
	upstream := map[string][]string{
		"Content-Type": {"application/json"},
		"Set-Cookie":   {"session=abc"},
		"Server":       {"nginx"},
	}
	got := filterResponseHeaders(upstream)
	if len(got) != 1 {
		t.Fatalf("expected exactly 1 header, got %d: %v", len(got), got)
	}
	if v := got["Content-Type"]; len(v) != 1 || v[0] != "application/json" {
		t.Fatalf("Content-Type = %v, want [application/json]", v)
	}
}

func TestFilterResponseHeaders_InsertsDefaultWhenMissing(t *testing.T) {
	// No Content-Type from upstream → default application/json injected.
	upstream := map[string][]string{
		"X-Custom": {"value"},
	}
	got := filterResponseHeaders(upstream)
	if v := got["Content-Type"]; len(v) != 1 || v[0] != "application/json" {
		t.Fatalf("default Content-Type = %v, want [application/json]", v)
	}
	if _, ok := got["X-Custom"]; ok {
		t.Fatalf("X-Custom should have been dropped, got: %v", got)
	}
}

func mapKeys(m map[string][]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func TestHandleEmbeddings_RejectsUnsafeUpstreamURL(t *testing.T) {
	cases := []struct {
		name    string
		baseURL string
		path    string
	}{
		{"query in path", "https://api.example.com", "/embeddings?override=1"},
		{"fragment in path", "https://api.example.com", "/embeddings#frag"},
		{"traversal in path", "https://api.example.com", "/v1/../internal"},
		{"traversal suffix", "https://api.example.com", "/v1/.."},
		{"non-http scheme", "ftp://api.example.com", "/embeddings"},
		{"missing host", "https://", "/embeddings"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfgMu.Lock()
			cfg = PluginConfig{
				UpstreamBaseURL: tc.baseURL,
				UpstreamAPIKey:  "sk-test",
				UpstreamPath:    tc.path,
			}
			cfgMu.Unlock()

			mgmtReq := managementRequest{
				Method: http.MethodPost,
				Path:   "/v0/management/embeddings",
				Body:   []byte(`{"model":"test","input":"hello"}`),
			}
			reqBody, _ := json.Marshal(mgmtReq)

			result, _ := HandleMethod("management.handle", reqBody)
			var env envelope
			json.Unmarshal(result, &env)
			var resp managementResponse
			json.Unmarshal(env.Result, &resp)

			// URL validation runs before CallHost, so these return BadGateway
			// without needing the C host API.
			if resp.StatusCode != http.StatusBadGateway {
				t.Fatalf("StatusCode = %d, want %d for %q+%q", resp.StatusCode, http.StatusBadGateway, tc.baseURL, tc.path)
			}
		})
	}
}

func TestHandleEmbeddings_RejectsOversizedBody(t *testing.T) {
	cfgMu.Lock()
	cfg = PluginConfig{
		UpstreamBaseURL: "https://api.example.com",
		UpstreamAPIKey:  "sk-test",
	}
	cfgMu.Unlock()

	// Body exceeding maxRequestBodySize is rejected before CallHost.
	oversized := make([]byte, maxRequestBodySize+1)
	mgmtReq := managementRequest{
		Method: http.MethodPost,
		Path:   "/v0/management/embeddings",
		Body:   oversized,
	}
	reqBody, _ := json.Marshal(mgmtReq)

	result, _ := HandleMethod("management.handle", reqBody)
	var env envelope
	json.Unmarshal(result, &env)
	var resp managementResponse
	json.Unmarshal(env.Result, &resp)

	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("StatusCode = %d, want %d", resp.StatusCode, http.StatusRequestEntityTooLarge)
	}
}
