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
	b, _ := json.Marshal(struct {
		ConfigYAML    []byte `json:"config_yaml"`
		SchemaVersion uint32 `json:"schema_version"`
	}{
		ConfigYAML:    []byte(configYAML),
		SchemaVersion: 1,
	})
	return b
}

// makeMgmtRequest builds a management.handle request body.
func makeMgmtRequest(method, path string, body any) []byte {
	var bodyBytes []byte
	switch v := body.(type) {
	case []byte:
		bodyBytes = v
	case string:
		bodyBytes = []byte(v)
	default:
		bodyBytes, _ = json.Marshal(v)
	}
	b, _ := json.Marshal(managementRequest{
		Method: method,
		Path:   path,
		Body:   bodyBytes,
	})
	return b
}

func resetConfig() {
	cfgMu.Lock()
	cfg = PluginConfig{}
	cfgMu.Unlock()
}

// --- ParseConfig ---

func TestParseConfig_ExtractsModuleSettings(t *testing.T) {
	yaml := `embeddings:
  enabled: true
  providers:
    - name: openai
      base_url: https://api.openai.com/v1
      path: /embeddings
      api_keys: [sk-key1, sk-key2]
      models:
        - name: text-embedding-3-small
          alias: emb-small
rerank:
  enabled: true
  providers:
    - name: aigc
      base_url: https://api.example.com/v1
      api_keys: [appid1]
      models:
        - name: Qwen3-Reranker-8B
`
	if err := ParseConfig(makeReconfigurePayload(yaml)); err != nil {
		t.Fatalf("ParseConfig error = %v", err)
	}

	c := GetConfig()
	if !c.Embeddings.Enabled {
		t.Fatal("Embeddings.Enabled = false, want true")
	}
	if len(c.Embeddings.Providers) != 1 {
		t.Fatalf("Embeddings providers = %d, want 1", len(c.Embeddings.Providers))
	}
	p := c.Embeddings.Providers[0]
	if p.Name != "openai" || p.BaseURL != "https://api.openai.com/v1" {
		t.Fatalf("provider = %+v", p)
	}
	if len(p.APIKeys) != 2 || p.APIKeys[0] != "sk-key1" {
		t.Fatalf("APIKeys = %v", p.APIKeys)
	}
	if len(p.Models) != 1 || p.Models[0].Name != "text-embedding-3-small" || p.Models[0].Alias != "emb-small" {
		t.Fatalf("Models = %+v", p.Models)
	}
	if !c.Rerank.Enabled {
		t.Fatal("Rerank.Enabled = false, want true")
	}
	// path defaults at use time, not parse time
	if c.Rerank.Providers[0].Path != "" {
		t.Fatalf("Rerank path = %q, want empty (defaults at use time)", c.Rerank.Providers[0].Path)
	}
}

func TestParseConfig_InvalidInputReturnsError(t *testing.T) {
	if err := ParseConfig(nil); err == nil {
		t.Fatal("ParseConfig(nil) = nil, want error")
	}
	if err := ParseConfig([]byte("not json")); err == nil {
		t.Fatal("ParseConfig(\"not json\") = nil, want error")
	}
	if err := ParseConfig([]byte(`{"config_yaml":"key: [\n"}`)); err == nil {
		t.Fatal(`ParseConfig({"config_yaml":"key: ["}) = nil, want error`)
	}
}

func TestParseConfig_InvalidJSONPreservesOldConfig(t *testing.T) {
	if err := ParseConfig(makeReconfigurePayload("embeddings:\n  enabled: true\n")); err != nil {
		t.Fatalf("initial ParseConfig error = %v", err)
	}
	if err := ParseConfig([]byte("this is not json at all")); err == nil {
		t.Fatal("ParseConfig(invalid json) = nil, want error")
	}
	c := GetConfig()
	if !c.Embeddings.Enabled {
		t.Fatal("Embeddings.Enabled = false, want preserved true")
	}
}

// --- HandleMethod dispatch ---

func TestHandleMethod_PluginRegisterParsesConfig(t *testing.T) {
	resetConfig()
	yaml := `embeddings:
  enabled: true
  providers:
    - name: p1
      base_url: https://api.example.com/v1
      api_keys: [k1]
      models:
        - name: m1
`
	result, err := HandleMethod("plugin.register", makeReconfigurePayload(yaml))
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
	c := GetConfig()
	if !c.Embeddings.Enabled {
		t.Fatal("register did not parse config")
	}
}

func TestHandleMethod_ManagementRegisterDeclaresRoutes(t *testing.T) {
	result, err := HandleMethod("management.register", nil)
	if err != nil {
		t.Fatalf("HandleMethod error = %v", err)
	}
	var env envelope
	json.Unmarshal(result, &env)
	var reg struct {
		Routes []struct {
			Method string `json:"Method"`
			Path   string `json:"Path"`
		} `json:"routes"`
	}
	json.Unmarshal(env.Result, &reg)
	if len(reg.Routes) != 2 {
		t.Fatalf("routes = %d, want 2", len(reg.Routes))
	}
	want := map[string]bool{"/embeddings": false, "/rerank": false}
	for _, r := range reg.Routes {
		if r.Method != "POST" {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if _, ok := want[r.Path]; !ok {
			t.Fatalf("unexpected path %s", r.Path)
		}
		want[r.Path] = true
	}
	for p, found := range want {
		if !found {
			t.Fatalf("route %s not declared", p)
		}
	}
}

func TestHandleMethod_UnknownMethodReturnsError(t *testing.T) {
	result, err := HandleMethod("nonexistent.method", nil)
	if err != nil {
		t.Fatalf("HandleMethod error = %v", err)
	}
	var env envelope
	json.Unmarshal(result, &env)
	if env.OK {
		t.Fatal("envelope OK, want error")
	}
}

// --- resolveProviders ---

func TestResolveProviders_MatchesAliasThenName(t *testing.T) {
	m := Module{
		Enabled: true,
		Providers: []Provider{
			{Name: "p1", Models: []ModelMapping{{Name: "real-1", Alias: "alias-1"}}},
			{Name: "p2", Models: []ModelMapping{{Name: "real-2", Alias: "alias-1"}}}, // same alias, different provider
		},
	}
	// alias match returns both providers
	matches := resolveProviders(m, "alias-1")
	if len(matches) != 2 {
		t.Fatalf("alias matches = %d, want 2", len(matches))
	}
	// name match also returns both providers (p1 has real-1, p2 has real-2)
	matches = resolveProviders(m, "real-1")
	if len(matches) != 1 || matches[0].Provider.Name != "p1" {
		t.Fatalf("name matches = %+v, want 1 (p1)", matches)
	}
}

func TestResolveProviders_AliasDefaultsToName(t *testing.T) {
	m := Module{
		Enabled: true,
		Providers: []Provider{
			{Name: "p1", Models: []ModelMapping{{Name: "only-name"}}}, // no alias
		},
	}
	matches := resolveProviders(m, "only-name")
	if len(matches) != 1 {
		t.Fatalf("matches = %d, want 1", len(matches))
	}
}

func TestResolveProviders_NoMatchReturnsEmpty(t *testing.T) {
	m := Module{
		Enabled: true,
		Providers: []Provider{
			{Name: "p1", Models: []ModelMapping{{Name: "m1", Alias: "a1"}}},
		},
	}
	if len(resolveProviders(m, "unknown")) != 0 {
		t.Fatal("want 0 matches for unknown model")
	}
}

// --- rewriteModel ---

func TestRewriteModel_ReplacesModelField(t *testing.T) {
	out, err := rewriteModel([]byte(`{"model":"alias-1","input":"hello","extra":{"k":"v"}}`), "real-1")
	if err != nil {
		t.Fatalf("rewriteModel error = %v", err)
	}
	var obj map[string]any
	json.Unmarshal(out, &obj)
	if obj["model"] != "real-1" {
		t.Fatalf("model = %v, want real-1", obj["model"])
	}
	if obj["input"] != "hello" {
		t.Fatalf("input = %v, want hello", obj["input"])
	}
}

func TestRewriteModel_PreservesNumericPrecision(t *testing.T) {
	// dimensions is an integer; with map[string]any it would become float64
	// and re-encode as "1536" or "1.536e+03". RawMessage keeps it verbatim.
	in := []byte(`{"model":"alias-1","input":"hello","dimensions":1536}`)
	out, err := rewriteModel(in, "real-1")
	if err != nil {
		t.Fatalf("rewriteModel error = %v", err)
	}
	s := string(out)
	if !strings.Contains(s, `"dimensions":1536`) {
		t.Fatalf("dimensions not preserved verbatim: %s", s)
	}
	if strings.Contains(s, "1.536e") || strings.Contains(s, `1536.0`) {
		t.Fatalf("dimensions became float: %s", s)
	}
}

func TestRewriteModel_InvalidJSONReturnsError(t *testing.T) {
	if _, err := rewriteModel([]byte("not json"), "x"); err == nil {
		t.Fatal("want error for invalid JSON")
	}
}

// --- buildUpstreamURL ---

func TestBuildUpstreamURL_DefaultPath(t *testing.T) {
	p := &Provider{BaseURL: "https://api.example.com/v1"}
	got := buildUpstreamURL(p, "/embeddings")
	want := "https://api.example.com/v1/embeddings"
	if got != want {
		t.Fatalf("URL = %q, want %q", got, want)
	}
}

func TestBuildUpstreamURL_CustomPath(t *testing.T) {
	p := &Provider{BaseURL: "https://api.example.com/v1/", Path: "/rerank"}
	got := buildUpstreamURL(p, "/embeddings")
	want := "https://api.example.com/v1/rerank"
	if got != want {
		t.Fatalf("URL = %q, want %q", got, want)
	}
}

func TestBuildUpstreamURL_RejectsInvalid(t *testing.T) {
	cases := []struct {
		name string
		p    *Provider
	}{
		{"empty base", &Provider{BaseURL: ""}},
		{"non-http scheme", &Provider{BaseURL: "ftp://example.com"}},
		{"traversal", &Provider{BaseURL: "https://example.com", Path: "/../etc"}},
		{"query in path", &Provider{BaseURL: "https://example.com", Path: "/embed?x=1"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := buildUpstreamURL(tc.p, "/embeddings"); got != "" {
				t.Fatalf("URL = %q, want empty", got)
			}
		})
	}
}

// --- handleModule routing (static, no CallHost needed) ---

func TestHandleEmbeddings_ModuleDisabledReturns404(t *testing.T) {
	resetConfig()
	req := makeMgmtRequest(http.MethodPost, "/v0/management/embeddings", `{"model":"m1","input":"hi"}`)
	result, _ := HandleMethod("management.handle", req)
	var env envelope
	json.Unmarshal(result, &env)
	var resp managementResponse
	json.Unmarshal(env.Result, &resp)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("StatusCode = %d, want 404", resp.StatusCode)
	}
}

func TestHandleRerank_ModuleDisabledReturns404(t *testing.T) {
	resetConfig()
	// Only embeddings enabled, rerank disabled
	cfgMu.Lock()
	cfg = PluginConfig{Embeddings: Module{Enabled: true}}
	cfgMu.Unlock()
	req := makeMgmtRequest(http.MethodPost, "/v0/management/rerank", `{"model":"m1","query":"q","documents":["a"]}`)
	result, _ := HandleMethod("management.handle", req)
	var env envelope
	json.Unmarshal(result, &env)
	var resp managementResponse
	json.Unmarshal(env.Result, &resp)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("StatusCode = %d, want 404", resp.StatusCode)
	}
}

func TestHandleModule_ModelNotConfiguredReturns404(t *testing.T) {
	cfgMu.Lock()
	cfg = PluginConfig{
		Embeddings: Module{
			Enabled: true,
			Providers: []Provider{
				{Name: "p1", BaseURL: "https://api.example.com/v1", APIKeys: []string{"k1"}, Models: []ModelMapping{{Name: "known-model"}}},
			},
		},
	}
	cfgMu.Unlock()
	req := makeMgmtRequest(http.MethodPost, "/v0/management/embeddings", `{"model":"unknown-model","input":"hi"}`)
	result, _ := HandleMethod("management.handle", req)
	var env envelope
	json.Unmarshal(result, &env)
	var resp managementResponse
	json.Unmarshal(env.Result, &resp)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("StatusCode = %d, want 404 for unknown model", resp.StatusCode)
	}
	if !strings.Contains(string(resp.Body), "not configured") {
		t.Fatalf("body = %s, want 'not configured'", string(resp.Body))
	}
}

func TestHandleModule_RejectsNonPost(t *testing.T) {
	cfgMu.Lock()
	cfg = PluginConfig{Embeddings: Module{Enabled: true, Providers: []Provider{{Name: "p1", BaseURL: "https://api.example.com", APIKeys: []string{"k1"}, Models: []ModelMapping{{Name: "m1"}}}}}}
	cfgMu.Unlock()
	req := makeMgmtRequest(http.MethodGet, "/v0/management/embeddings", `{"model":"m1"}`)
	result, _ := HandleMethod("management.handle", req)
	var env envelope
	json.Unmarshal(result, &env)
	var resp managementResponse
	json.Unmarshal(env.Result, &resp)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("StatusCode = %d, want 405", resp.StatusCode)
	}
}

func TestHandleModule_RejectsOversizedBody(t *testing.T) {
	cfgMu.Lock()
	cfg = PluginConfig{Embeddings: Module{Enabled: true, Providers: []Provider{{Name: "p1", BaseURL: "https://api.example.com", APIKeys: []string{"k1"}, Models: []ModelMapping{{Name: "m1"}}}}}}
	cfgMu.Unlock()
	oversized := make([]byte, maxRequestBodySize+1)
	req := makeMgmtRequest(http.MethodPost, "/v0/management/embeddings", oversized)
	result, _ := HandleMethod("management.handle", req)
	var env envelope
	json.Unmarshal(result, &env)
	var resp managementResponse
	json.Unmarshal(env.Result, &resp)
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("StatusCode = %d, want 413", resp.StatusCode)
	}
}

func TestHandleModule_MalformedRequestBody(t *testing.T) {
	cfgMu.Lock()
	cfg = PluginConfig{Embeddings: Module{Enabled: true, Providers: []Provider{{Name: "p1", BaseURL: "https://api.example.com", APIKeys: []string{"k1"}, Models: []ModelMapping{{Name: "m1"}}}}}}
	cfgMu.Unlock()
	req := makeMgmtRequest(http.MethodPost, "/v0/management/embeddings", "not json")
	result, _ := HandleMethod("management.handle", req)
	var env envelope
	json.Unmarshal(result, &env)
	var resp managementResponse
	json.Unmarshal(env.Result, &resp)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("StatusCode = %d, want 400", resp.StatusCode)
	}
}

func TestHandleModule_MissingModelField(t *testing.T) {
	cfgMu.Lock()
	cfg = PluginConfig{Embeddings: Module{Enabled: true, Providers: []Provider{{Name: "p1", BaseURL: "https://api.example.com", APIKeys: []string{"k1"}, Models: []ModelMapping{{Name: "m1"}}}}}}
	cfgMu.Unlock()
	req := makeMgmtRequest(http.MethodPost, "/v0/management/embeddings", `{"input":"hi"}`)
	result, _ := HandleMethod("management.handle", req)
	var env envelope
	json.Unmarshal(result, &env)
	var resp managementResponse
	json.Unmarshal(env.Result, &resp)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("StatusCode = %d, want 400 for missing model", resp.StatusCode)
	}
}

// --- HandleManagement dispatch ---

func TestHandleManagement_UnknownRouteReturns404(t *testing.T) {
	cfgMu.Lock()
	cfg = PluginConfig{Embeddings: Module{Enabled: true}, Rerank: Module{Enabled: true}}
	cfgMu.Unlock()
	req := makeMgmtRequest(http.MethodPost, "/v0/management/unknown", `{}`)
	result, _ := HandleMethod("management.handle", req)
	var env envelope
	json.Unmarshal(result, &env)
	var resp managementResponse
	json.Unmarshal(env.Result, &resp)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("StatusCode = %d, want 404", resp.StatusCode)
	}
}

// --- filterResponseHeaders ---

func TestParseConfig_LegacyConfigMigratedToModules(t *testing.T) {
	yaml := `upstream_base_url: https://api.example.com/v1
upstream_api_key: sk-legacy-key
upstream_path: /embeddings
`
	if err := ParseConfig(makeReconfigurePayload(yaml)); err != nil {
		t.Fatalf("ParseConfig error = %v", err)
	}
	c := GetConfig()
	// Only embeddings is migrated; legacy config does not enable rerank.
	if !c.Embeddings.Enabled || len(c.Embeddings.Providers) != 1 {
		t.Fatalf("Embeddings not migrated: %+v", c.Embeddings)
	}
	if c.Rerank.Enabled {
		t.Fatalf("Rerank should not be enabled by legacy migration: %+v", c.Rerank)
	}
	ep := c.Embeddings.Providers[0]
	if ep.BaseURL != "https://api.example.com/v1" || ep.Path != "/embeddings" {
		t.Fatalf("embeddings provider = %+v", ep)
	}
	if len(ep.APIKeys) != 1 || ep.APIKeys[0] != "sk-legacy-key" {
		t.Fatalf("APIKeys = %v", ep.APIKeys)
	}
}

func TestResolveProviders_CatchAllEmptyModels(t *testing.T) {
	m := Module{
		Enabled: true,
		Providers: []Provider{
			{Name: "p1"}, // no models → catch-all
		},
	}
	matches := resolveProviders(m, "any-model")
	if len(matches) != 1 {
		t.Fatalf("matches = %d, want 1 (catch-all)", len(matches))
	}
	if matches[0].Mapping.Name != "any-model" {
		t.Fatalf("mapping name = %q, want any-model (passthrough)", matches[0].Mapping.Name)
	}
}

func TestParseConfig_NewSchemaNotMigratedWhenPresent(t *testing.T) {
	// When modular fields exist, legacy fields should NOT override them.
	yaml := `upstream_base_url: https://legacy.example.com
upstream_api_key: sk-legacy
embeddings:
  enabled: true
  providers:
    - name: new
      base_url: https://new.example.com/v1
      api_keys: [sk-new]
      models:
        - name: m1
`
	if err := ParseConfig(makeReconfigurePayload(yaml)); err != nil {
		t.Fatalf("ParseConfig error = %v", err)
	}
	c := GetConfig()
	if len(c.Embeddings.Providers) != 1 || c.Embeddings.Providers[0].BaseURL != "https://new.example.com/v1" {
		t.Fatalf("new schema not used: %+v", c.Embeddings)
	}
}

func TestFilterResponseHeaders_CanonicalizesLowercaseHeader(t *testing.T) {
	upstream := map[string][]string{"content-type": {"application/json"}}
	out := filterResponseHeaders(upstream)
	if _, ok := out["Content-Type"]; !ok {
		t.Fatal("missing canonical Content-Type")
	}
	if _, ok := out["content-type"]; ok {
		t.Fatal("duplicate lowercase content-type present")
	}
}

func TestFilterResponseHeaders_DropsUnsafeHeaders(t *testing.T) {
	upstream := map[string][]string{
		"Content-Type": {"application/json"},
		"Set-Cookie":   {"session=abc"},
		"Server":       {"nginx"},
	}
	out := filterResponseHeaders(upstream)
	if len(out) != 1 {
		t.Fatalf("headers = %d, want 1 (Content-Type only)", len(out))
	}
}

func TestFilterResponseHeaders_InsertsDefaultWhenMissing(t *testing.T) {
	out := filterResponseHeaders(map[string][]string{"X-Custom": {"value"}})
	if ct, ok := out["Content-Type"]; !ok || len(ct) != 1 || ct[0] != "application/json" {
		t.Fatalf("Content-Type = %v, want [application/json]", ct)
	}
}
