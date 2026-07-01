// Package main implements a CLIProxyAPI native plugin that forwards
// OpenAI-compatible /v1/embeddings requests to an upstream provider.
//
// The plugin registers a Management API route under /v0/management/embeddings.
// Clients configure their OpenAI SDK base_url to point at the CLIProxyAPI
// host with the /v0/management path prefix (e.g. http://host:port/v0/management)
// so that embedding requests land on this plugin's route.
//
// The upstream API key is read from plugin configuration and never relayed
// from the inbound Authorization header (which carries the management key).
package main

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	void* ptr;
	size_t len;
} cliproxy_buffer;

typedef int (*cliproxy_host_call_fn)(void*, const char*, const uint8_t*, size_t, cliproxy_buffer*);
typedef void (*cliproxy_host_free_fn)(void*, size_t);

typedef struct {
	uint32_t abi_version;
	void* host_ctx;
	cliproxy_host_call_fn call;
	cliproxy_host_free_fn free_buffer;
} cliproxy_host_api;

typedef int (*cliproxy_plugin_call_fn)(char*, uint8_t*, size_t, cliproxy_buffer*);
typedef void (*cliproxy_plugin_free_fn)(void*, size_t);
typedef void (*cliproxy_plugin_shutdown_fn)(void);

typedef struct {
	uint32_t abi_version;
	cliproxy_plugin_call_fn call;
	cliproxy_plugin_free_fn free_buffer;
	cliproxy_plugin_shutdown_fn shutdown;
} cliproxy_plugin_api;

extern int cliproxyPluginCall(char*, uint8_t*, size_t, cliproxy_buffer*);
extern void cliproxyPluginFree(void*, size_t);
extern void cliproxyPluginShutdown(void);

static const cliproxy_host_api* stored_host;

static void store_host_api(const cliproxy_host_api* host) {
	stored_host = host;
}

static int call_host_api(const char* method, const uint8_t* request, size_t request_len, cliproxy_buffer* response) {
	if (stored_host == NULL || stored_host->call == NULL) {
		return 1;
	}
	return stored_host->call(stored_host->host_ctx, method, request, request_len, response);
}

static void free_host_buffer(void* ptr, size_t len) {
	if (stored_host != NULL && stored_host->free_buffer != NULL && ptr != NULL) {
		stored_host->free_buffer(ptr, len);
	}
}
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"unsafe"

	"gopkg.in/yaml.v3"
)

const abiVersion uint32 = 1

// maxRequestBodySize caps the forwarded embeddings request body. The host
// already enforces HTTP-layer limits, but this is defense-in-depth so a
// misconfigured host cannot let an oversized payload reach the upstream.
const maxRequestBodySize = 10 << 20 // 10 MB

// ABIVersion is exported for tests; mirrors the C ABI version negotiated at init.
var ABIVersion = abiVersion

type envelope struct {
	OK     bool            `json:"ok"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *envelopeError  `json:"error,omitempty"`
}

type envelopeError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ModelMapping maps a client-visible alias to an upstream model name.
type ModelMapping struct {
	Name  string `yaml:"name"`  // upstream real model name
	Alias string `yaml:"alias"` // optional client-visible alias; defaults to Name if empty
}

// Provider is one upstream endpoint with its own keys and model mappings.
type Provider struct {
	Name    string         `yaml:"name"`
	BaseURL string         `yaml:"base_url"`
	Path    string         `yaml:"path"` // optional, defaults per module
	APIKeys []string       `yaml:"api_keys"`
	Models  []ModelMapping `yaml:"models"`
}

// Module groups providers for one route (embeddings or rerank).
type Module struct {
	Enabled   bool       `yaml:"enabled"`
	Providers []Provider `yaml:"providers"`
}

// PluginConfig holds upstream connection settings parsed from config_yaml.
// Embeddings and Rerank are independent modules — configure either or both.
type PluginConfig struct {
	Embeddings Module `yaml:"embeddings"`
	Rerank     Module `yaml:"rerank"`
}

var (
	cfgMu sync.RWMutex
	cfg   PluginConfig
)

func main() {}

//export cliproxy_plugin_init
func cliproxy_plugin_init(host *C.cliproxy_host_api, plugin *C.cliproxy_plugin_api) C.int {
	if host == nil || plugin == nil {
		return 1
	}
	if uint32(host.abi_version) != abiVersion {
		return 1
	}
	C.store_host_api(host)
	plugin.abi_version = C.uint32_t(abiVersion)
	plugin.call = C.cliproxy_plugin_call_fn(C.cliproxyPluginCall)
	plugin.free_buffer = C.cliproxy_plugin_free_fn(C.cliproxyPluginFree)
	plugin.shutdown = C.cliproxy_plugin_shutdown_fn(C.cliproxyPluginShutdown)
	return 0
}

//export cliproxyPluginCall
func cliproxyPluginCall(method *C.char, request *C.uint8_t, requestLen C.size_t, response *C.cliproxy_buffer) C.int {
	if method == nil {
		return 1
	}
	m := C.GoString(method)
	var reqBody []byte
	if request != nil && requestLen > 0 {
		if requestLen > C.size_t(math.MaxInt32) {
			writeResponse(response, errorEnvelope("handler_error", "request payload too large"))
			return 0
		}
		reqBody = C.GoBytes(unsafe.Pointer(request), C.int(requestLen))
	}
	result, err := HandleMethod(m, reqBody)
	if err != nil {
		writeResponse(response, errorEnvelope("handler_error", err.Error()))
		return 0
	}
	writeResponse(response, result)
	return 0
}

//export cliproxyPluginFree
func cliproxyPluginFree(ptr unsafe.Pointer, len C.size_t) {
	if ptr == nil {
		return
	}
	C.free(ptr)
}

//export cliproxyPluginShutdown
func cliproxyPluginShutdown() {}

const registerJSON = `{"schema_version":1,"metadata":{"Name":"embeddings-rerank-forward","Version":"0.2.1","Author":"KorenKrita","GitHubRepository":"https://github.com/KorenKrita/cliproxy-embeddings-rerank-forward","ConfigFields":[{"Name":"upstream_base_url","Type":"string","Description":"Embeddings upstream base URL. Example: https://api.openai.com/v1"},{"Name":"upstream_api_key","Type":"string","Description":"Embeddings upstream API key. Example: sk-xxx"},{"Name":"upstream_path","Type":"string","Description":"Embeddings path, defaults to /embeddings. Usually leave empty."},{"Name":"upstream_models","Type":"string","Description":"Embeddings models, comma-separated. Format: alias=name (or just name). Leave empty to accept any model."},{"Name":"rerank_base_url","Type":"string","Description":"Rerank upstream base URL. Leave empty to disable rerank."},{"Name":"rerank_api_key","Type":"string","Description":"Rerank upstream API key."},{"Name":"rerank_path","Type":"string","Description":"Rerank path, defaults to /rerank. Usually leave empty."},{"Name":"rerank_models","Type":"string","Description":"Rerank models, comma-separated. Format: alias=name (or just name). Leave empty to accept any model."},{"Name":"embeddings","Type":"object","Description":"Advanced only: multi-provider embeddings config. Paste YAML module content (enabled + providers list). Overrides the flat fields above. See README."},{"Name":"rerank","Type":"object","Description":"Advanced only: multi-provider rerank config. Paste YAML module content (enabled + providers list). Overrides the flat fields above. See README."}]},"capabilities":{"management_api":true}}`

// HandleMethod dispatches an RPC method. Exported for unit testing.
func HandleMethod(method string, reqBody []byte) ([]byte, error) {
	switch method {
	case "plugin.register", "plugin.reconfigure":
		// ParseConfig failure is intentionally swallowed: host requires the
		// registration envelope (schema/metadata/capabilities) to load or
		// reload the plugin at all — returning an error envelope here would
		// make host reject the plugin entirely. A bad config leaves cfg at
		// its previous value: empty on first register (so HandleEmbeddings
		// returns a clear "upstream_base_url not configured" error at first
		// request), or the last working config on reconfigure (so requests
		// keep using the prior valid settings).
		_ = ParseConfig(reqBody)
		return okEnvelopeJSON(registerJSON)
	case "management.register":
		return okEnvelopeJSON(`{"routes":[{"Method":"POST","Path":"/embeddings"},{"Method":"POST","Path":"/rerank"}]}`)
	case "management.handle":
		return HandleManagement(reqBody)
	default:
		return errorEnvelope("unknown_method", "unknown method: "+method), nil
	}
}

// legacyConfig holds the v0.1/v0.2 single-provider flat fields for backward
// compat and UI-friendliness. Each module has its own flat fields; when the
// modular object is absent, flat fields migrate into a single provider.
type legacyConfig struct {
	UpstreamBaseURL string `yaml:"upstream_base_url"`
	UpstreamAPIKey  string `yaml:"upstream_api_key"`
	UpstreamPath    string `yaml:"upstream_path"`
	UpstreamModels  string `yaml:"upstream_models"`
	RerankBaseURL   string `yaml:"rerank_base_url"`
	RerankAPIKey    string `yaml:"rerank_api_key"`
	RerankPath      string `yaml:"rerank_path"`
	RerankModels    string `yaml:"rerank_models"`
}

// moduleIsEmpty reports whether a module has no providers configured.
func moduleIsEmpty(m Module) bool {
	return !m.Enabled && len(m.Providers) == 0
}

// parseModels parses a comma-separated model list into ModelMapping slices.
// Each entry is either "name" (alias defaults to name) or "alias=name".
// Empty entries are skipped. Returns nil if input is empty.
func parseModels(s string) []ModelMapping {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	var models []ModelMapping
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if idx := strings.Index(part, "="); idx >= 0 {
			alias := strings.TrimSpace(part[:idx])
			name := strings.TrimSpace(part[idx+1:])
			if alias == "" || name == "" {
				continue
			}
			models = append(models, ModelMapping{Alias: alias, Name: name})
		} else {
			models = append(models, ModelMapping{Name: part})
		}
	}
	return models
}

// ParseConfig extracts upstream settings from the config_yaml payload.
// Supports three config styles, in priority order:
//  1. Modular: embeddings/rerank objects with providers[] (multi-provider)
//  2. Flat rerank: rerank_base_url/rerank_api_key/rerank_path (single provider)
//  3. Flat legacy: upstream_base_url/upstream_api_key/upstream_path (embeddings only)
//
// Flat fields are migrated per-module: a module uses modular config if present,
// otherwise its flat fields (if any). Exported for unit testing.
func ParseConfig(reqBody []byte) error {
	var req struct {
		ConfigYAML []byte `json:"config_yaml"`
	}
	if err := json.Unmarshal(reqBody, &req); err != nil {
		return fmt.Errorf("decode config request: %w", err)
	}
	var c PluginConfig
	if err := yaml.Unmarshal(req.ConfigYAML, &c); err != nil {
		return fmt.Errorf("parse config_yaml: %w", err)
	}
	var lc legacyConfig
	yaml.Unmarshal(req.ConfigYAML, &lc) // best-effort; ignored on error

	// Embeddings: modular wins, else flat legacy fields.
	if moduleIsEmpty(c.Embeddings) && lc.UpstreamBaseURL != "" && lc.UpstreamAPIKey != "" {
		c.Embeddings = Module{
			Enabled: true,
			Providers: []Provider{{
				Name:    "legacy",
				BaseURL: lc.UpstreamBaseURL,
				Path:    lc.UpstreamPath,
				APIKeys: []string{lc.UpstreamAPIKey},
				Models:  parseModels(lc.UpstreamModels),
			}},
		}
	}

	// Rerank: modular wins, else flat rerank fields.
	if moduleIsEmpty(c.Rerank) && lc.RerankBaseURL != "" && lc.RerankAPIKey != "" {
		c.Rerank = Module{
			Enabled: true,
			Providers: []Provider{{
				Name:    "legacy",
				BaseURL: lc.RerankBaseURL,
				Path:    lc.RerankPath,
				APIKeys: []string{lc.RerankAPIKey},
				Models:  parseModels(lc.RerankModels),
			}},
		}
	}

	cfgMu.Lock()
	cfg = c
	cfgMu.Unlock()
	return nil
}

// GetConfig returns a copy of the current plugin configuration. Thread-safe.
func GetConfig() PluginConfig {
	cfgMu.RLock()
	defer cfgMu.RUnlock()
	return cfg
}

// managementRequest is the inbound request from ServeManagementHTTP.
// Field names match pluginapi.ManagementRequest (no JSON tags → Go field names).
type managementRequest struct {
	Method  string              `json:"Method"`
	Path    string              `json:"Path"`
	Headers map[string][]string `json:"Headers"`
	Query   map[string][]string `json:"Query"`
	Body    []byte              `json:"Body"`
}

// managementResponse is the outbound response for ServeManagementHTTP.
// Field names match pluginapi.ManagementResponse (no JSON tags → Go field names).
type managementResponse struct {
	StatusCode int                 `json:"StatusCode"`
	Headers    map[string][]string `json:"Headers"`
	Body       []byte              `json:"Body"`
}

// hostHTTPRequest is the payload for the host.http.do callback.
// JSON tags are lowercase to match rpcHostHTTPRequest in host_callbacks.go.
type hostHTTPRequest struct {
	Method  string              `json:"method,omitempty"`
	URL     string              `json:"url,omitempty"`
	Headers map[string][]string `json:"headers,omitempty"`
	Body    []byte              `json:"body,omitempty"`
}

// hostHTTPResponse is the result from host.http.do.
// pluginapi.HTTPResponse has no JSON tags, so Go field names are used.
type hostHTTPResponse struct {
	StatusCode int                 `json:"StatusCode"`
	Headers    map[string][]string `json:"Headers"`
	Body       []byte              `json:"Body"`
}

// HandleManagement dispatches a management.handle request to the route-specific
// handler based on the inbound Path. Exported for unit testing.
func HandleManagement(reqBody []byte) ([]byte, error) {
	var mgmtReq managementRequest
	if err := json.Unmarshal(reqBody, &mgmtReq); err != nil {
		return okManagementResponse(http.StatusBadRequest, nil, []byte(`{"error":{"message":"failed to decode management request","type":"invalid_request_error"}}`))
	}
	switch {
	case strings.HasSuffix(mgmtReq.Path, "/embeddings"):
		return HandleEmbeddings(mgmtReq)
	case strings.HasSuffix(mgmtReq.Path, "/rerank"):
		return HandleRerank(mgmtReq)
	default:
		return okManagementResponse(http.StatusNotFound, nil, errorJSONBody("unknown route: "+mgmtReq.Path))
	}
}

// HandleEmbeddings routes an embeddings request through the embeddings module.
// Exported for unit testing.
func HandleEmbeddings(mgmtReq managementRequest) ([]byte, error) {
	c := GetConfig()
	if !c.Embeddings.Enabled {
		return okManagementResponse(http.StatusNotFound, nil, errorJSONBody("embeddings module not enabled"))
	}
	return handleModule(mgmtReq, c.Embeddings, "/embeddings")
}

// HandleRerank routes a rerank request through the rerank module.
// Exported for unit testing.
func HandleRerank(mgmtReq managementRequest) ([]byte, error) {
	c := GetConfig()
	if !c.Rerank.Enabled {
		return okManagementResponse(http.StatusNotFound, nil, errorJSONBody("rerank module not enabled"))
	}
	return handleModule(mgmtReq, c.Rerank, "/rerank")
}

// resolvedProvider pairs a matched provider with its model mapping.
type resolvedProvider struct {
	Provider *Provider
	Mapping  *ModelMapping
}

// resolveProviders finds all providers that support the client model.
// Iterates providers in config order; matches alias (defaults to name when
// empty) or name exactly. A provider with no models is a catch-all: it
// matches any model (legacy passthrough) with a synthetic identity mapping
// (no alias→name rewrite). Returns one entry per matching provider.
func resolveProviders(m Module, clientModel string) []resolvedProvider {
	var matches []resolvedProvider
	for i := range m.Providers {
		p := &m.Providers[i]
		if len(p.Models) == 0 {
			// catch-all: accept any model, no rewrite
			matches = append(matches, resolvedProvider{p, &ModelMapping{Name: clientModel}})
			continue
		}
		for j := range p.Models {
			mm := &p.Models[j]
			alias := mm.Alias
			if alias == "" {
				alias = mm.Name
			}
			if alias == clientModel || mm.Name == clientModel {
				matches = append(matches, resolvedProvider{p, mm})
				break // one match per provider
			}
		}
	}
	return matches
}

// rewriteModel replaces the top-level "model" field in a JSON body with
// the upstream real name, preserving all other fields verbatim (using
// json.RawMessage so numeric fields like "dimensions" keep their exact
// representation instead of being re-encoded as float64).
func rewriteModel(body []byte, upstreamName string) ([]byte, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil, err
	}
	nameJSON, err := json.Marshal(upstreamName)
	if err != nil {
		return nil, err
	}
	obj["model"] = nameJSON
	return json.Marshal(obj)
}

// buildUpstreamURL validates and constructs the upstream URL for a provider.
// Returns empty string on validation failure.
func buildUpstreamURL(provider *Provider, defaultPath string) string {
	upstreamPath := strings.TrimSpace(provider.Path)
	if upstreamPath == "" {
		upstreamPath = defaultPath
	}
	if !strings.HasPrefix(upstreamPath, "/") {
		upstreamPath = "/" + upstreamPath
	}
	baseURL := strings.TrimRight(provider.BaseURL, "/")
	parsed, err := url.Parse(baseURL + upstreamPath)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return ""
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" || strings.Contains(parsed.Path, "/../") || strings.HasSuffix(parsed.Path, "/..") {
		return ""
	}
	return parsed.String()
}

// handleModule is the shared logic: validate, resolve all matching providers,
// then try provider→key in order (first priority, 429/5xx switches to next key,
// then next provider). Returns 404 if no provider supports the model.
func handleModule(mgmtReq managementRequest, m Module, defaultPath string) ([]byte, error) {
	if strings.ToUpper(mgmtReq.Method) != http.MethodPost {
		return okManagementResponse(http.StatusMethodNotAllowed, nil, []byte(`{"error":{"message":"only POST is supported","type":"invalid_request_error"}}`))
	}

	if len(mgmtReq.Body) > maxRequestBodySize {
		return okManagementResponse(http.StatusRequestEntityTooLarge, nil, errorJSONBody("request body too large"))
	}

	var bodyPeek struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(mgmtReq.Body, &bodyPeek); err != nil {
		return okManagementResponse(http.StatusBadRequest, nil, errorJSONBody("invalid request body"))
	}
	if bodyPeek.Model == "" {
		return okManagementResponse(http.StatusBadRequest, nil, errorJSONBody("missing model field"))
	}

	matches := resolveProviders(m, bodyPeek.Model)
	if len(matches) == 0 {
		return okManagementResponse(http.StatusNotFound, nil, errorJSONBody("model not configured: "+bodyPeek.Model))
	}

	// Provider→key failover: try each matching provider in order; within each
	// provider try keys in order. First non-429/non-5xx response wins.
	var lastResp *hostHTTPResponse
	var skipReason string // last skip reason for diagnostics
	for _, match := range matches {
		provider := match.Provider
		mapping := match.Mapping

		if len(provider.APIKeys) == 0 {
			skipReason = "provider " + provider.Name + " has no api_keys"
			continue
		}

		upstreamURL := buildUpstreamURL(provider, defaultPath)
		if upstreamURL == "" {
			skipReason = "invalid upstream URL for provider " + provider.Name
			continue
		}

		// Rewrite body.model to this provider's upstream real name (may differ
		// across providers, so rewrite per-provider, not once upfront).
		upstreamBody := mgmtReq.Body
		if bodyPeek.Model != mapping.Name {
			rewritten, err := rewriteModel(mgmtReq.Body, mapping.Name)
			if err != nil {
				skipReason = "failed to rewrite model for provider " + provider.Name
				continue
			}
			upstreamBody = rewritten
		}

		for _, apiKey := range provider.APIKeys {
			hostReq := hostHTTPRequest{
				Method: http.MethodPost,
				URL:    upstreamURL,
				Headers: map[string][]string{
					"Content-Type":  {"application/json"},
					"Authorization": {"Bearer " + apiKey},
				},
				Body: upstreamBody,
			}
			hostResp, err := CallHost("host.http.do", hostReq)
			if err != nil {
				skipReason = "upstream request failed for provider " + provider.Name + ": " + err.Error()
				continue
			}
			var httpResp hostHTTPResponse
			if err := json.Unmarshal(hostResp, &httpResp); err != nil {
				skipReason = "failed to decode upstream response for provider " + provider.Name
				continue
			}
			if httpResp.StatusCode != 429 && httpResp.StatusCode < 500 {
				respHeaders := filterResponseHeaders(httpResp.Headers)
				return okManagementResponse(httpResp.StatusCode, respHeaders, httpResp.Body)
			}
			lastResp = &httpResp
		}
	}

	if lastResp != nil {
		respHeaders := filterResponseHeaders(lastResp.Headers)
		return okManagementResponse(lastResp.StatusCode, respHeaders, lastResp.Body)
	}
	if skipReason != "" {
		// skipReason contains internal details (provider names, error messages);
		// return a generic message to the client.
		return okManagementResponse(http.StatusBadGateway, nil, errorJSONBody("upstream request failed"))
	}
	return okManagementResponse(http.StatusBadGateway, nil, errorJSONBody("all providers/keys exhausted for model "+bodyPeek.Model))
}

// filterResponseHeaders copies only safe headers from the upstream response,
// canonicalizing keys so that a lowercase "content-type" from the upstream
// does not produce a duplicate "Content-Type" entry.
func filterResponseHeaders(upstream map[string][]string) map[string][]string {
	allowed := map[string]bool{
		"Content-Type": true,
	}
	out := map[string][]string{}
	for k, v := range upstream {
		canonical := http.CanonicalHeaderKey(k)
		if allowed[canonical] {
			out[canonical] = v
		}
	}
	if _, ok := out["Content-Type"]; !ok {
		out["Content-Type"] = []string{"application/json"}
	}
	return out
}

// CallHost invokes a host callback method and returns the decoded result.
// Exported for unit testing; uses the C host API stored at init time.
func CallHost(method string, payload any) (json.RawMessage, error) {
	rawPayload, errMarshal := json.Marshal(payload)
	if errMarshal != nil {
		return nil, fmt.Errorf("marshal host callback payload %s: %w", method, errMarshal)
	}
	cMethod := C.CString(method)
	defer C.free(unsafe.Pointer(cMethod))

	var response C.cliproxy_buffer
	var requestPtr *C.uint8_t
	if len(rawPayload) > 0 {
		cPayload := C.CBytes(rawPayload)
		if cPayload == nil {
			return nil, fmt.Errorf("allocate host callback payload %s", method)
		}
		defer C.free(cPayload)
		requestPtr = (*C.uint8_t)(cPayload)
	}
	callCode := C.call_host_api(cMethod, requestPtr, C.size_t(len(rawPayload)), &response)
	var rawResponse []byte
	if response.ptr != nil && response.len > 0 {
		// C.GoBytes takes a C.int length; guard against sizes exceeding MaxInt32
		// which would silently truncate (or produce a negative length).
		if response.len > C.size_t(math.MaxInt32) {
			C.free_host_buffer(response.ptr, response.len)
			return nil, fmt.Errorf("host callback %s response too large: %d bytes", method, uint64(response.len))
		}
		rawResponse = C.GoBytes(response.ptr, C.int(response.len))
	}
	if response.ptr != nil {
		C.free_host_buffer(response.ptr, response.len)
	}
	if len(rawResponse) == 0 {
		return nil, fmt.Errorf("host callback %s returned no response, code=%d", method, int(callCode))
	}

	var env envelope
	if errUnmarshal := json.Unmarshal(rawResponse, &env); errUnmarshal != nil {
		return nil, fmt.Errorf("decode host callback envelope %s (code=%d): %w", method, int(callCode), errUnmarshal)
	}
	if callCode != 0 {
		if env.Error != nil {
			return nil, fmt.Errorf("%s: %s", env.Error.Code, env.Error.Message)
		}
		return nil, fmt.Errorf("host callback %s returned code=%d", method, int(callCode))
	}
	if !env.OK {
		if env.Error != nil {
			return nil, fmt.Errorf("%s: %s", env.Error.Code, env.Error.Message)
		}
		return nil, fmt.Errorf("host callback %s failed", method)
	}
	return append(json.RawMessage(nil), env.Result...), nil
}

func okManagementResponse(statusCode int, headers map[string][]string, body []byte) ([]byte, error) {
	if headers == nil {
		headers = map[string][]string{"Content-Type": {"application/json"}}
	}
	resp := managementResponse{
		StatusCode: statusCode,
		Headers:    headers,
		Body:       body,
	}
	return json.Marshal(envelope{OK: true, Result: json.RawMessage(mustMarshal(resp))})
}

func okEnvelopeJSON(result string) ([]byte, error) {
	return json.Marshal(envelope{OK: true, Result: json.RawMessage(result)})
}

func errorEnvelope(code, message string) []byte {
	raw, _ := json.Marshal(envelope{OK: false, Error: &envelopeError{Code: code, Message: message}})
	return raw
}

func errorJSONBody(msg string) []byte {
	type errBody struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	var e errBody
	e.Error.Message = msg
	e.Error.Type = "server_error"
	raw, _ := json.Marshal(e)
	return raw
}

func mustMarshal(v any) []byte {
	raw, _ := json.Marshal(v)
	return raw
}

func writeResponse(response *C.cliproxy_buffer, raw []byte) {
	if response == nil || len(raw) == 0 {
		return
	}
	ptr := C.CBytes(raw)
	if ptr == nil {
		return
	}
	response.ptr = ptr
	response.len = C.size_t(len(raw))
}
