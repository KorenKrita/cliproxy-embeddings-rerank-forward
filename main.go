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

// PluginConfig holds upstream connection settings parsed from config_yaml.
type PluginConfig struct {
	UpstreamBaseURL string `yaml:"upstream_base_url"`
	UpstreamAPIKey  string `yaml:"upstream_api_key"`
	// UpstreamPath overrides the appended path segment. Defaults to "/embeddings".
	UpstreamPath string `yaml:"upstream_path"`
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

const registerJSON = `{"schema_version":1,"metadata":{"Name":"embeddings-forward","Version":"0.1.0","Author":"KorenKrita","GitHubRepository":"https://github.com/KorenKrita/cliproxy-embeddings-forward","ConfigFields":[{"Name":"upstream_base_url","Type":"string","Description":"Upstream OpenAI-compatible base URL, e.g. https://api.openai.com/v1"},{"Name":"upstream_api_key","Type":"string","Description":"API key for the upstream embeddings provider"},{"Name":"upstream_path","Type":"string","Description":"Path appended to upstream_base_url; defaults to /embeddings"}]},"capabilities":{"management_api":true}}`

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
		return okEnvelopeJSON(`{"routes":[{"Method":"POST","Path":"/embeddings"}]}`)
	case "management.handle":
		return HandleEmbeddings(reqBody)
	default:
		return errorEnvelope("unknown_method", "unknown method: "+method), nil
	}
}

// ParseConfig extracts upstream settings from the config_yaml payload.
// Exported for unit testing. Returns an error describing why the config
// could not be parsed, so the caller can surface it instead of silently
// succeeding with a stale (or empty) configuration.
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

// HandleEmbeddings processes an inbound management request by forwarding it
// to the configured upstream embeddings endpoint. Exported for unit testing.
func HandleEmbeddings(reqBody []byte) ([]byte, error) {
	var mgmtReq managementRequest
	if err := json.Unmarshal(reqBody, &mgmtReq); err != nil {
		return okManagementResponse(http.StatusBadRequest, nil, []byte(`{"error":{"message":"failed to decode management request","type":"invalid_request_error"}}`))
	}

	if strings.ToUpper(mgmtReq.Method) != http.MethodPost {
		return okManagementResponse(http.StatusMethodNotAllowed, nil, []byte(`{"error":{"message":"only POST is supported","type":"invalid_request_error"}}`))
	}

	if len(mgmtReq.Body) > maxRequestBodySize {
		return okManagementResponse(http.StatusRequestEntityTooLarge, nil, errorJSONBody("request body too large"))
	}

	c := GetConfig()
	baseURL := strings.TrimRight(c.UpstreamBaseURL, "/")
	apiKey := c.UpstreamAPIKey
	upstreamPath := strings.TrimSpace(c.UpstreamPath)
	if upstreamPath == "" {
		upstreamPath = "/embeddings"
	}

	if baseURL == "" {
		return okManagementResponse(http.StatusBadGateway, nil, []byte(`{"error":{"message":"upstream_base_url is not configured","type":"server_error"}}`))
	}
	if apiKey == "" {
		return okManagementResponse(http.StatusBadGateway, nil, []byte(`{"error":{"message":"upstream_api_key is not configured","type":"server_error"}}`))
	}

	if !strings.HasPrefix(upstreamPath, "/") {
		upstreamPath = "/" + upstreamPath
	}
	// Parse and validate the combined URL so a misconfigured upstream_path
	// or base URL produces a clear error instead of a request to an
	// unintended endpoint. Only http/https schemes are allowed; query,
	// fragment, and path-traversal sequences are rejected.
	parsed, errURL := url.Parse(baseURL + upstreamPath)
	if errURL != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return okManagementResponse(http.StatusBadGateway, nil, errorJSONBody("invalid upstream URL"))
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" || strings.Contains(parsed.Path, "/../") || strings.HasSuffix(parsed.Path, "/..") {
		return okManagementResponse(http.StatusBadGateway, nil, errorJSONBody("upstream URL must not contain query, fragment, or traversal sequences"))
	}
	upstreamURL := parsed.String()

	// Forward only the body; build fresh headers with the upstream API key.
	// Never relay the inbound Authorization header — it carries the management key.
	upstreamHeaders := map[string][]string{
		"Content-Type":  {"application/json"},
		"Authorization": {"Bearer " + apiKey},
	}

	hostReq := hostHTTPRequest{
		Method:  http.MethodPost,
		URL:     upstreamURL,
		Headers: upstreamHeaders,
		Body:    mgmtReq.Body,
	}

	hostResp, err := CallHost("host.http.do", hostReq)
	if err != nil {
		return okManagementResponse(http.StatusBadGateway, nil, errorJSONBody("upstream request failed"))
	}

	var httpResp hostHTTPResponse
	if err := json.Unmarshal(hostResp, &httpResp); err != nil {
		return okManagementResponse(http.StatusBadGateway, nil, errorJSONBody("failed to decode upstream response"))
	}

	respHeaders := filterResponseHeaders(httpResp.Headers)

	return okManagementResponse(httpResp.StatusCode, respHeaders, httpResp.Body)
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
