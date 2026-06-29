# cliproxy-embeddings-forward

A native plugin for [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) that forwards OpenAI-compatible `/v1/embeddings` requests to an upstream provider.

## How It Works

The plugin registers a Management API route at `/v0/management/embeddings`. When a request arrives, the plugin:

1. Reads the request body (an OpenAI embeddings request)
2. Forwards it to the configured upstream `/embeddings` endpoint via the host's `host.http.do` callback
3. Returns the upstream response verbatim

The upstream API key is read from plugin configuration — it is **never** relayed from the inbound `Authorization` header (which carries the CLIProxyAPI management key).

## Limitations

This is a **pure plugin** — no server-side modifications required. The tradeoffs:

| Limitation | Detail |
|---|---|
| **Path** | Route is `/v0/management/embeddings`, not `/v1/embeddings` |
| **Auth** | Client must send the management key (`Authorization: Bearer <MANAGEMENT_PASSWORD>` or `X-Management-Key`) |
| **Response sanitization** | CLIProxyAPI's `ServeManagementHTTP` applies `htmlsanitize` to JSON-looking bodies. Numeric embedding vectors are preserved (Go's `json.Number` retains precision); standard model names without `& < > " '` are unaffected. |
| **No AuthManager** | Upstream API key is managed in plugin config, not via CLIProxyAPI's auth pool (no round-robin/retry/cooldown) |

## Installation

Build the shared library:

```bash
go build -buildmode=c-shared -o embeddings-forward.dylib .
```

Place the `.dylib` (macOS) / `.so` (Linux) / `.dll` (Windows) file in your CLIProxyAPI plugins directory.

## Configuration

Add to your `config.yaml`:

```yaml
plugins:
  enabled: true
  dir: plugins
  configs:
    embeddings-forward:
      enabled: true
      upstream_base_url: "https://api.openai.com/v1"
      upstream_api_key: "sk-your-upstream-key"
      upstream_path: "/embeddings"  # optional, defaults to /embeddings
```

### Upstream URL construction

The final upstream URL is `upstream_base_url + upstream_path`:

| `upstream_base_url` | `upstream_path` | Final URL |
|---|---|---|
| `https://api.openai.com/v1` | *(default)* | `https://api.openai.com/v1/embeddings` |
| `https://aigc.example.com/v1/openai/native` | `/embeddings` | `https://aigc.example.com/v1/openai/native/embeddings` |

## Client Usage

Set your OpenAI SDK's `base_url` to the CLIProxyAPI host with the `/v0/management` prefix:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8317/v0/management",
    api_key="<MANAGEMENT_PASSWORD>",  # used as management key
)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="hello world",
)
```

The SDK sends `POST /v0/management/embeddings`, which the plugin forwards to the upstream.

## License

MIT
