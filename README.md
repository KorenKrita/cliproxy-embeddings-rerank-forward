# cliproxy-embeddings-rerank-forward

A native plugin for [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) that forwards OpenAI-compatible `/v1/embeddings` and Cohere-style rerank requests to upstream providers. Supports multiple providers per route with key failover.

## How It Works

The plugin registers two Management API routes:

| Route | Purpose |
|---|---|
| `/v0/management/embeddings` | OpenAI embeddings |
| `/v0/management/rerank` | Cohere-style rerank |

Each route is an independent **module** (`embeddings` / `rerank`). Enable either or both. Within a module, configure one or more **providers**. A request is routed by matching `body.model` against provider model mappings; the first provider with a matching model is tried first, and if its keys all return 429/5xx, the next matching provider is tried.

The upstream API key is read from plugin configuration — it is **never** relayed from the inbound `Authorization` header (which carries the CLIProxyAPI management key).

## Features

- **Two independent modules**: configure embeddings, rerank, or both.
- **Multiple providers per module**: same model across providers for failover.
- **Multiple keys per provider**: first key priority, 429/5xx switches to next key.
- **Model alias**: client-visible alias mapped to upstream real model name (body rewritten on forward).
- **Catch-all provider**: a provider with no `models` list accepts any model (passthrough, no rewrite).
- **Legacy config compat**: v0.1/v0.2 single-provider config still works — auto-migrated to the new schema.

## Installation

Build the shared library:

```bash
go build -buildmode=c-shared -o embeddings-forward.dylib .
```

Place the `.dylib` (macOS) / `.so` (Linux) / `.dll` (Windows) file in your CLIProxyAPI plugins directory.

> **Note on naming**: The repo and Go module are `cliproxy-embeddings-rerank-forward`, but the **artifact filename and plugin config key remain `embeddings-forward`** for CLIProxyAPI store-upgrade compatibility. CPA derives the plugin ID from the dylib filename, so keeping `embeddings-forward.dylib` means existing `plugins.configs.embeddings-forward:` configs keep working after upgrade with zero changes. The plugin's internal metadata `Name` (`embeddings-rerank-forward`) is display-only and does not affect config matching.

## Configuration

### Multi-provider (recommended)

```yaml
plugins:
  enabled: true
  dir: plugins
  configs:
    embeddings-forward:
      enabled: true
      embeddings:
        enabled: true
        providers:
          - name: openai
            base_url: "https://api.openai-compatible.example.com/v1"
            path: "/embeddings"            # optional, defaults to /embeddings
            api_keys: ["sk-key1", "sk-key2"]
            models:
              - name: "text-embedding-3-small"
                alias: "emb-small"          # optional; client uses this alias
              - name: "text-embedding-3-large"
          - name: backup
            base_url: "https://backup.example.com/v1"
            api_keys: ["sk-backup"]
            models:
              - name: "text-embedding-3-small"  # same model, failover provider
      rerank:
        enabled: true
        providers:
          - name: primary
            base_url: "https://api.openai-compatible.example.com/v1"
            path: "/rerank"               # optional, defaults to /rerank
            api_keys: ["key1"]
            models:
              - name: "rerank-model-8b"
                alias: "rerank-large"
          - name: secondary
            base_url: "https://secondary.example.com/v1"
            api_keys: ["key2"]
            # no models → catch-all: accepts any model, forwards as-is
```

### Flat single-provider (simple, UI-friendly)

For the common case of one embeddings provider and/or one rerank provider, use flat fields — these map directly to CPA's config form:

```yaml
plugins:
  enabled: true
  dir: plugins
  configs:
    embeddings-forward:
      enabled: true
      # Embeddings (optional — configure if you need /v1/embeddings)
      upstream_base_url: "https://api.openai-compatible.example.com/v1"
      upstream_api_key: "sk-your-key"
      upstream_path: "/embeddings"                              # optional, defaults to /embeddings
      upstream_models: "small=text-embedding-3-small, large"   # optional, comma-separated; empty = passthrough
      # Rerank (optional — configure if you need /v1/rerank)
      rerank_base_url: "https://api.openai-compatible.example.com/v1"
      rerank_api_key: "rk-your-key"
      rerank_path: "/rerank"                                    # optional, defaults to /rerank
      rerank_models: "fast=rerank-8b, rerank-0.6b"              # optional, comma-separated; empty = passthrough
```

Each module is independent — configure embeddings only, rerank only, or both. Flat fields are migrated to a single `legacy` provider per module. The `*_models` format is comma-separated entries of either `name` (alias defaults to name) or `alias=name`. Empty `*_models` means catch-all (accept any model, passthrough).

### Legacy single-provider (v0.1, still supported)

```yaml
plugins:
  enabled: true
  dir: plugins
  configs:
    embeddings-forward:
      enabled: true
      upstream_base_url: "https://api.openai-compatible.example.com/v1"
      upstream_api_key: "sk-your-key"
      upstream_path: "/embeddings"          # optional, defaults to /embeddings
```

Legacy v0.1 config (embeddings only, no models) is auto-migrated: embeddings module enabled with a single `legacy` provider, catch-all passthrough. Rerank requires flat `rerank_*` fields or the modular schema.

### Provider/model routing

1. Request arrives at `/v0/management/embeddings` or `/v0/management/rerank`.
2. Module must be `enabled`, else 404.
3. `body.model` is matched against providers in config order:
   - Provider with no `models` → catch-all, matches any model.
   - Else, match by `alias` (defaults to `name` when empty) or `name` exactly.
4. First matching provider's first key is tried. On 429/5xx, next key; if all keys fail, next matching provider.
5. If `alias` was used and differs from `name`, `body.model` is rewritten to the upstream `name` before forwarding.
6. No provider matches → 404.

### Upstream URL construction

The final upstream URL is `provider.base_url + provider.path`:

| `base_url` | `path` | Final URL |
|---|---|---|
| `https://api.example.com/v1` | *(default)* `/embeddings` | `https://api.example.com/v1/embeddings` |
| `https://api.example.com/v1` | *(default)* `/rerank` | `https://api.example.com/v1/rerank` |
| `https://api.example.com/v1` | `/custom/path` | `https://api.example.com/v1/custom/path` |

## Client Usage

Set your SDK's `base_url` to the CLIProxyAPI host with the `/v0/management` prefix.

### Embeddings

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8317/v0/management",
    api_key="<MANAGEMENT_PASSWORD>",  # used as management key
)

# Use alias "emb-small" or upstream name "text-embedding-3-small"
response = client.embeddings.create(
    model="emb-small",
    input="hello world",
)
```

### Rerank

```python
import requests

resp = requests.post(
    "http://localhost:8317/v0/management/rerank",
    headers={
        "Authorization": "Bearer <MANAGEMENT_PASSWORD>",
        "Content-Type": "application/json",
    },
    json={
        "model": "rerank-large",  # alias or upstream name
        "query": "apple",
        "documents": ["apple is a fruit", "banana is a fruit"],
    },
)
print(resp.json())
# {"object":"list","data":[{"index":0,"relevance_score":0.95},{"index":1,"relevance_score":0.03}]}
```

## Limitations

This is a **pure plugin** — no server-side modifications required. The tradeoffs:

| Limitation | Detail |
|---|---|
| **Path** | Routes are `/v0/management/embeddings` and `/v0/management/rerank`, not `/v1/...` |
| **Auth** | Client must send the management key (`Authorization: Bearer <MANAGEMENT_PASSWORD>` or `X-Management-Key`) |
| **Response sanitization** | CLIProxyAPI's `ServeManagementHTTP` applies `htmlsanitize` to JSON-looking bodies. Numeric embedding vectors are preserved (Go's `json.Number` retains precision). |
| **No AuthManager** | Provider keys are managed in plugin config, not via CLIProxyAPI's auth pool |

## License

MIT
