# llamaswap-exporter

Prometheus exporter for llama-swap **inference** activity — tokens, throughput,
spec-decode accept rate, and model residency, per model, across a fleet.

## Why this exists

llama-swap already serves a Prometheus endpoint at `/metrics`, so this looks
redundant. It isn't. That endpoint is host and GPU telemetry only:

```
llamaswap_cpu_util_percent{core}       llamaswap_gpu_temperature_celsius
llamaswap_memory_{total,used,free}     llamaswap_gpu_{util,memory_util}_percent
llamaswap_swap_{total,used}            llamaswap_gpu_memory_{used,total}_bytes
llamaswap_load_average{interval}       llamaswap_gpu_{fan_speed,power_draw}
llamaswap_network_bytes_total          llamaswap_gpu_vram_temperature_celsius
```

Sixteen metric families, none of them about inference. All of it is already
covered by `node_exporter` and by whichever GPU exporter you run
(`dcgm-exporter` on NVIDIA, `device-metrics-exporter` / `rocm-exporter` on AMD).
**Do not scrape llama-swap's own `/metrics`** — it adds nothing and creates a
second, disagreeing source of truth for GPU temperature and power.

The inference data has no exporter, and lives in the JSON API instead. That is
what this polls.

### Why not scrape llama-server directly

The obvious alternative — what
[flox/llamacpp-monitoring](https://github.com/flox/llamacpp-monitoring) does — is
to run `llama-server --metrics` and scrape its native endpoint. That does not fit
a llama-swap deployment:

1. **Configs generally don't pass `--metrics`.** Without it,
   `/upstream/<model>/metrics` returns `501: This server does not support
   metrics endpoint. Start it with --metrics`.
2. **Scraping through `/upstream/` goes through the swap logic.** Scraping an
   unloaded model would load it. Prometheus hitting every model every 15s would
   thrash the GPUs continuously, and under `globalTTL: 0` anything a scrape
   loads stays resident forever.
3. **llama-server counters are per-process.** They reset to zero on every load
   and the series vanishes on unload, so `rate()` eats a reset on every swap and
   `up` flaps by design.

Polling llama-swap's own activity buffer sidesteps all three: stable port, no
model loading, and counters that survive swaps because this process owns them.

### What it does not replace

If you front llama-swap with a proxy such as LiteLLM, that proxy already exports
tokens, latency and deployment success/failure for traffic routed through it.
This covers what such a proxy cannot see: requests that hit llama-swap directly
— which, if your clients are pointed at llama-swap's port, is a lot of them —
plus llama.cpp-internal throughput and swap behaviour.

## Metrics

| Metric | Type | Labels | Notes |
| --- | --- | --- | --- |
| `llamaswap_requests_total` | counter | `model`, `path`, `status` | completed requests |
| `llamaswap_input_tokens_total` | counter | `model` | prompt tokens, cache included |
| `llamaswap_output_tokens_total` | counter | `model` | generated tokens |
| `llamaswap_cache_tokens_total` | counter | `model` | prompt tokens served from KV cache |
| `llamaswap_draft_tokens_total` | counter | `model` | spec decode: tokens proposed |
| `llamaswap_draft_accepted_tokens_total` | counter | `model` | spec decode: tokens accepted |
| `llamaswap_generation_tokens_per_second` | histogram | `model` | per-request gen speed |
| `llamaswap_prompt_tokens_per_second` | histogram | `model` | per-request prompt speed |
| `llamaswap_request_duration_seconds` | histogram | `model`, `path` | wall clock |
| `llamaswap_slots_total` | gauge | `model` | parallel slots the model was started with (`-np`) |
| `llamaswap_slots_busy` | gauge | `model` | slots currently decoding |
| `llamaswap_slot_context_size_tokens` | gauge | `model` | context one slot gets |
| `llamaswap_kv_cache_tokens` | gauge | `model` | tokens held across all slots |
| `llamaswap_kv_cache_capacity_tokens` | gauge | `model` | total KV capacity |
| `llamaswap_slots_supported` | gauge | `model` | 0 for backends with no `/slots` |
| `llamaswap_model_loaded` | gauge | `model` | 1 when resident |
| `llamaswap_model_state` | gauge | `model`, `state` | 1 on current state only |
| `llamaswap_models_configured` | gauge | — | models defined in config |
| `llamaswap_models_loaded` | gauge | — | models currently resident |
| `llamaswap_up` | gauge | — | last poll of the llama-swap API succeeded |
| `llamaswap_exporter_*` | mixed | — | self-instrumentation, see below |

There is deliberately **no `host` label** — Prometheus supplies `instance` from
the scrape target, and a second host identifier fights relabeling.

## Deployment

Run one exporter per llama-swap host. Either use `docker-compose.yml` as-is, or
copy the `llamaswap-exporter:` service block into an existing monitoring compose
file next to your GPU exporter, then:

```sh
docker compose up -d llamaswap-exporter
```

If you run several hosts, building once and pushing to a registry avoids
rebuilding on each — see [Building](#building).

**Hosts where llama-swap is not on 8080**: set
`LLAMASWAP_URL=http://127.0.0.1:<port>` in that host's environment, or copy
`.env.example` to `.env`. The exporter still serves `9820` on every host, so the
scrape job stays uniform.

Getting that wrong does **not** fail cleanly. If anything else answers the port
you pointed at — even with an HTTP error — the exporter starts, serves `:9820`,
and Prometheus shows the target `up`, while `llamaswap_up` sits at 0 and no
inference metrics ever arrive. `llamaswap_up`, not target health, is the signal
that the exporter is actually reaching llama-swap; that is why
`LlamaSwapAPIUnreachable` alerts on it separately from `LlamaSwapExporterDown`.

### Configuration

| Env | Default | Notes |
| --- | --- | --- |
| `LLAMASWAP_URL` | `http://127.0.0.1:8080` | per-host, if llama-swap is elsewhere |
| `EXPORTER_PORT` | `9820` | |
| `EXPORTER_ADDR` | `0.0.0.0` | must be reachable by Prometheus |
| `POLL_INTERVAL` | `15` | seconds; see *buffer lapping* below |
| `LLAMASWAP_API_KEY` | *(unset)* | bearer token, only if llama-swap requires one |
| `PAGE_SIZE` | `200` | rows per page on the modern API |
| `HTTP_TIMEOUT` | `10` | seconds |
| `LOG_LEVEL` | `INFO` | |

The exporter serves `/metrics` with no authentication, so keep `:9820` on a
trusted network or bind it to an interface Prometheus can reach and nothing else
can.

### Prometheus

`prometheus/scrape-job.yml` and `prometheus/llamaswap.rules.yml` go on your
Prometheus host, in `prometheus.yml` and your rules directory respectively.
Edit the target list in the scrape job to match your hosts. The rules are
labelled `scope: llamaswap`; add a matching Alertmanager route if you want them
delivered somewhere specific, otherwise they fall through to your default
receiver.

## How it works, and the three edge cases that matter

llama-swap exposes a **ring buffer of past requests**, capped at 1000 rows.
Turning that into counters means folding in only rows newer than the last one
seen, tracked by row id. Three cases are handled explicitly:

* **Cold start** — record the watermark, emit nothing. Backfilling 1000 stale
  rows would land an hours-old burst of tokens at exporter start.
* **Id reset** — llama-swap restarted and ids went backwards. Without detection
  the watermark would sit above every row forever and the exporter would go
  permanently silent. Counted by `llamaswap_exporter_id_resets_total`.
* **Buffer lapping** — more than 1000 requests arrived between two polls, so
  rows were overwritten before being counted. Estimated from the id gap and
  counted by `llamaswap_exporter_rows_lost_total`, which is alerted on. If it is
  ever non-zero, lower `POLL_INTERVAL` on that host. At 15s this needs a
  sustained ~66 req/s to trigger.

## Slot metrics, and why they are safe to collect

`llamaswap_slots_*` and `llamaswap_kv_cache_*` come from
`/upstream/<model>/slots`, which is the **one** place this exporter talks to
llama-server rather than to llama-swap. That is the same `/upstream/` path the
README argues against scraping above, so the distinction matters:

> The slot poller iterates **only the models in `/running`**. It never requests
> `/upstream/` for an unloaded model, so it can never trigger a load. Widening
> that loop past the running set would reintroduce exactly the swap-thrash
> problem described earlier — and under `globalTTL: 0`, anything a scrape loaded
> would stay resident forever.

Slot polling has its own shorter timeout (`SLOTS_TIMEOUT`, default 5s) because
`/slots` is answered by the llama-server event loop, and a server deep in a
decode can be slow to reply. A stall there must not delay the activity poll.

Backends that do not serve `/slots` — vLLM, or llama-server started with
`--no-slots` — report `llamaswap_slots_supported 0` and no slot gauges, so an
empty panel is explainable rather than mysterious. The gauges are removed when a
model unloads, so the series ends instead of flatlining at a stale value.

**Sampling caveat.** These are gauges sampled every `POLL_INTERVAL`, not
counters. A request shorter than the interval can fall entirely between two
samples, so `llamaswap_slots_busy` reads sustained concurrency well and bursty
short traffic poorly. For request volume use `llamaswap_requests_total`, which
is exact.

### What the slot metrics are actually for

`-np N` splits a model's context N ways. A model started with `-np 8` over
131 072 tokens gives **each request only ~16k** — a longer prompt is truncated
no matter how large the model's context looks from the config. That per-slot
ceiling is `llamaswap_slot_context_size_tokens`, and it is invisible everywhere
else.

The two saturation signals are different problems:

* `llamaswap_slots_busy / llamaswap_slots_total` near 1.0 — requests are
  queueing behind full slots. `-np` is the bottleneck, not the GPU.
* `llamaswap_kv_cache_tokens / llamaswap_kv_cache_capacity_tokens` near 1.0 —
  slots are about to evict cached prefixes, which surfaces as the cache hit
  ratio falling and prompt speed dropping.

## Two API shapes

A fleet is rarely on one llama-swap version, and the activity API changed. Both
shapes are auto-detected, and re-detected if a host is upgraded underneath the
exporter.

| | modern (`v243`+) | legacy (pre-`v243`) |
| --- | --- | --- |
| endpoint | `/api/metrics/activity?page&limit` | `/api/metrics` |
| envelope | `{"data": […], "total_pages": …}` | bare array |
| order | descending by id | ascending by id |
| paging | yes | none — returns all 1000 rows every call |
| `draft_tokens` / `draft_acc_tokens` | present | **absent** |
| `/v1/models` status field | present | absent |
| `/running` | present | present |

Two consequences:

* **Model residency comes from `/running`**, not `/v1/models`, because only
  `/running` carries state on both builds. `/v1/models` is used solely to
  enumerate the configured set for `llamaswap_models_configured`.
* **Legacy hosts report no spec-decode metrics.** Those builds predate the draft
  token fields, so `llamaswap_draft_*` stays absent there. Upgrading llama-swap
  to v243+ fixes this, and is independent of the llama.cpp builds under it — a
  binary referenced by path from `config.yaml` is untouched by a llama-swap
  upgrade.

Legacy also means transferring the full 1000-row buffer on every poll (~500 KB).
Tolerable over loopback, but another reason to upgrade.

## Useful queries

Generation throughput, p50 per model:

```promql
histogram_quantile(0.5, sum by (le, model, instance) (
  rate(llamaswap_generation_tokens_per_second_bucket[15m])))
```

Speculative decoding accept rate (v243+ hosts only) — the number that says
whether the draft model is earning its VRAM:

```promql
  sum by (model, instance) (rate(llamaswap_draft_accepted_tokens_total[1h]))
/ sum by (model, instance) (rate(llamaswap_draft_tokens_total[1h]))
```

KV cache hit ratio:

```promql
  sum by (model) (rate(llamaswap_cache_tokens_total[1h]))
/ sum by (model) (rate(llamaswap_input_tokens_total[1h]))
```

Slot utilisation — is `-np` the bottleneck?

```promql
llamaswap_slots_busy / clamp_min(llamaswap_slots_total, 1)
```

KV cache occupancy — how close slots are to evicting cached prefixes:

```promql
llamaswap_kv_cache_tokens / clamp_min(llamaswap_kv_cache_capacity_tokens, 1)
```

Swap rate — load/unload transitions per hour, the thing nothing else can see:

```promql
sum by (instance) (changes(llamaswap_model_loaded[1h]))
```

Which models actually get used, vs. which just occupy config:

```promql
sum by (model) (increase(llamaswap_requests_total[7d])) > 0
```

Error rate by model:

```promql
  sum by (model) (rate(llamaswap_requests_total{status!~"2.."}[1h]))
/ sum by (model) (rate(llamaswap_requests_total[1h]))
```

## Building

For a single host, `docker compose up -d --build` is enough. For several,
build once and push to a registry so hosts pull rather than each rebuilding:

```sh
docker build -t <registry>/llamaswap-exporter:<ver> \
             -t <registry>/llamaswap-exporter:latest .
docker push <registry>/llamaswap-exporter:<ver>
docker push <registry>/llamaswap-exporter:latest
```

Then replace `build: .` in `docker-compose.yml` with the pushed image
reference. Pin the version tag rather than `:latest`, so a host restart cannot
silently pick up a different exporter than the one your config was reasoned
about, and bump it deliberately after each rebuild.
