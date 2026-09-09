#!/usr/bin/env python3
"""
llamaswap-exporter — Prometheus exporter for llama-swap *inference* activity.

Why this exists
---------------
llama-swap already serves a Prometheus /metrics endpoint, but it is host and GPU
telemetry only (llamaswap_cpu_util_percent, llamaswap_memory_*, llamaswap_gpu_*).
That duplicates node_exporter and dcgm-exporter/rocm-exporter, which already
scrape these hosts. It contains no inference metrics at all.

The data with no exporter lives in llama-swap's JSON API: a ring buffer of past
requests carrying per-request token counts and throughput. This polls that buffer,
folds only newly-seen rows into Prometheus counters, and exposes model residency.

Supported API shapes (auto-detected, re-detected on failure)
------------------------------------------------------------
modern (v243+)
    GET /api/metrics/activity?page=N&limit=M
    -> {"data": [...], "page", "limit", "total", "total_pages"}
    Rows DESCENDING by id. Rows carry draft_tokens / draft_acc_tokens.

legacy (pre-v243 builds)
    GET /api/metrics
    -> [ ... ]  bare array, 1000-row ring, ASCENDING by id, no draft fields,
       no pagination (returns the whole buffer every call).

Both builds serve GET /running, which is used for model residency.

The watermark
-------------
The activity buffer holds *past* requests and is capped (1000 rows on both
builds). Counters must therefore accumulate only rows newer than the last one
seen, tracked by row id. Three edge cases are handled explicitly:

  * cold start   — record the watermark, emit nothing. Backfilling 1000 stale
                   rows would land an hours-old spike at exporter start.
  * id reset     — llama-swap restarted and ids went backwards. Re-baseline.
  * buffer lap   — more requests arrived between polls than the buffer holds.
                   The lost rows are counted, not silently dropped.

There is deliberately no `host` label: Prometheus supplies `instance` from the
scrape target, and a second host identifier fights relabeling.
"""

import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from prometheus_client import Counter, Gauge, Histogram, start_http_server

LOG = logging.getLogger("llamaswap-exporter")

# --- configuration ---------------------------------------------------------

BASE_URL = os.environ.get("LLAMASWAP_URL", "http://127.0.0.1:8080").rstrip("/")
LISTEN_PORT = int(os.environ.get("EXPORTER_PORT", "9820"))
LISTEN_ADDR = os.environ.get("EXPORTER_ADDR", "0.0.0.0")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "15"))
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "10"))
API_KEY = os.environ.get("LLAMASWAP_API_KEY", "").strip()
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "200"))
# The buffer is 1000 rows, so 40 pages at limit=25 is the whole thing. This is a
# safety stop for the paging loop, not a tuning knob.
MAX_PAGES = int(os.environ.get("MAX_PAGES", "40"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
ENABLE_SLOTS = os.environ.get("ENABLE_SLOTS", "true").lower() not in ("0", "false", "no")
# Slot polling gets its own, shorter timeout: /slots is answered by the
# llama-server event loop, so a server deep in a decode can be slow to reply.
# A stall there must not hold up the activity poll behind it.
SLOTS_TIMEOUT = float(os.environ.get("SLOTS_TIMEOUT", "5"))

# Observed generation speeds span ~4 tok/s (large dense models) to ~140 tok/s
# (small vision models). Buckets are widened at both ends for spec-decode and
# for 100B-class models.
GEN_TPS_BUCKETS = (1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200,
                   300, 500, 1000, float("inf"))
# Prompt processing spans ~26 tok/s to ~9500 tok/s across the fleet.
PROMPT_TPS_BUCKETS = (50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 7500,
                      10000, 15000, 25000, 50000, float("inf"))
DURATION_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600,
                    1800, float("inf"))

# --- metrics ---------------------------------------------------------------
# prometheus_client appends _total to Counter names itself, so these are
# declared without the suffix and exported with it.

REQUESTS = Counter(
    "llamaswap_requests",
    "Requests completed through llama-swap, by model, request path and HTTP status.",
    ["model", "path", "status"],
)
INPUT_TOKENS = Counter(
    "llamaswap_input_tokens",
    "Prompt tokens submitted, including tokens served from cache.",
    ["model"],
)
OUTPUT_TOKENS = Counter(
    "llamaswap_output_tokens", "Tokens generated.", ["model"]
)
CACHE_TOKENS = Counter(
    "llamaswap_cache_tokens",
    "Prompt tokens served from KV cache rather than reprocessed.",
    ["model"],
)
DRAFT_TOKENS = Counter(
    "llamaswap_draft_tokens",
    "Speculative decoding: tokens proposed by the draft model. "
    "Only reported by builds that expose draft_tokens.",
    ["model"],
)
DRAFT_ACCEPTED = Counter(
    "llamaswap_draft_accepted_tokens",
    "Speculative decoding: proposed tokens accepted by the target model. "
    "Divide by llamaswap_draft_tokens_total for the accept rate.",
    ["model"],
)
GEN_TPS = Histogram(
    "llamaswap_generation_tokens_per_second",
    "Per-request generation throughput reported by llama-server.",
    ["model"],
    buckets=GEN_TPS_BUCKETS,
)
PROMPT_TPS = Histogram(
    "llamaswap_prompt_tokens_per_second",
    "Per-request prompt processing throughput reported by llama-server.",
    ["model"],
    buckets=PROMPT_TPS_BUCKETS,
)
REQ_DURATION = Histogram(
    "llamaswap_request_duration_seconds",
    "Wall-clock duration of completed requests, as measured by llama-swap.",
    ["model", "path"],
    buckets=DURATION_BUCKETS,
)

MODEL_LOADED = Gauge(
    "llamaswap_model_loaded",
    "1 if the model currently has a running llama-server process, else 0.",
    ["model"],
)
MODEL_STATE = Gauge(
    "llamaswap_model_state",
    "1 for the model's current llama-swap state, 0 for every other state.",
    ["model", "state"],
)
SLOTS_TOTAL = Gauge(
    "llamaswap_slots_total",
    "Parallel request slots the loaded llama-server was started with (-np).",
    ["model"],
)
SLOTS_BUSY = Gauge(
    "llamaswap_slots_busy",
    "Slots currently processing a request. Divide by llamaswap_slots_total for "
    "concurrency utilisation.",
    ["model"],
)
SLOT_CONTEXT_SIZE = Gauge(
    "llamaswap_slot_context_size_tokens",
    "Context window of a single slot, in tokens. With -np N the model's total "
    "context is split N ways, so this is what one request actually gets.",
    ["model"],
)
KV_TOKENS = Gauge(
    "llamaswap_kv_cache_tokens",
    "Tokens currently held across all slots' KV cache.",
    ["model"],
)
KV_CAPACITY = Gauge(
    "llamaswap_kv_cache_capacity_tokens",
    "Total KV cache capacity across all slots, in tokens. Divide "
    "llamaswap_kv_cache_tokens by this for cache occupancy.",
    ["model"],
)
SLOTS_SUPPORTED = Gauge(
    "llamaswap_slots_supported",
    "1 if the loaded backend answers /slots. 0 for backends that do not expose "
    "it (vLLM, or llama-server started with --no-slots), which is why the slot "
    "gauges are absent for that model rather than zero.",
    ["model"],
)

MODELS_CONFIGURED = Gauge(
    "llamaswap_models_configured", "Number of models defined in the llama-swap config."
)
MODELS_LOADED_TOTAL = Gauge(
    "llamaswap_models_loaded", "Number of models currently resident."
)

UP = Gauge("llamaswap_up", "1 if the last poll of the llama-swap API succeeded.")
API_FLAVOR = Gauge(
    "llamaswap_exporter_api_flavor",
    "1 for the activity API shape currently in use.",
    ["flavor"],
)
POLL_DURATION = Gauge(
    "llamaswap_exporter_poll_duration_seconds", "Duration of the last poll cycle."
)
LAST_SUCCESS = Gauge(
    "llamaswap_exporter_last_success_timestamp_seconds",
    "Unix timestamp of the last fully successful poll.",
)
ROWS_INGESTED = Counter(
    "llamaswap_exporter_rows_ingested", "Activity rows folded into counters."
)
ROWS_LOST = Counter(
    "llamaswap_exporter_rows_lost",
    "Activity rows the exporter provably missed because the ring buffer lapped "
    "between polls. Non-zero means POLL_INTERVAL is too long for the request rate.",
)
ID_RESETS = Counter(
    "llamaswap_exporter_id_resets",
    "Times the activity row id went backwards, indicating a llama-swap restart.",
)
POLL_ERRORS = Counter(
    "llamaswap_exporter_poll_errors", "Failed polls, by endpoint.", ["endpoint"]
)

# --- HTTP ------------------------------------------------------------------


class NotFound(Exception):
    """The endpoint returned 404 — used to fall back to the legacy API shape."""


def get_json(path, timeout=None):
    url = BASE_URL + path
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    if API_KEY:
        req.add_header("Authorization", "Bearer " + API_KEY)
    try:
        with urllib.request.urlopen(req, timeout=timeout or HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise NotFound(path) from exc
        raise


# --- row normalisation -----------------------------------------------------


def normalise(row):
    """Flatten one activity row into the fields the metrics need.

    Missing keys are tolerated: the legacy build omits draft_tokens entirely,
    and -1 is llama-swap's 'not applicable' sentinel for the draft fields on
    builds that do report them. Neither may be folded into a counter.
    """
    tokens = row.get("tokens") or {}

    def sentinel(key):
        value = tokens.get(key)
        if value is None or value < 0:
            return None
        return value

    return {
        "id": row["id"],
        "model": row.get("model") or "unknown",
        "path": row.get("req_path") or "unknown",
        "status": str(row.get("resp_status_code", 0)),
        "input": sentinel("input_tokens"),
        "output": sentinel("output_tokens"),
        "cache": sentinel("cache_tokens"),
        "draft": sentinel("draft_tokens"),
        "draft_acc": sentinel("draft_acc_tokens"),
        "gen_tps": tokens.get("tokens_per_second"),
        "prompt_tps": tokens.get("prompt_per_second"),
        "duration_ms": row.get("duration_ms"),
    }


# --- collector -------------------------------------------------------------


class Collector:
    def __init__(self):
        self.watermark = None      # highest row id already folded into counters
        self.flavor = None         # "modern" | "legacy"
        self._known_states = {}    # model -> last state, so stale states get zeroed
        self._running = set()      # models with a live process, set by poll_models
        self._slot_models = set()  # models currently carrying slot gauges

    # -- activity ----------------------------------------------------------

    def fetch_modern(self):
        """Page the v243+ activity API until we reach the watermark.

        Rows come back newest-first, so paging forward walks backwards in time
        and we stop as soon as we see something already counted.
        """
        collected = []
        reached_watermark = self.watermark is None
        highest = None

        # On a cold start every row is discarded anyway — only the highest id
        # matters — so stop after one page instead of paging the whole buffer.
        max_pages = 1 if self.watermark is None else MAX_PAGES

        for page in range(1, max_pages + 1):
            query = urllib.parse.urlencode({"page": page, "limit": PAGE_SIZE})
            payload = get_json("/api/metrics/activity?" + query)
            rows = payload.get("data") or []
            if not rows:
                reached_watermark = True
                break

            if highest is None:
                highest = max(r["id"] for r in rows)
                # A restart resets the id sequence. Without this check the
                # watermark would sit above every row forever and the exporter
                # would go permanently silent.
                if self.watermark is not None and highest < self.watermark:
                    LOG.warning(
                        "activity id went backwards (%s < %s) — llama-swap restarted; re-baselining",
                        highest, self.watermark,
                    )
                    ID_RESETS.inc()
                    self.watermark = None
                    reached_watermark = True

            stop = False
            for row in rows:
                if self.watermark is not None and row["id"] <= self.watermark:
                    stop = True
                    reached_watermark = True
                    break
                collected.append(row)
            if stop:
                break

            total_pages = payload.get("total_pages") or 1
            if page >= total_pages:
                reached_watermark = True
                break
        else:
            # Ran out of pages without meeting the watermark. On a cold start
            # that is the deliberate single-page stop above, not a gap.
            if self.watermark is None:
                reached_watermark = True

        return collected, reached_watermark

    def fetch_legacy(self):
        """The legacy build returns the entire 1000-row ring, ascending, uncapped."""
        rows = get_json("/api/metrics")
        if not isinstance(rows, list):
            raise ValueError("legacy /api/metrics did not return a list")
        if not rows:
            return [], True

        highest = max(r["id"] for r in rows)
        if self.watermark is not None and highest < self.watermark:
            LOG.warning(
                "activity id went backwards (%s < %s) — llama-swap restarted; re-baselining",
                highest, self.watermark,
            )
            ID_RESETS.inc()
            self.watermark = None

        if self.watermark is None:
            return rows, True

        oldest = min(r["id"] for r in rows)
        # If the oldest row still in the buffer is newer than what we last saw,
        # everything between was overwritten before we got to it.
        reached_watermark = oldest <= self.watermark + 1
        return [r for r in rows if r["id"] > self.watermark], reached_watermark

    def fetch_rows(self):
        """Return (rows ascending by id, reached_watermark), detecting API shape."""
        if self.flavor != "legacy":
            try:
                rows, reached = self.fetch_modern()
                self.flavor = "modern"
            except NotFound:
                LOG.info("/api/metrics/activity absent — falling back to legacy /api/metrics")
                self.flavor = "legacy"
                rows, reached = self.fetch_legacy()
        else:
            try:
                rows, reached = self.fetch_legacy()
            except NotFound:
                # llama-swap was upgraded under us.
                LOG.info("legacy /api/metrics absent — re-detecting API shape")
                self.flavor = None
                rows, reached = self.fetch_modern()
                self.flavor = "modern"

        API_FLAVOR.labels(flavor="modern").set(1 if self.flavor == "modern" else 0)
        API_FLAVOR.labels(flavor="legacy").set(1 if self.flavor == "legacy" else 0)
        rows.sort(key=lambda r: r["id"])
        return rows, reached

    def poll_activity(self):
        rows, reached_watermark = self.fetch_rows()
        if not rows:
            return

        cold_start = self.watermark is None

        if not reached_watermark and not cold_start:
            # ids are dense in practice, so the gap size is a good estimate of
            # how many completed requests were never counted.
            lost = max(0, rows[0]["id"] - self.watermark - 1)
            if lost:
                LOG.warning(
                    "ring buffer lapped: ~%d rows lost between polls (lower POLL_INTERVAL)",
                    lost,
                )
                ROWS_LOST.inc(lost)

        self.watermark = rows[-1]["id"]

        if cold_start:
            # Baseline only. These rows predate the exporter; counting them would
            # inject an arbitrarily old burst at startup.
            LOG.info("cold start: baselining at activity id %s (%d rows skipped)",
                     self.watermark, len(rows))
            return

        for raw in rows:
            row = normalise(raw)
            model = row["model"]

            REQUESTS.labels(model=model, path=row["path"], status=row["status"]).inc()

            if row["input"]:
                INPUT_TOKENS.labels(model=model).inc(row["input"])
            if row["output"]:
                OUTPUT_TOKENS.labels(model=model).inc(row["output"])
            if row["cache"]:
                CACHE_TOKENS.labels(model=model).inc(row["cache"])
            if row["draft"]:
                DRAFT_TOKENS.labels(model=model).inc(row["draft"])
            if row["draft_acc"]:
                DRAFT_ACCEPTED.labels(model=model).inc(row["draft_acc"])

            # Throughput is only meaningful for requests that actually ran; a 400
            # carries a zero rate that would drag every quantile down.
            if row["gen_tps"]:
                GEN_TPS.labels(model=model).observe(row["gen_tps"])
            if row["prompt_tps"]:
                PROMPT_TPS.labels(model=model).observe(row["prompt_tps"])
            if row["duration_ms"]:
                REQ_DURATION.labels(model=model, path=row["path"]).observe(
                    row["duration_ms"] / 1000.0
                )

        ROWS_INGESTED.inc(len(rows))

    # -- residency ---------------------------------------------------------

    def poll_models(self):
        """Model residency from /running, which both builds serve.

        /v1/models carries a status field on v243+ but not on the legacy build,
        so it is used only to enumerate the configured set.
        """
        try:
            catalog = get_json("/v1/models").get("data") or []
            configured = [m["id"] for m in catalog if m.get("id")]
        except (NotFound, KeyError, TypeError):
            configured = []
        MODELS_CONFIGURED.set(len(configured))

        running = get_json("/running").get("running") or []
        states = {}
        for entry in running:
            model = entry.get("model")
            if model:
                states[model] = entry.get("state") or "unknown"

        MODELS_LOADED_TOTAL.set(len(states))
        # Only these are polled for slots — never an unloaded model, which is
        # what keeps /upstream/ scraping from triggering a load.
        self._running = {m for m, st in states.items() if st == "ready"}

        for model in set(configured) | set(states) | set(self._known_states):
            state = states.get(model, "unloaded")
            MODEL_LOADED.labels(model=model).set(1 if state == "ready" else 0)
            # Zero the previous state so only one state series per model is ever 1;
            # without this a model that moves ready -> unloaded reports both.
            previous = self._known_states.get(model)
            if previous and previous != state:
                MODEL_STATE.labels(model=model, state=previous).set(0)
            MODEL_STATE.labels(model=model, state=state).set(1)
            self._known_states[model] = state

    # -- slots -------------------------------------------------------------

    def poll_slots(self):
        """Per-slot state for loaded models, via /upstream/<model>/slots.

        This is the one place the exporter talks to llama-server rather than to
        llama-swap, and it is safe *only* because it iterates `self._running`.
        Requesting /upstream/ for an unloaded model would make llama-swap load
        it; under `globalTTL: 0` that model would then stay resident forever.
        Never widen this loop past the running set.
        """
        if not ENABLE_SLOTS:
            return

        seen = set()
        for model in sorted(self._running):
            path = "/upstream/" + urllib.parse.quote(model, safe="") + "/slots"
            try:
                slots = get_json(path, timeout=SLOTS_TIMEOUT)
            except NotFound:
                # vLLM and other non-llama.cpp backends have no /slots.
                SLOTS_SUPPORTED.labels(model=model).set(0)
                seen.add(model)
                continue
            except Exception as exc:
                LOG.debug("slots poll failed for %s: %s", model, exc)
                continue

            if not isinstance(slots, list) or not slots:
                # --no-slots answers 501 with a JSON error object, not a list.
                SLOTS_SUPPORTED.labels(model=model).set(0)
                seen.add(model)
                continue

            busy = sum(1 for s in slots if s.get("is_processing"))
            ctx = [s.get("n_ctx") or 0 for s in slots]
            held = sum(s.get("n_prompt_tokens") or 0 for s in slots)

            SLOTS_SUPPORTED.labels(model=model).set(1)
            SLOTS_TOTAL.labels(model=model).set(len(slots))
            SLOTS_BUSY.labels(model=model).set(busy)
            SLOT_CONTEXT_SIZE.labels(model=model).set(ctx[0] if ctx else 0)
            KV_CAPACITY.labels(model=model).set(sum(ctx))
            KV_TOKENS.labels(model=model).set(held)
            seen.add(model)

        # Slots exist only while a model is loaded, so the series should end
        # when it unloads rather than flatline at a stale value.
        for model in self._slot_models - seen:
            for metric in (SLOTS_TOTAL, SLOTS_BUSY, SLOT_CONTEXT_SIZE,
                           KV_TOKENS, KV_CAPACITY, SLOTS_SUPPORTED):
                try:
                    metric.remove(model)
                except KeyError:
                    pass
        self._slot_models = seen

    # -- cycle -------------------------------------------------------------

    def poll(self):
        started = time.monotonic()
        ok = True

        # models before slots: poll_models populates the running set that
        # poll_slots iterates.
        for name, fn in (("activity", self.poll_activity),
                         ("models", self.poll_models),
                         ("slots", self.poll_slots)):
            try:
                fn()
            except Exception as exc:  # keep serving the last good values
                ok = False
                POLL_ERRORS.labels(endpoint=name).inc()
                LOG.warning("poll of %s failed: %s", name, exc)

        POLL_DURATION.set(time.monotonic() - started)
        UP.set(1 if ok else 0)
        if ok:
            LAST_SUCCESS.set(time.time())


def main():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    LOG.info(
        "llamaswap-exporter: polling %s every %ss, serving %s:%d/metrics as %s",
        BASE_URL, POLL_INTERVAL, LISTEN_ADDR, LISTEN_PORT, socket.gethostname(),
    )

    start_http_server(LISTEN_PORT, addr=LISTEN_ADDR)
    UP.set(0)

    while not stop.is_set():
        collector_poll_start = time.monotonic()
        COLLECTOR.poll()
        # Drift-free pacing: sleep the remainder of the interval, not the whole
        # interval on top of however long the poll took.
        remaining = POLL_INTERVAL - (time.monotonic() - collector_poll_start)
        stop.wait(max(1.0, remaining))

    LOG.info("shutting down")
    return 0


COLLECTOR = Collector()

if __name__ == "__main__":
    sys.exit(main())
