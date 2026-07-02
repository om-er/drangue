"""The book's SRE tools, ported to drangue.

Six tools from the *Production AI Agents* companion agent (sre-agent),
re-expressed as drangue @tool functions against the same synthetic environment:
Prometheus, Loki, and Tempo for telemetry, a deploy ledger, a runbooks
directory, and the services' admin endpoints for the one action that changes
the world.

The originals sit behind a hand-built defensive wrapper (timeout, failure
classification, retry, schema check, honest failure). Here those properties
come from @tool options, so each body shrinks to its domain logic.

`remediate` is the only mutating tool. Its hard refusals (scope, forbidden
actions) live in the tool body because the model can be convinced and the tool
cannot. Whether an *allowed* remediation needs a human is rollout policy,
configured on the Agent with Autonomy, not hardcoded here.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from drangue import tool

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
LOKI_URL = os.environ.get("LOKI_URL", "http://localhost:3100")
TEMPO_URL = os.environ.get("TEMPO_URL", "http://localhost:3200")

# Data files. Point SRE_ENV at a checkout of the book's sre-agent repo to use
# its live ledger and runbooks; without it, the bundled fixtures make the
# filesystem tools (and the offline demo) self-contained.
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_ENV_ROOT = os.environ.get("SRE_ENV")
LEDGER = Path(os.environ.get("DEPLOY_LEDGER") or (
    Path(_ENV_ROOT) / "deploys.jsonl" if _ENV_ROOT else _FIXTURES / "deploys.jsonl"))
RUNBOOKS = Path(os.environ.get("RUNBOOKS_DIR") or (
    Path(_ENV_ROOT) / "env" / "runbooks" if _ENV_ROOT else _FIXTURES / "runbooks"))

IN_SCOPE = {"web", "api-gateway", "orders", "payments", "inventory", "notifications"}
_PORTS = {"web": 8081, "api-gateway": 8082, "orders": 8083,
          "payments": 8084, "inventory": 8085, "notifications": 8086}

# Log content is untrusted (customer-controlled fields land there), so sampled
# lines are scrubbed and scanned before the model reasons over them.
_INJECTION = re.compile(
    r"ignore (all )?(previous|prior)|disregard .*instructions|system prompt",
    re.IGNORECASE)


def _get_json(url: str, params: dict) -> dict:
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v != ""})
    with urllib.request.urlopen(f"{url}?{qs}", timeout=8) as resp:
        return json.loads(resp.read().decode())


@tool(timeout=10, retries=2, backoff=0.5)
def promql_query(query: str) -> str:
    """Run a PromQL instant query against the environment's Prometheus.
    Use for error rates, latency percentiles, and resource saturation."""
    body = _get_json(f"{PROM_URL}/api/v1/query", {"query": query})
    # NaN/Inf (a ratio over a zero denominator) are not valid JSON; an
    # undefined ratio is honestly "no sample", not garbage.
    samples = [
        {"labels": r.get("metric", {}), "value": float(r["value"][1])}
        for r in body["data"]["result"]
        if r.get("value") and math.isfinite(float(r["value"][1]))
    ]
    return json.dumps({"query": query, "samples": samples})


@tool(timeout=10, retries=2, backoff=0.5)
def log_search(service: str = "", query: str = "") -> str:
    """Search the last 5 minutes of logs in Loki, newest first.
    Pass a service name, or a full LogQL query for anything finer."""
    logql = query or (f'{{service="{service}"}}' if service else '{job="docker"}')
    end = int(time.time() * 1e9)
    body = _get_json(f"{LOKI_URL}/loki/api/v1/query_range", {
        "query": logql, "start": end - int(300 * 1e9), "end": end,
        "limit": 20, "direction": "backward"})
    streams = body["data"]["result"]
    raw = [line for s in streams[:3] for _, line in s.get("values", [])[:2]]
    cleaned = [re.sub(r"[\x00-\x1f]", " ", line)[:300] for line in raw]
    flags = sorted({"possible-injection" for line in cleaned if _INJECTION.search(line)})
    return json.dumps({
        "query": logql,
        "streams": len(streams),
        "lines": sum(len(s.get("values", [])) for s in streams),
        "samples": [f"[untrusted log data] {line}" for line in cleaned],
        "injection_flags": flags,
    })


@tool(timeout=10, retries=2, backoff=0.5)
def trace_lookup(service: str = "") -> str:
    """Count recent traces for a service in Tempo. Zero traces is a real
    answer: services only emit traces once instrumented."""
    body = _get_json(f"{TEMPO_URL}/api/search", {
        "tags": f"service.name={service}" if service else "", "limit": 10})
    return json.dumps({"service": service, "traces": len(body.get("traces") or [])})


@tool(timeout=5)
def deploy_history(service: str) -> str:
    """List a service's deploys from the ledger and whether one is recent
    (within 2 hours of the newest ledger entry). Recent deploys are the
    first suspect for a new incident."""
    if not LEDGER.exists():
        raise FileNotFoundError(f"deploy ledger not found: {LEDGER}")
    entries = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    if not entries:
        return json.dumps({"service": service, "deploys": [], "recent": False})
    now = max(_ts(e["ts"]) for e in entries)
    svc = [e for e in entries if e.get("service") == service]
    recent, age = False, None
    if svc:
        latest = max(svc, key=lambda e: _ts(e["ts"]))
        age = round((now - _ts(latest["ts"])).total_seconds() / 3600.0, 1)
        recent = age <= 2.0
    return json.dumps({"service": service, "deploys": svc, "recent": recent,
                       "most_recent_age_hours": age})


@tool(timeout=5)
def runbook_search(query: str) -> str:
    """Keyword-search the runbooks for known failure modes and their
    documented remediations."""
    if not RUNBOOKS.exists():
        raise FileNotFoundError(f"runbooks dir not found: {RUNBOOKS}")
    terms = [t for t in query.lower().split() if t]
    matches = []
    for path in sorted(RUNBOOKS.glob("*.md")):
        text = path.read_text()
        if any(term in text.lower() for term in terms):
            matches.append({"runbook": path.name,
                            "snippet": text.strip().splitlines()[0][:120]})
    return json.dumps({"query": query, "matches": matches})


@tool(timeout=10, reversible=True)
def remediate(service: str, action: str = "restart", idempotency_key: str = "") -> str:
    """Apply a reversible remediation to a service (currently: restart, which
    also restores its degraded state). Runs in assisted mode: a human approves
    before it executes."""
    service = service.lower().strip()
    action = action.lower().strip()

    # Hard refusals, enforced here regardless of what the model concluded.
    # These mirror the book agent's scope.yaml forbidden_actions.
    if service in ("all", "*"):
        return _refused("blind action across all services destroys the signal "
                        "that locates the root cause")
    if service not in IN_SCOPE:
        return _refused(f"'{service}' is outside the agent's scope; escalate to a human")
    if "delete" in action:
        return _refused("delete is a forbidden destructive action")
    if action != "restart":
        return _refused(f"'{action}' is not in the remediation allowlist")
    if service == "payments":
        return _refused("restarting payments during provider slowness makes "
                        "the incident worse (known-bad remediation)")

    if os.environ.get("SRE_DRY_RUN"):
        return json.dumps({"ok": True, "service": service, "action": action,
                           "executed": False, "idempotency_key": idempotency_key,
                           "note": "dry run: recorded, not applied"})

    base = os.environ.get(f"{service.upper().replace('-', '_')}_URL",
                          f"http://localhost:{_PORTS[service]}")
    req = urllib.request.Request(f"{base}/admin/reset", method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()
    return json.dumps({"ok": True, "service": service, "action": action,
                       "executed": True, "idempotency_key": idempotency_key})


def _refused(reason: str) -> str:
    return json.dumps({"ok": False, "refused": reason})


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


ALL_TOOLS = [promql_query, log_search, trace_lookup, deploy_history,
             runbook_search, remediate]
