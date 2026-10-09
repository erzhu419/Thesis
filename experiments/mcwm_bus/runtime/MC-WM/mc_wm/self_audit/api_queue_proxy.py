#!/usr/bin/env python3
"""Credential-isolating LLM proxy with fair, adaptive request admission."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlsplit, urlunsplit


MAX_REQUEST_BYTES = 8 * 1024 * 1024
LOCAL_REQUEST_KEY_HEADER = "x-mcwm-request-key"
HOP_BY_HOP_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "host",
    "keep-alive",
    LOCAL_REQUEST_KEY_HEADER,
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
RESPONSE_HEADERS = {
    "content-encoding",
    "content-type",
    "openai-processing-ms",
    "retry-after",
    "x-request-id",
}
RETRYABLE_UPSTREAM_STATUSES = frozenset({408, 409})


class AdmissionQueueFull(RuntimeError):
    pass


class AdmissionQueueTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class Admission:
    wait_s: float
    active: int
    limit: int


@dataclass(frozen=True)
class ProxyResponse:
    status: int
    body: bytes
    headers: dict[str, str]
    upstream_status: int | None = None


def normalize_upstream_response(response: ProxyResponse) -> ProxyResponse:
    """Expose timeout/conflict responses as retryable gateway failures."""
    if response.status not in RETRYABLE_UPSTREAM_STATUSES:
        return response
    headers = dict(response.headers)
    headers.setdefault("retry-after", "2")
    headers["x-mcwm-upstream-status"] = str(response.status)
    return ProxyResponse(
        status=503,
        body=response.body,
        headers=headers,
        upstream_status=response.status,
    )


class AdaptiveConcurrencyGate:
    """FIFO admission with additive increase and fast overload backoff."""

    def __init__(
        self,
        *,
        minimum: int,
        initial: int,
        maximum: int,
        max_queue: int,
        increase_every: int,
        increase_below_s: float,
        decrease_above_s: float,
        cooldown_base_s: float = 2.0,
        cooldown_max_s: float = 60.0,
    ) -> None:
        if not 1 <= minimum <= initial <= maximum:
            raise ValueError("concurrency must satisfy 1 <= min <= initial <= max")
        if max_queue < 1 or increase_every < 1:
            raise ValueError("max_queue and increase_every must be positive")
        if not 0 < increase_below_s < decrease_above_s:
            raise ValueError("latency thresholds must be positive and ordered")
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.max_queue = int(max_queue)
        self.increase_every = int(increase_every)
        self.increase_below_s = float(increase_below_s)
        self.decrease_above_s = float(decrease_above_s)
        self.cooldown_base_s = float(cooldown_base_s)
        self.cooldown_max_s = float(cooldown_max_s)
        self._limit = int(initial)
        self._active = 0
        self._queue: deque[object] = deque()
        self._success_streak = 0
        self._failure_streak = 0
        self._cooldown_until = 0.0
        self._condition = threading.Condition()

    def acquire(self, timeout_s: float) -> Admission:
        if timeout_s <= 0:
            raise ValueError("queue timeout must be positive")
        started = time.monotonic()
        deadline = started + timeout_s
        ticket = object()
        with self._condition:
            if len(self._queue) >= self.max_queue:
                raise AdmissionQueueFull("LLM admission queue is full")
            self._queue.append(ticket)
            while True:
                now = time.monotonic()
                at_head = bool(self._queue and self._queue[0] is ticket)
                cooldown_s = max(0.0, self._cooldown_until - now)
                if at_head and self._active < self._limit and cooldown_s <= 0:
                    self._queue.popleft()
                    self._active += 1
                    self._condition.notify_all()
                    return Admission(
                        wait_s=now - started,
                        active=self._active,
                        limit=self._limit,
                    )
                remaining_s = deadline - now
                if remaining_s <= 0:
                    try:
                        self._queue.remove(ticket)
                    except ValueError:
                        pass
                    self._condition.notify_all()
                    raise AdmissionQueueTimeout("LLM admission queue wait timed out")
                wake_s = min(remaining_s, cooldown_s) if cooldown_s else remaining_s
                self._condition.wait(timeout=max(0.01, wake_s))

    def release(self, *, status: int, upstream_elapsed_s: float) -> dict[str, object]:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("admission release without an active request")
            self._active -= 1
            overloaded = status == 429 or status >= 500
            too_slow = upstream_elapsed_s >= self.decrease_above_s
            fast_success = (
                200 <= status < 300
                and upstream_elapsed_s <= self.increase_below_s
            )
            now = time.monotonic()
            if overloaded:
                self._success_streak = 0
                self._failure_streak += 1
                self._limit = max(
                    self.minimum,
                    int(math.ceil(self._limit * 0.75)),
                )
                cooldown_s = min(
                    self.cooldown_max_s,
                    self.cooldown_base_s * (2 ** min(self._failure_streak - 1, 5)),
                )
                self._cooldown_until = max(self._cooldown_until, now + cooldown_s)
            elif too_slow:
                self._success_streak = 0
                self._failure_streak = 0
                self._limit = max(self.minimum, self._limit - 1)
            elif fast_success:
                self._failure_streak = 0
                self._success_streak += 1
                if (
                    self._queue
                    and self._success_streak >= self.increase_every
                    and self._limit < self.maximum
                ):
                    self._limit += 1
                    self._success_streak = 0
            else:
                self._success_streak = 0
                if status < 500 and status != 429:
                    self._failure_streak = 0
            snapshot = self._snapshot_locked(now)
            self._condition.notify_all()
            return snapshot

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return self._snapshot_locked(time.monotonic())

    def _snapshot_locked(self, now: float) -> dict[str, object]:
        return {
            "active": self._active,
            "queued": len(self._queue),
            "limit": self._limit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "max_queue": self.max_queue,
            "success_streak": self._success_streak,
            "failure_streak": self._failure_streak,
            "cooldown_remaining_ms": round(
                max(0.0, self._cooldown_until - now) * 1000
            ),
        }


@dataclass
class _Flight:
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[ProxyResponse] = None


class SingleFlightCache:
    """Coalesce retries for one logical run without sharing across seeds."""

    def __init__(self, *, ttl_s: float, max_entries: int) -> None:
        if ttl_s <= 0 or max_entries < 1:
            raise ValueError("cache ttl and size must be positive")
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self._cache: OrderedDict[str, tuple[float, ProxyResponse]] = OrderedDict()
        self._flights: dict[str, _Flight] = {}
        self._lock = threading.Lock()
        self._cache_hits = 0
        self._coalesced = 0

    def claim(self, key: str) -> tuple[str, ProxyResponse | _Flight]:
        now = time.monotonic()
        with self._lock:
            self._purge_expired_locked(now)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._cache_hits += 1
                return "cached", cached[1]
            flight = self._flights.get(key)
            if flight is not None:
                self._coalesced += 1
                return "follower", flight
            flight = _Flight()
            self._flights[key] = flight
            return "leader", flight

    def complete(self, key: str, response: ProxyResponse) -> None:
        with self._lock:
            flight = self._flights.pop(key, None)
            if 200 <= response.status < 300:
                self._cache[key] = (time.monotonic() + self.ttl_s, response)
                self._cache.move_to_end(key)
                while len(self._cache) > self.max_entries:
                    self._cache.popitem(last=False)
            if flight is not None:
                flight.response = response
                flight.event.set()

    @staticmethod
    def wait(flight: _Flight, timeout_s: float) -> Optional[ProxyResponse]:
        if not flight.event.wait(timeout_s):
            return None
        return flight.response

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self._purge_expired_locked(time.monotonic())
            return {
                "entries": len(self._cache),
                "inflight_keys": len(self._flights),
                "cache_hits": self._cache_hits,
                "coalesced_waiters": self._coalesced,
            }

    def _purge_expired_locked(self, now: float) -> None:
        expired = [key for key, (until, _) in self._cache.items() if until <= now]
        for key in expired:
            self._cache.pop(key, None)


def relay_token(api_key: str) -> str:
    configured = os.environ.get("MCWM_RELAY_TOKEN")
    if configured:
        return configured
    return hmac.new(
        api_key.encode("utf-8"),
        b"mcwm-hpc-relay-v1",
        hashlib.sha256,
    ).hexdigest()


def upstream_url(base_url: str, request_path: str) -> str:
    base = urlsplit(base_url.rstrip("/"))
    request = urlsplit(request_path)
    base_path = base.path.rstrip("/")
    request_path_only = request.path
    if base_path.endswith("/v1") and request_path_only.startswith("/v1/"):
        request_path_only = request_path_only[3:]
    path = f"{base_path}/{request_path_only.lstrip('/')}"
    return urlunsplit((base.scheme, base.netloc, path, request.query, ""))


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 512

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        api_key: str,
        upstream_base: str,
        gate: AdaptiveConcurrencyGate,
        cache: SingleFlightCache,
        queue_timeout_s: float,
        upstream_timeout_s: float,
    ) -> None:
        super().__init__(server_address, ProxyHandler)
        self.api_key = api_key
        self.relay_token = relay_token(api_key)
        self.upstream_base = upstream_base
        self.gate = gate
        self.cache = cache
        self.queue_timeout_s = float(queue_timeout_s)
        self.upstream_timeout_s = float(upstream_timeout_s)
        self._metrics: Counter[str] = Counter()
        self._metrics_lock = threading.Lock()

    def increment(self, name: str) -> None:
        with self._metrics_lock:
            self._metrics[name] += 1

    def health_snapshot(self) -> dict[str, object]:
        with self._metrics_lock:
            metrics = dict(self._metrics)
        return {
            "status": "ok",
            "admission": self.gate.snapshot(),
            "deduplication": self.cache.snapshot(),
            "metrics": metrics,
        }


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MCWMRelay/2.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _write_response(self, response: ProxyResponse) -> None:
        try:
            self.send_response(response.status)
            for name, value in response.headers.items():
                self.send_header(name, value)
            if "content-type" not in {name.lower() for name in response.headers}:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response.body)
        except (BrokenPipeError, ConnectionResetError):
            self.server.increment("client_disconnects")
        self.close_connection = True

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.relay_token}"
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, expected)

    @staticmethod
    def _error(status: int, message: str, retry_after: int | None = None) -> ProxyResponse:
        headers = {"content-type": "application/json"}
        if retry_after is not None:
            headers["retry-after"] = str(retry_after)
        return ProxyResponse(
            status=status,
            body=json.dumps({"error": message}, sort_keys=True).encode("utf-8"),
            headers=headers,
        )

    def _request_key(self, body: bytes) -> str:
        supplied = self.headers.get("X-MCWM-Request-Key", "").strip()
        if not supplied or len(supplied) > 128:
            return ""
        if any(char not in "0123456789abcdefABCDEF" for char in supplied):
            return ""
        payload = supplied.encode("ascii") + b"\0" + self.path.encode("utf-8") + b"\0" + body
        return hashlib.sha256(payload).hexdigest()

    def _fetch_upstream(self, body: bytes) -> ProxyResponse:
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP_HEADERS
            and name.lower() != "accept-encoding"
        }
        headers["Authorization"] = f"Bearer {self.server.api_key}"
        request = urllib.request.Request(
            upstream_url(self.server.upstream_base, self.path),
            data=body or None,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.server.upstream_timeout_s,
                context=ssl.create_default_context(),
            ) as response:
                return normalize_upstream_response(ProxyResponse(
                    status=response.status,
                    body=response.read(),
                    headers={
                        name: value
                        for name, value in response.headers.items()
                        if name.lower() in RESPONSE_HEADERS
                    },
                ))
        except urllib.error.HTTPError as error:
            return normalize_upstream_response(ProxyResponse(
                status=error.code,
                body=error.read(),
                headers={
                    name: value
                    for name, value in error.headers.items()
                    if name.lower() in RESPONSE_HEADERS
                },
            ))
        except (urllib.error.URLError, TimeoutError, OSError):
            return self._error(502, "upstream unavailable", retry_after=2)

    def _log_request(
        self,
        *,
        response: ProxyResponse,
        started: float,
        queue_wait_s: float = 0.0,
        upstream_s: float = 0.0,
        cache_hit: bool = False,
        coalesced: bool = False,
        request_key: str = "",
    ) -> None:
        snapshot = self.server.gate.snapshot()
        print(
            json.dumps(
                {
                    "event": "proxy_request",
                    "method": self.command,
                    "path": urlsplit(self.path).path,
                    "status": response.status,
                    "upstream_status": (
                        response.upstream_status
                        if response.upstream_status is not None
                        else response.status
                    ),
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                    "queue_wait_ms": round(queue_wait_s * 1000),
                    "upstream_ms": round(upstream_s * 1000),
                    "admission_limit": snapshot["limit"],
                    "admission_active": snapshot["active"],
                    "admission_queued": snapshot["queued"],
                    "cache_hit": cache_hit,
                    "coalesced": coalesced,
                    "request_key": request_key[:12],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _handle(self) -> None:
        started = time.monotonic()
        self.server.increment("client_requests")
        if self.path in {"/healthz", "/metrics"}:
            response = ProxyResponse(
                status=200,
                body=json.dumps(
                    self.server.health_snapshot(), sort_keys=True
                ).encode("utf-8"),
                headers={"content-type": "application/json"},
            )
            self._write_response(response)
            return
        if not self.path.startswith("/v1/"):
            self._write_response(self._error(404, "unsupported path"))
            return
        if not self._authorized():
            self.server.increment("unauthorized")
            self._write_response(self._error(401, "unauthorized"))
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            self._write_response(self._error(400, "invalid content length"))
            return
        if content_length < 0 or content_length > MAX_REQUEST_BYTES:
            self._write_response(self._error(413, "request too large"))
            return
        body = self.rfile.read(content_length) if content_length else b""
        request_key = self._request_key(body)
        flight_role = "unscoped"
        flight: Optional[_Flight] = None
        if request_key:
            flight_role, claimed = self.server.cache.claim(request_key)
            if flight_role == "cached":
                self.server.increment("cache_hits")
                response = claimed
                assert isinstance(response, ProxyResponse)
                self._write_response(response)
                self._log_request(
                    response=response,
                    started=started,
                    cache_hit=True,
                    request_key=request_key,
                )
                return
            flight = claimed
            assert isinstance(flight, _Flight)
            if flight_role == "follower":
                self.server.increment("coalesced_waiters")
                response = self.server.cache.wait(
                    flight,
                    self.server.queue_timeout_s + self.server.upstream_timeout_s,
                )
                if response is None:
                    response = self._error(503, "coalesced request wait timed out", 2)
                self._write_response(response)
                self._log_request(
                    response=response,
                    started=started,
                    coalesced=True,
                    request_key=request_key,
                )
                return

        admission: Optional[Admission] = None
        response = self._error(502, "upstream unavailable", retry_after=2)
        queue_wait_s = 0.0
        upstream_s = 0.0
        try:
            admission = self.server.gate.acquire(self.server.queue_timeout_s)
            queue_wait_s = admission.wait_s
            self.server.increment("upstream_requests")
            upstream_started = time.monotonic()
            response = self._fetch_upstream(body)
            upstream_s = time.monotonic() - upstream_started
            self.server.gate.release(
                status=response.status,
                upstream_elapsed_s=upstream_s,
            )
            admission = None
        except AdmissionQueueFull:
            self.server.increment("queue_full")
            response = self._error(503, "LLM admission queue is full", retry_after=5)
        except AdmissionQueueTimeout:
            self.server.increment("queue_timeouts")
            response = self._error(503, "LLM admission queue wait timed out", retry_after=5)
        except Exception:
            self.server.increment("proxy_errors")
            response = self._error(502, "upstream unavailable", retry_after=2)
        finally:
            if admission is not None:
                self.server.gate.release(
                    status=502,
                    upstream_elapsed_s=upstream_s,
                )
            if request_key and flight_role == "leader":
                self.server.cache.complete(request_key, response)

        self.server.increment(f"status_{response.status}")
        self._write_response(response)
        self._log_request(
            response=response,
            started=started,
            queue_wait_s=queue_wait_s,
            upstream_s=upstream_s,
            request_key=request_key,
        )

    do_GET = _handle
    do_HEAD = _handle
    do_POST = _handle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18761)
    parser.add_argument("--upstream-base", required=True)
    parser.add_argument("--min-concurrency", type=int, default=2)
    parser.add_argument("--initial-concurrency", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--max-queue", type=int, default=512)
    parser.add_argument("--queue-timeout", type=float, default=900.0)
    parser.add_argument("--upstream-timeout", type=float, default=180.0)
    parser.add_argument("--increase-every", type=int, default=16)
    parser.add_argument("--increase-below", type=float, default=30.0)
    parser.add_argument("--decrease-above", type=float, default=55.0)
    parser.add_argument("--cache-ttl", type=float, default=3600.0)
    parser.add_argument("--cache-entries", type=int, default=2048)
    args = parser.parse_args()

    api_key = os.environ.get("GLM_API_KEY", "")
    if not api_key:
        raise SystemExit("GLM_API_KEY must be present in the process environment")
    gate = AdaptiveConcurrencyGate(
        minimum=args.min_concurrency,
        initial=args.initial_concurrency,
        maximum=args.max_concurrency,
        max_queue=args.max_queue,
        increase_every=args.increase_every,
        increase_below_s=args.increase_below,
        decrease_above_s=args.decrease_above,
    )
    cache = SingleFlightCache(
        ttl_s=args.cache_ttl,
        max_entries=args.cache_entries,
    )
    server = ProxyServer(
        (args.bind, args.port),
        api_key=api_key,
        upstream_base=args.upstream_base,
        gate=gate,
        cache=cache,
        queue_timeout_s=args.queue_timeout,
        upstream_timeout_s=args.upstream_timeout,
    )
    print(
        json.dumps(
            {
                "event": "proxy_started",
                "bind": args.bind,
                "port": args.port,
                "upstream": args.upstream_base,
                "admission": gate.snapshot(),
                "queue_timeout_s": args.queue_timeout,
                "upstream_timeout_s": args.upstream_timeout,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
