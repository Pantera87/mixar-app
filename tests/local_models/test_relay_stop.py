# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Relay stoppability: flag lifecycle, in-flight abort, relay refusal.

Covers core/relay_stop.py (flag + tracked-connection registry) and the two
relay hooks it drives: pre-I/O refusal while stopped, and the
transport-failure -> ``relay_stopped`` translation.
"""

import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from mixar.modules.local_models.core import relay
from mixar.modules.local_models.core import relay_stop

URL = "http://127.0.0.1:11500/v1/chat/completions"


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


class FakeResponse:
    def __init__(self, body=b'{"ok": true}', status=200):
        self._body = body
        self.status = status
        self.headers = {"Content-Type": "application/json"}

    def read(self, n=-1):
        return self._body

    def getcode(self):
        return self.status

    def close(self):
        pass


@pytest.fixture(autouse=True)
def approved_base():
    relay.set_approved_bases(["http://127.0.0.1:11500"])
    yield
    relay.set_approved_bases([])


@pytest.fixture(autouse=True)
def _clean_stop_state():
    relay_stop.clear_stop()
    with relay_stop._inflight_lock:
        relay_stop._inflight.clear()
    yield
    relay_stop.clear_stop()
    with relay_stop._inflight_lock:
        relay_stop._inflight.clear()


def _post(url=URL):
    return {"method": "POST", "url": url, "headers": {}, "body": '{"messages": []}'}


def _run(params):
    results = []
    relay.handle_llm_request(params, results.append)
    assert len(results) == 1
    return results[0]


# ---------------------------------------------------------------------------
# Flag lifecycle
# ---------------------------------------------------------------------------

def test_flag_latch_and_clear():
    assert not relay_stop.is_stopped()
    relay_stop.request_stop()
    assert relay_stop.is_stopped()
    relay_stop.request_stop()        # idempotent
    assert relay_stop.is_stopped()
    relay_stop.clear_stop()
    assert not relay_stop.is_stopped()


def test_stop_inflight_latches_the_flag():
    relay_stop.stop_inflight()       # safe with nothing in flight
    assert relay_stop.is_stopped()


def test_stop_inflight_closes_registered_connections():
    # An unconnected connection is enough: close() is what the abort path
    # relies on, and it must leave the registry empty.
    conn = relay_stop.StopHTTPConnection("127.0.0.1", 1)
    relay_stop._register(conn)
    assert conn in relay_stop._inflight
    relay_stop.stop_inflight()
    assert relay_stop.is_stopped()
    assert conn not in relay_stop._inflight


def test_stop_inflight_aborts_a_live_request():
    """The real contract: a request blocked on the local server dies when
    stop_inflight() closes its tracked socket — from another thread."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        opener = relay_stop.build_tracked_opener(_NoRedirects)
        errors = []

        def fetch():
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/", data=b"x", method="POST")
                with opener.open(request, timeout=10) as response:
                    response.read()
            except Exception as exc:  # noqa: BLE001 — recorded, not raised
                errors.append(exc)

        worker = threading.Thread(target=fetch, daemon=True)
        worker.start()
        server.settimeout(5)
        conn, _ = server.accept()     # connect() -> registered as tracked
        try:
            time.sleep(0.3)           # worker now blocks on the response
            relay_stop.stop_inflight()
        finally:
            server.settimeout(None)
        worker.join(timeout=5)
        assert not worker.is_alive(), "worker still blocked after stop"
        assert errors, "the in-flight request did not fail after the stop"
        assert relay_stop.is_stopped()
        conn.close()
    finally:
        server.close()


# ---------------------------------------------------------------------------
# Relay hooks
# ---------------------------------------------------------------------------

def test_relay_refuses_new_calls_while_stopped(monkeypatch):
    relay_stop.request_stop()
    opened = []
    monkeypatch.setattr(relay, "_urlopen",
                        lambda req, timeout=None: opened.append(1))
    result = _run(_post())
    assert result["error"]["code"] == "relay_stopped"
    assert not opened, "a stopped relay must not perform any network I/O"


def test_clear_stop_reopens_the_relay(monkeypatch):
    relay_stop.request_stop()
    assert _run(_post())["error"]["code"] == "relay_stopped"
    relay_stop.clear_stop()
    monkeypatch.setattr(relay, "_urlopen",
                        lambda req, timeout=None: FakeResponse())
    result = _run(_post())
    assert result["status_code"] == 200
    assert result["body"] == '{"ok": true}'


def test_transport_failure_maps_to_stopped_when_latched(monkeypatch):
    def opener(req, timeout=None):
        # The stop lands while the call is in flight (stop_inflight closed
        # the tracked socket) — AFTER the relay's pre-I/O refusal ran.
        relay_stop.request_stop()
        raise urllib.error.URLError(ConnectionResetError(104, "reset"))

    monkeypatch.setattr(relay, "_urlopen", opener)
    result = _run(_post())
    assert result["error"]["code"] == "relay_stopped"


def test_same_transport_failure_is_plain_transport_when_not_stopped(monkeypatch):
    assert not relay_stop.is_stopped()

    def opener(req, timeout=None):
        raise urllib.error.URLError(ConnectionResetError(104, "reset"))

    monkeypatch.setattr(relay, "_urlopen", opener)
    result = _run(_post())
    assert result["error"]["code"] == "relay_transport"