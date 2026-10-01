# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stoppability for the in-app LLM relay.

The user's Stop button must abort the in-flight local-model inference, not
just close the conversation with the backend. This module provides both
halves, mirroring the standalone RAG bridge's stop mechanism:

- **follow-up refusal** — ``request_stop()`` latches a process-wide flag.
  While it is set, ``relay.handle_llm_request`` refuses every new
  ``llm.request`` without any network I/O (``relay_stopped``), so no queued
  follow-up LLM call (gate fix, auto-retry, ...) is ever dispatched after a
  stop. ``clear_stop()`` is called when the user starts a fresh
  generation (composer send) or resumes a task — a new user intent reopens
  the relay, exactly like a new HTTP request reopens the bridge.
- **in-flight abort** — ``StopHTTPConnection`` connections register
  themselves while live; ``stop_inflight()`` force-closes every live
  socket, which tears down the connection to the model server immediately
  and llama.cpp aborts its decode on the disconnected connection. The
  worker thread blocked in the relay's ``urlopen`` then fails fast and
  reports ``relay_stopped``.

bpy-free, thread-safe (any thread may call; the abort operator runs on the
main thread, the relay on worker threads).
"""

import http.client
import sys
import threading
import urllib.request
from typing import Optional, Set

from mixar.config.logging_config import get_logger
from mixar.modules.common.i18n import n_

from ..constants import LOG_PREFIX

logger = get_logger(__name__)

# Windows only: closing the fd of a socket whose ``_io_refs`` is pinned by a
# live makefile (see _force_close) must go through the Winsock API — CPython
# defers ``socket.close()`` in that case, and ``ws2_32.closesocket`` is the
# exact call the deferred close would have made.
if sys.platform == "win32":
    import ctypes

    _ws2_32 = ctypes.windll.ws2_32
    _ws2_32.closesocket.argtypes = (ctypes.c_int,)
    _ws2_32.closesocket.restype = ctypes.c_int
else:
    _ws2_32 = None

# One latched stop for the process — the single-user Mixar workflow has one
# relay, and Stop means "stop THE generation".
_stop_flag = threading.Event()

_inflight_lock = threading.Lock()
_inflight: Set["StopHTTPConnection"] = set()


# ---------------------------------------------------------------------------
# Flag lifecycle (UI side)
# ---------------------------------------------------------------------------

def request_stop() -> None:
    """Latch the stop (user pressed Stop / cancelled the session).

    Latching only — it refuses follow-up ``llm.request`` frames but does not
    touch the live connection; pair with :func:`stop_inflight` to also abort
    whatever is decoding right now. Never raises."""
    global _stop_flag
    _stop_flag.set()
    logger.info("%s stop requested — in-flight + queued local LLM calls abort",
                LOG_PREFIX)


def clear_stop() -> None:
    """Release the latched stop (the user started a fresh generation)."""
    global _stop_flag
    _stop_flag.clear()


def is_stopped() -> bool:
    return _stop_flag.is_set()


def stop_inflight() -> None:
    """Latch the stop AND force-close every in-flight local-model connection.

    llama.cpp aborts its decode when the client connection drops, so this is
    what makes the "decoding" indicator die within a second instead of
    after the full generation. Safe from any thread, safe to call when
    nothing is in flight."""
    global _stop_flag
    _stop_flag.set()
    with _inflight_lock:
        conns = list(_inflight)
    for conn in conns:
        try:
            _force_close(conn)
        except Exception:  # noqa: BLE001 — closing is best-effort
            pass
    if conns:
        logger.info("%s closed %d in-flight model connection(s)",
                    LOG_PREFIX, len(conns))


def _force_close(conn: "StopHTTPConnection") -> None:
    """Tear one tracked connection down from another thread, for real.

    ``HTTPConnection.close()`` alone is not enough on Windows:
    ``http.client`` reads the response body through ``sock.makefile()``,
    which pins the socket's ``_io_refs`` — and since CPython 3.7 a
    ``sock.close()`` made while a pinned io object exists only marks the
    Python object closed and *defers* the real ``closesocket()`` until that
    io object goes away. The worker thread blocked in ``recv()`` would then
    keep waiting for the whole generation. Closing the fd through
    ``ws2_32.closesocket`` bypasses the deferral: the Winsock handle dies
    immediately, the blocked read unwinds with ``OSError`` (WSAENOTSOCK,
    10038), and the model server aborts its decode.

    On POSIX the plain ``sock.close()`` is used instead — there a blocked
    read is bounded by the request's socket timeout, so the stop remains
    correct (just slower to surface) without touching ``ws2_32``.
    """
    sock = conn.sock
    if sock is not None:
        try:
            fd = sock.fileno()       # live (unclosed) socket only
        except (ValueError, OSError):
            fd = None
        if fd is not None:
            if _ws2_32 is not None:
                _ws2_32.closesocket(fd)
            else:
                sock.close()
    conn.close()                     # deregister + http.client bookkeeping


# ---------------------------------------------------------------------------
# In-flight connection registry (relay side)
# ---------------------------------------------------------------------------

def _register(conn: "StopHTTPConnection") -> None:
    with _inflight_lock:
        _inflight.add(conn)


def _unregister(conn: "StopHTTPConnection") -> None:
    with _inflight_lock:
        _inflight.discard(conn)


class StopHTTPConnection(http.client.HTTPConnection):
    """An ``http.client`` connection that registers itself while live.

    The relay builds its opener with this as the ``http_class``; the
    connection is registered once its socket is connected and unregistered
    on ``close()`` (which ``urllib`` calls when the response finishes).
    ``stop_inflight()`` can close it from any other thread at any moment —
    an already-``closed()`` socket raises in the caller's
    ``send()``/``read()`` and the relay translates that into
    ``relay_stopped``.
    """

    def connect(self) -> None:  # noqa: D102 - http.client contract
        super().connect()
        _register(self)

    def close(self) -> None:  # noqa: D102 - http.client contract
        try:
            _unregister(self)
        finally:
            super().close()


class _StopHTTPSConnection(StopHTTPConnection, http.client.HTTPSConnection):
    """A TLS connection that is also stop-tracked: the connect() TLS wrap
    comes from HTTPSConnection (MRO order), the registration bookkeeping
    from StopHTTPConnection."""


class _StopHTTPHandler(urllib.request.HTTPHandler):
    """urllib handler wiring :class:`StopHTTPConnection` in for http URLs.

    The default ``HTTPHandler`` hardcodes ``http.client.HTTPConnection``
    in ``http_open``, so the connection class cannot be set via
    ``__init__`` — the open hooks are overridden to feed the tracked
    class into ``do_open`` instead. Being an ``HTTPHandler`` subclass,
    ``build_opener`` skips the default (untracked) http handler when
    this one is installed."""

    def http_open(self, request):  # noqa: D102 - urllib handler contract
        return self.do_open(StopHTTPConnection, request)


class _StopHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):  # noqa: D102 - urllib handler contract
        return self.do_open(_StopHTTPSConnection, request)


def build_tracked_opener(refuse_redirects: type) -> urllib.request.OpenerDirector:
    """An opener like the relay's but with tracked (stoppable) sockets."""
    return urllib.request.build_opener(
        refuse_redirects(), _StopHTTPHandler(), _StopHTTPSHandler()
    )
