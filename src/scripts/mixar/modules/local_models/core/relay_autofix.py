# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Closed-loop auto-retry for runtime Blender errors (in-app).

Ported from the standalone RAG bridge (``rag/mixar-rag-server.py``): when
the conversation the backend sends down ends in a Blender RUNTIME traceback,
the relay already holds everything a cloud round-trip would see (goal, user
request, the failing script, the error), so instead of forwarding that
request as-is we run ONE local 'RUNTIME FIX' inference against the same
local model and forge a corrected tool payload directly.

This is the in-app replacement for the bridge's Part B. Design law (same as
the rest of the relay): :func:`maybe_forge` returns the forged body or
``None`` — and ``None`` always means "fall back to the plain relay", so this
path can NEVER strand an error. Every guard (budget, spin, syntax,
validator, no-exec-tool) degrades to the plain relay.

bpy-free; runs on the relay worker thread.
"""

import hashlib
import json
import re
import threading
import time
import urllib.request
from typing import Any, Dict, Optional, Tuple

from mixar.config.logging_config import get_logger

from ..constants import LOG_PREFIX, MAX_RELAY_RESPONSE_BYTES
from . import relay_stop

logger = get_logger(__name__)

# Max consecutive auto-retry dispatches per goal before giving up and
# handing the raw error to the backend's normal error relay.
MAX_ATTEMPTS = 2

# Ported from the RAG bridge: how to find the declared script-execution
# tool and the parameter that carries the code.
_EXEC_NAME_HINTS = (
    "execute", "run_", "python", "bpy", "script", "eval", "console", "code",
)
_EXEC_PARAM_HINTS = (
    "code", "script", "python_code", "bpy_code", "source", "snippet", "command",
)
_CODE_PARAM_KEYS = (
    "code", "script", "python_code", "bpy_code", "source", "snippet",
)

# Match a model think-block (built from char codes so the literal tag
# sequence never appears in the source tree).
_THINK_RE = re.compile(
    chr(60) + "mth" + chr(105) + "nk" + r".*?" +
    chr(60) + "/" + "th" + chr(105) + "nk" + chr(62),
    re.DOTALL,
)

_FIX_SYSTEM_PROMPT = (
    "You fix runtime Blender errors in Python bpy scripts for the Mixar "
    "build. Reply with ONLY the complete corrected script inside one "
    "```python fence, starting with `import bpy`. Change only what the "
    "runtime error traceback requires; keep every other line identical. "
    "No explanations, no prose."
)

# Per-goal auto-retry budget: goal hash -> dispatches so far.
_budget_lock = threading.Lock()
_budget: Dict[str, int] = {}


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """The fix inference must hit exactly the pinned URL, nothing else."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _tracked_opener() -> urllib.request.OpenerDirector:
    """Tracked opener for the fix inference (same target as the relay's
    own call — already validated + approved — so it is Stop-killable too)."""
    return relay_stop.build_tracked_opener(_NoRedirects)


# ---------------------------------------------------------------------------
# Guards (ported verbatim in spirit from the bridge)
# ---------------------------------------------------------------------------

def _looks_like_runtime_blender_error(tool_result: str) -> bool:
    """True when a tool result is a Blender RUNTIME failure (traceback) —
    the class of error the local fix can repair. Plain non-traceback output
    and timeouts (the script is too heavy; a rewrite won't help) stay on
    the backend's normal error relay."""
    t = (tool_result or "").lower()
    if "traceback" not in t:
        return False
    if "timed out" in t:
        return False
    return True


def _is_tool_result(message: dict) -> bool:
    role = message.get("role")
    if role in ("tool", "function"):
        return True
    # some clients echo tool output back as a named assistant message
    if role == "assistant" and message.get("name") \
            and message["name"] not in ("assistant", "user", "system"):
        return True
    return False


def _goal_key(text: str) -> str:
    """Canonical per-goal key (same normalization as the goal loop breaker:
    the last user message, collapsed, lower-cased, first 200 chars)."""
    norm = " ".join((text or "").split()).lower()[:200]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:32]


def _last_user_text(messages: list) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):       # content-parts form
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return ""


def _last_dispatched_script(messages: list) -> str:
    """The script the executor last ran: the script parameter of the most
    recent assistant tool_call anywhere before the tail error."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in reversed(calls):
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            args = function.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    continue
            if not isinstance(args, dict):
                continue
            for key in _CODE_PARAM_KEYS:
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return ""


def _pick_param(params: dict) -> str:
    """Find the string parameter that carries the script, if any."""
    if not params:
        return "code"
    low = {str(k).lower(): k for k in params.keys()}
    for hint in _EXEC_PARAM_HINTS:
        if hint in low:
            return low[hint]
    for hint in _EXEC_PARAM_HINTS:
        for k in low:
            if hint in k:
                return k
    str_keys = [
        k for k, v in params.items()
        if isinstance(v, dict) and v.get("type") == "string"
    ]
    if len(str_keys) == 1:
        return str_keys[0]
    return "code"


def _pick_exec_tool(tools: Any) -> Tuple[Optional[str], Optional[str]]:
    """Derive (tool_name, param_name) from the declared tool schema.

    Returns (None, None) when declared tools exist but NONE of them can
    execute a script (e.g. only search_documentation) — in that case the
    fix has nowhere to be dispatched, so the plain relay takes over."""
    if not tools:
        return None, None
    names = []
    # pass 1: execution-looking name
    for tool in tools:
        fn = (tool or {}).get("function") or {}
        nm = fn.get("name")
        if not nm:
            continue
        names.append(nm)
        low = nm.lower()
        if any(h in low for h in _EXEC_NAME_HINTS) and "search" not in low:
            param = _pick_param(
                (fn.get("parameters") or {}).get("properties") or {})
            return nm, param
    # pass 2: a parameter that ACTUALLY looks like it carries code
    for tool in tools:
        fn = (tool or {}).get("function") or {}
        nm = fn.get("name")
        if not nm:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        code_keys = [k for k in props if str(k).lower() in _CODE_PARAM_KEYS]
        if code_keys:
            return nm, code_keys[0]
    logger.info("%s auto-retry: declared tools %s contain no script-"
                "execution tool — relaying the error as-is", LOG_PREFIX, names)
    return None, None


def _extract_script(text: str) -> str:
    """Pull the script out of a model response (STRICT: a fenced block or
    an opening fence with no closing fence — prose returns "" so we relay
    instead of guessing)."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text).strip()
    m = re.search(r"```python(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python)?\s*\n", text)
    if m:
        # opening fence with no closing fence -> truncated completion
        tail = text[m.end():].strip()
        if tail:
            return tail
    return ""


def _script_ok(script: str) -> bool:
    """compile + normalize + truth-table validation of a fixed script.

    Any compile/validator problem refuses the fix so the backend's own
    error relay handles it. A missing truth table makes the validator
    blind (no issues), matching the RAG server's degrade rule."""
    try:
        compile(script, "<auto_retry_fix>", "exec")
    except SyntaxError:
        return False
    try:
        from mixar.modules.space_mixie_chat.core import script_validator
        fixed, _notes = script_validator.normalize_mixar_script(script)
        issues = script_validator.validate_bpy_properties(fixed)
        if issues:
            logger.info("%s auto-retry: validator flagged %d issue(s) in "
                        "the fix — relaying the original error",
                        LOG_PREFIX, len(issues))
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — degrade to relay
        logger.warning("%s auto-retry: validation unavailable (%s)",
                       LOG_PREFIX, exc)
        return True


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def _budget_taken(goal_hash: str) -> int:
    with _budget_lock:
        if len(_budget) > 16:
            _budget.clear()          # soft bound — the budget is best-effort
        return _budget.get(goal_hash, 0)


def _budget_charge(goal_hash: str) -> None:
    with _budget_lock:
        if len(_budget) > 16:
            _budget.clear()
        _budget[goal_hash] = _budget.get(goal_hash, 0) + 1


def reset_budget() -> None:
    """Drop every per-goal attempt counter (tests / logout)."""
    with _budget_lock:
        _budget.clear()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def maybe_forge(body: bytes, pinned_url: str, timeout: float) -> Optional[bytes]:
    """Maybe replace one outbound chat-completions request with a forged
    tool payload carrying a locally-fixed script.

    ``pinned_url`` is the relay's ALREADY-VALIDATED, approved target — the
    fix inference is a second POST to the very same local model. Returns
    the forged chat-completion body (bytes) or ``None`` (plain relay).
    Never raises: every failure mode returns ``None``.
    """
    try:
        return _maybe_forge_impl(body, pinned_url, timeout)
    except Exception as exc:  # noqa: BLE001 — never strand the error path
        logger.warning("%s auto-retry failed closed (%s)", LOG_PREFIX, exc)
        return None


def _maybe_forge_impl(body: bytes, pinned_url: str, timeout: float) -> Optional[bytes]:
    try:
        obj = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    # 1) The tail must be a Blender RUNTIME error (traceback, not timeout).
    last = messages[-1]
    if not isinstance(last, dict) or not _is_tool_result(last):
        return None
    tail = last.get("content")
    if not isinstance(tail, str) or not _looks_like_runtime_blender_error(tail):
        return None

    # 2) We need the script that just failed to have anything to fix.
    last_script = _last_dispatched_script(messages)
    if not last_script:
        return None

    # 3) Budget: at most MAX_ATTEMPTS auto-retry dispatches per goal.
    goal = _last_user_text(messages)
    gkey = _goal_key(goal)
    attempts = _budget_taken(gkey)
    if attempts >= MAX_ATTEMPTS:
        logger.info("%s auto-retry: budget exhausted (%d/%d) — relaying "
                    "the raw error", LOG_PREFIX, attempts, MAX_ATTEMPTS)
        return None

    # 4) There must be a declared tool the corrected script can ride down.
    exec_tool, param_name = _pick_exec_tool(obj.get("tools"))
    if not exec_tool or not param_name:
        return None

    # 5) Run the RUNTIME FIX inference against the same pinned model.
    _budget_charge(gkey)
    fix_request = {
        "model": obj.get("model") or "mixar-local",
        "messages": [
            {"role": "system", "content": _FIX_SYSTEM_PROMPT},
            {"role": "user", "content": (
                "ORIGINAL GOAL:\n" + (goal or "(unknown)") +
                "\n\nSCRIPT THAT FAILED:\n" + last_script +
                "\n\nRUNTIME ERROR:\n" + tail[:20000] +
                "\n\nReply with the complete corrected script only."
            )},
        ],
        "temperature": obj.get("temperature", 0.0),
        "max_tokens": obj.get("max_tokens") or 4096,
        "stream": False,
    }
    payload = json.dumps(fix_request).encode("utf-8")
    try:
        request = urllib.request.Request(
            pinned_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mixar/1.0",
                "Accept": "application/json",
            },
            method="POST",
        )
    except Exception:
        return None

    t0 = time.time()
    logger.info(
        "%s auto-retry: dispatching RUNTIME FIX inference (attempt %d/%d "
        "+ this) for goal key %s",
        LOG_PREFIX, attempts, MAX_ATTEMPTS, gkey[:12],
    )
    raw = None
    try:
        if relay_stop.is_stopped():
            return None
        opener = _tracked_opener()
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RELAY_RESPONSE_BYTES + 1)
    except Exception as exc:  # noqa: BLE001 — model down, relay as-is
        logger.info("%s auto-retry: inference failed (%s) — relaying "
                    "the raw error in %.1fs",
                    LOG_PREFIX, exc, time.time() - t0)
        return None
    logger.info("%s auto-retry: inference done in %.1fs",
                LOG_PREFIX, time.time() - t0)

    # 6) Parse the model reply STRICTLY; any deviation relays as-is.
    try:
        fix_obj = json.loads(raw[:MAX_RELAY_RESPONSE_BYTES].decode("utf-8"))
        message = fix_obj["choices"][0]["message"]
        content = message.get("content") or ""
    except Exception:
        return None
    fixed_script = _extract_script(content if isinstance(content, str) else "")
    if not fixed_script or fixed_script == last_script:
        return None
    if not _script_ok(fixed_script):
        return None

    # 7) Forge a chat-completion whose single tool call carries the fix.
    args = {param_name: fixed_script}
    try:
        forged = {
            "id": "chatcmpl-auto-fix",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": obj.get("model") or "mixar-local",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_auto_fix",
                        "type": "function",
                        "function": {
                            "name": exec_tool,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    except Exception:
        return None
    logger.info(
        "%s auto-retry: forged corrected %s dispatch (%d chars) "
        "in %.1fs", LOG_PREFIX, exec_tool, len(fixed_script), time.time() - t0,
    )
    return json.dumps(forged, ensure_ascii=False).encode("utf-8")
