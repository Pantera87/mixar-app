# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native LLM message gate for local-model relay traffic.

Runs inside mixar.exe on the relay path (local_models/core/relay.py) — the
single choke point every BYOK / local-LLM chat-completions request passes
through — and replaces the optional ``rag/mixar-rag-server.py`` bridge:

* :func:`inspect_request` — quarantines hallucinated ``tool_calls`` from
  the conversation history BEFORE the request reaches the local server. A
  call is quarantined when its name is not in the request's declared
  ``tools`` list or its ``arguments`` are not valid JSON (the tool-result
  message that answered a quarantined call is removed too, so the history
  stays a valid OpenAI pairing). Schema / required-key mismatches and
  script bodies are audited (logged) but never touched. It also shrinks the prefill (part A): only the system
  messages plus the last ``HISTORY_TAIL`` non-system messages are sent
  upstream, and tool-result content over ``TOOL_RESULT_MAX_CHARS`` is
  head/tail-trimmed.
* :func:`inspect_response` — log-only audit of the model output: every
  fenced Python block and every script argument of a returned
  ``tool_call`` is run through the space_mixie_chat script validator (the
  same hard gate the executor applies before dispatch).
* Goal-boundary detection inside :func:`inspect_request` — the last user
  message is hashed; when it changes, the relay may clear the local
  server's prompt cache so a fresh goal doesn't start with the previous
  goal's KV cache.

Design law (same as the whole relay): this module NEVER blocks a turn.
Every failure mode degrades to "log and pass through". It must stay
bpy-free: it runs on the relay worker thread.
"""

import hashlib
import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from mixar.config.logging_config import get_logger

logger = get_logger(__name__)

LOG_PREFIX = "[MessageGate]"

# Tool-call parameter keys whose value is an executable Mixar script.
_SCRIPT_PARAM_KEYS = ("script", "code", "bpy_script", "python_code", "source")

# ```python / ```py fenced blocks (the form the generation model emits).
_FENCED_RE = re.compile(r"```(?:python|py)\b[^\n]*\n(.*?)```",
                        re.IGNORECASE | re.DOTALL)

# How many validator issues we log per script block (this is telemetry,
# not the gate — the executor's prepare_script() is what enforces).
_MAX_LOGGED_ISSUES = 5

# --- Prefill reduction (part A: shrink the request, never the semantics) ---
# How many NON-SYSTEM messages of the conversation to keep in the outbound
# request. The model only needs the recent context, not the full history —
# the tail carries the live tool loop. The window is extended backward
# across tool results so OpenAI call/result pairing stays valid.
HISTORY_TAIL = 4
# Head/tail cap for a single tool-result message: huge executor output
# (tracebacks, file dumps) is the main prefill bloat, and only the head
# (where a traceback starts) and the tail (where the final error is)
# matter to the model.
TOOL_RESULT_MAX_CHARS = 2000

_state_lock = threading.Lock()
_last_goal_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _script_issues(source: str) -> List[str]:
    """Run the Mixar script validator over one source block.

    The import is lazy (it pulls in the truth-table machinery) and any
    failure degrades to an empty issue list — the gate must never raise
    into the relay. The script validator itself is blind-safe: with no
    truth table it reports nothing rather than guessing."""
    try:
        from mixar.modules.space_mixie_chat.core import script_validator
        issues = script_validator.validate_bpy_properties(source)
        return [str(issue) for issue in (issues or [])]
    except Exception as exc:  # noqa: BLE001 — telemetry must not raise
        logger.warning("%s script validation unavailable (%s)",
                       LOG_PREFIX, exc)
        return []


def _log_script_issues(source: str, label: str) -> None:
    issues = _script_issues(source)
    for issue in issues[:_MAX_LOGGED_ISSUES]:
        logger.warning("%s %s script issue: %s", LOG_PREFIX, label, issue)


def _declared_tool_names(request_obj: dict) -> set:
    """Tool names the request declared (``tools[].function.name``)."""
    names = set()
    tools = request_obj.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict) \
                    and isinstance(function.get("name"), str):
                names.add(function["name"])
            elif isinstance(tool.get("name"), str):
                names.add(tool["name"])
    return names


def _tool_schemas(request_obj: dict) -> Dict[str, dict]:
    """name -> parameters schema, for the log-only schema audit."""
    schemas: Dict[str, dict] = {}
    tools = request_obj.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict) \
                    and isinstance(function.get("name"), str) \
                    and isinstance(function.get("parameters"), dict):
                schemas[function["name"]] = function["parameters"]
    return schemas


def _audit_tool_call(call: dict, declared: set, schemas: dict) -> bool:
    """Audit one tool_call. Returns True when the call is QUARANTINED
    (undeclared name or unparseable / non-object arguments). Schema /
    required-key mismatches and script bodies are logged only — dropping
    such a call would orphan its tool-result message and break the
    pairing."""
    function = call.get("function")
    if not isinstance(function, dict):
        return True          # malformed beyond repair — quarantine
    name = function.get("name")
    if not isinstance(name, str) or (declared and name not in declared):
        logger.warning("%s quarantining undeclared tool_call: %r",
                       LOG_PREFIX, name)
        return True
    args = function.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            logger.warning("%s quarantining tool_call %s: arguments are not "
                           "valid JSON", LOG_PREFIX, name)
            return True
    if args is None:
        args = {}
    elif not isinstance(args, dict):
        logger.warning("%s quarantining tool_call %s: arguments are not a "
                       "JSON object", LOG_PREFIX, name)
        return True
    # ---- log-only: schema mismatch ----
    schema = schemas.get(name)
    if isinstance(schema, dict):
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        required = schema.get("required")
        required = required if isinstance(required, list) else []
        missing = [k for k in required if k not in args]
        unknown = [k for k in args if props and k not in props]
        if missing or unknown:
            logger.warning("%s tool_call %s schema mismatch: missing=%s "
                           "unknown=%s", LOG_PREFIX, name, missing, unknown)
    # ---- log-only: the script body itself ----
    for key in _SCRIPT_PARAM_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            _log_script_issues(value, f"tool_call {name}.{key}")
    return False


def _quarantine(messages: list, declared: set, schemas: dict) -> Tuple[list, int]:
    """Remove hallucinated tool_calls (and the tool results that answer
    them) so the history stays a valid OpenAI pairing."""
    out: List[Any] = []
    removed_ids = set()
    quarantined = 0
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        if message.get("role") == "tool":
            # orphaned tool result for a quarantined call -> drop
            cid = message.get("tool_call_id")
            if isinstance(cid, str) and cid in removed_ids:
                continue
            out.append(message)
            continue
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list):
            out.append(message)
            continue
        kept: List[Any] = []
        for call in calls:
            if not isinstance(call, dict):
                quarantined += 1
                continue
            if _audit_tool_call(call, declared, schemas):
                cid = call.get("id")
                if isinstance(cid, str):
                    removed_ids.add(cid)
                quarantined += 1
                continue
            kept.append(call)
        if len(kept) == len(calls):
            out.append(message)
            continue
        reduced = dict(message)
        if kept:
            reduced["tool_calls"] = kept
            out.append(reduced)
        elif message.get("content") not in (None, ""):
            # every call quarantined: keep the content, drop the key
            reduced.pop("tool_calls", None)
            out.append(reduced)
    return out, quarantined


def _prefill_reduce(messages: list) -> Tuple[list, int]:
    """Shrink the conversation for prefill (part A):

    * keep every ``system`` message plus only the last ``HISTORY_TAIL``
      non-system messages, extending the window backward across tool
      results so a tool result never orphans its assistant tool_call;
    * head/tail-trim any tool-result content above
      ``TOOL_RESULT_MAX_CHARS``.

    Returns ``(reduced, n_changed)`` where ``n_changed`` is the number of
    messages dropped or trimmed. Never raises; any surprise degrades to
    "pass through".
    """
    if not isinstance(messages, list) or not messages:
        return messages, 0
    changed = 0
    try:
        non_system = [
            i for i, message in enumerate(messages)
            if not (isinstance(message, dict) and message.get("role") == "system")
        ]
        keep_from = max(0, len(non_system) - HISTORY_TAIL)
        window_start = (
            non_system[keep_from] if keep_from < len(non_system) else None
        )
        if window_start is not None:
            # extend backward: the window must not start on a tool result
            # whose answering assistant call is outside the window
            while window_start > 0:
                head = messages[window_start]
                if (isinstance(head, dict)
                        and head.get("role") in ("tool", "function")):
                    window_start -= 1
                    continue
                break
            if window_start < non_system[keep_from]:
                changed += non_system[keep_from] - window_start
        kept = []
        for index, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "system":
                kept.append(message)
                continue
            if window_start is not None and index >= window_start:
                kept.append(message)
                continue
            if window_start is not None:
                changed += 1          # dropped from the head
        # oversized tool results -> head + tail
        trimmed_kept: List[Any] = []
        for message in kept:
            if (isinstance(message, dict)
                    and message.get("role") in ("tool", "function")):
                content = message.get("content")
                if (isinstance(content, str)
                        and len(content) > TOOL_RESULT_MAX_CHARS):
                    half = TOOL_RESULT_MAX_CHARS // 2
                    marker = ("\n... [%d chars truncated] ...\n"
                              % (len(content) - TOOL_RESULT_MAX_CHARS))
                    reduced = dict(message)
                    reduced["content"] = (
                        content[:half] + marker + content[-half:])
                    trimmed_kept.append(reduced)
                    changed += 1
                    continue
            trimmed_kept.append(message)
        return trimmed_kept, changed
    except Exception:
        return messages, 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inspect_request(body: bytes) -> Tuple[bytes, bool]:
    """Gate one inbound chat-completions request body.

    Returns ``(body_out, goal_changed)``: ``body_out`` is the body to send
    upstream — re-serialized ONLY when the quarantine or the prefill
    reduction actually changed the history (system messages + the last
    ``HISTORY_TAIL`` non-system messages, oversized tool results
    head/tail-trimmed) — and ``goal_changed`` is True when the last user
    message differs from the one seen on the previous request, a goal
    boundary where the relay may clear the local server's prompt cache.
    Never raises; a body that cannot be parsed is returned untouched with
    ``goal_changed`` False.
    """
    global _last_goal_hash
    try:
        obj = json.loads(body.decode("utf-8"))
    except Exception:
        return body, False
    if not isinstance(obj, dict):
        return body, False

    # ---- goal-boundary detection (last user message) ----
    goal_changed = False
    last_user = None
    messages = obj.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                last_user = message.get("content")
    text = ""
    if isinstance(last_user, str):
        text = last_user
    elif isinstance(last_user, list):        # content-parts form
        text = " ".join(
            part.get("text", "")
            for part in last_user
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if text.strip():
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        with _state_lock:
            # The first goal ever seen is NOT a boundary (there is no
            # previous goal whose cache could leak into it).
            goal_changed = (_last_goal_hash is not None
                            and digest != _last_goal_hash)
            _last_goal_hash = digest

    # ---- history shaping (re-serializes only when something changed) ----
    if not isinstance(messages, list):
        return body, goal_changed
    declared = _declared_tool_names(obj)
    schemas = _tool_schemas(obj)
    out_messages, quarantined = _quarantine(messages, declared, schemas)
    if quarantined:
        logger.info("%s quarantined %d hallucinated tool_call(s) from history",
                    LOG_PREFIX, quarantined)
    out_messages, reduced = _prefill_reduce(out_messages)
    if not quarantined and not reduced:
        return body, goal_changed
    if reduced:
        logger.info("%s prefill reduction: %d message(s) dropped/trimmed",
                    LOG_PREFIX, reduced)
    new_obj = dict(obj)
    new_obj["messages"] = out_messages
    new_body = json.dumps(new_obj, ensure_ascii=False).encode("utf-8")
    return new_body, goal_changed


def inspect_response(body: bytes) -> None:
    """Log-only audit of a chat-completions RESPONSE body (never mutates,
    never raises): fenced Python blocks in the message content and script
    arguments of returned tool_calls are run through the Mixar script
    validator."""
    try:
        obj = json.loads(body.decode("utf-8"))
    except Exception:
        return
    if not isinstance(obj, dict):
        return
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        _audit_message_payload(message)


def _audit_message_payload(message: dict) -> None:
    content = message.get("content")
    if isinstance(content, str):
        for block in _FENCED_RE.findall(content):
            _log_script_issues(block, "response content")
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        args = function.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                logger.warning("%s response tool_call %s: arguments are not "
                               "valid JSON", LOG_PREFIX, name)
                continue
        if not isinstance(args, dict):
            continue
        for key in _SCRIPT_PARAM_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                _log_script_issues(value,
                                   f"response tool_call {name}.{key}")


def reset_state() -> None:
    """Drop the goal-boundary state (tests / session teardown)."""
    global _last_goal_hash
    with _state_lock:
        _last_goal_hash = None
