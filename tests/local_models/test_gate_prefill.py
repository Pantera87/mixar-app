# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native message gate: Part A prefill reduction + goal-boundary detection.

Covers the request-shaping half of core/message_gate.py:
* system messages are always kept, only the last HISTORY_TAIL non-system
  messages are sent upstream;
* the window extends backward across tool results so an assistant
  tool_call is never orphaned from its result;
* tool-result content above TOOL_RESULT_MAX_CHARS is head/tail trimmed;
* bodies under the limits pass through byte-identical;
* goal-boundary (prompt-cache clear) detection off the last user message.
"""

import json

import pytest

from mixar.modules.local_models.core import message_gate
from mixar.modules.local_models.core.message_gate import (
    HISTORY_TAIL,
    TOOL_RESULT_MAX_CHARS,
    inspect_request,
    reset_state,
)


@pytest.fixture(autouse=True)
def _fresh_state():
    reset_state()
    yield
    reset_state()


def build_body(messages, **extra):
    obj = {"model": "m", "messages": messages}
    obj.update(extra)
    return json.dumps(obj).encode("utf-8")


def roles(body):
    return [m.get("role") for m in json.loads(body)["messages"]]


def contents(body, role=None):
    out = []
    for message in json.loads(body)["messages"]:
        if role is None or message.get("role") == role:
            out.append(message.get("content"))
    return out


# ---------------------------------------------------------------------------
# No-op paths
# ---------------------------------------------------------------------------

def test_body_under_limits_passes_through_unchanged():
    body = build_body([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "goal one"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "done"},
    ])
    out, goal_changed = inspect_request(body)
    assert out == body                      # byte-identical, not re-serialized
    assert goal_changed is False            # first goal is never a boundary


def test_unparseable_body_passes_through():
    out, goal_changed = inspect_request(b"this is not json{")
    assert out == b"this is not json{"
    assert goal_changed is False


def test_non_list_messages_pass_through():
    body = json.dumps({"model": "m", "messages": "nope"}).encode("utf-8")
    out, _ = inspect_request(body)
    assert out == body


# ---------------------------------------------------------------------------
# History tail window
# ---------------------------------------------------------------------------

def test_keeps_system_plus_last_tail_messages():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(1, 7):                   # 6 non-system messages
        messages.append({"role": "user" if i % 2 else "assistant",
                         "content": f"m{i}"})
    out, _ = inspect_request(build_body(messages))
    kept = json.loads(out)["messages"]
    assert [m.get("content") for m in kept] == ["sys", "m3", "m4", "m5", "m6"]
    assert len(kept) == 1 + HISTORY_TAIL


def test_tail_window_backtracks_across_tool_results():
    """A window that would start on a tool result is extended back to the
    assistant message whose tool_call answers it."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "execute_bpy_script",
                          "arguments": json.dumps({"code": "print(1)"})}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "executed"},
        {"role": "assistant", "content": "done, a cube was added"},
        {"role": "user", "content": "now make it bigger"},
        {"role": "user", "content": "and make it yellow"},
    ]
    # 6 non-system messages -> the tail of 4 would start ON the tool
    # result; the gate must pull the assistant tool_call in with it.
    out, _ = inspect_request(build_body(messages))
    kept = json.loads(out)["messages"]
    assert len(kept) == 6
    assert roles(out) == ["system", "assistant", "tool",
                          "assistant", "user", "user"]
    first_tool = next(m for m in kept if m.get("role") == "assistant"
                      and m.get("tool_calls"))
    assert first_tool["tool_calls"][0]["id"] == "call_1"
    # the tool result is present, so the pairing is intact
    assert contents(out, role="tool") == ["executed"]


def test_no_backtrack_when_window_starts_on_plain_message():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    out, _ = inspect_request(build_body(messages))
    assert roles(out) == ["system", "user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------------------
# Tool-result trimming
# ---------------------------------------------------------------------------

def test_oversized_tool_result_is_head_tail_trimmed():
    big = "L" * 5000
    messages = [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "execute_bpy_script",
                          "arguments": json.dumps({"code": "print(1)"})}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": big},
    ]
    out, _ = inspect_request(build_body(messages))
    trimmed = contents(out, role="tool")[0]
    assert len(trimmed) < len(big)
    # head + marker + tail; the marker carries the truncated length
    assert trimmed.startswith("L" * 50)
    assert trimmed.endswith("L" * 50)
    assert f"[{len(big) - TOOL_RESULT_MAX_CHARS} chars truncated]" in trimmed


def test_tool_result_at_limit_is_not_trimmed():
    exact = "x" * TOOL_RESULT_MAX_CHARS
    messages = [
        {"role": "user", "content": "goal"},
        {"role": "tool", "tool_call_id": "call_1", "content": exact},
    ]
    out, _ = inspect_request(build_body(messages))
    assert contents(out, role="tool")[0] == exact


def test_short_tool_result_is_untouched():
    body = build_body([
        {"role": "user", "content": "goal"},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ])
    out, _ = inspect_request(body)
    assert out == body
    assert contents(body, role="tool") == ["ok"]


# ---------------------------------------------------------------------------
# Goal-boundary detection
# ---------------------------------------------------------------------------

def _user_only(goal):
    return build_body([{"role": "user", "content": goal}])


def test_first_goal_is_not_a_boundary():
    body = _user_only("make a cube")
    out, changed = inspect_request(body)
    assert changed is False
    assert out == body


def test_same_goal_repeats_are_not_boundaries():
    inspect_request(_user_only("make a cube"))
    _, changed = inspect_request(_user_only("make a cube"))
    assert changed is False


def test_new_goal_is_a_boundary_and_updates_state():
    inspect_request(_user_only("make a cube"))
    _, changed = inspect_request(_user_only("make it bigger"))
    assert changed is True
    # the new goal becomes the baseline
    _, changed = inspect_request(_user_only("make it bigger"))
    assert changed is False
    # switching back is a boundary again
    _, changed = inspect_request(_user_only("make a cube"))
    assert changed is True


def test_multimodal_user_content_uses_text_parts():
    a = build_body([
        {"role": "user", "content": [
            {"type": "text", "text": "part one"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
        ]},
    ])
    b = build_body([
        {"role": "user", "content": [
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
        ]},
    ])
    _, first = inspect_request(a)
    assert first is False
    _, changed = inspect_request(b)
    assert changed is True


def test_reset_state_forces_no_boundary_on_next_goal():
    inspect_request(_user_only("make a cube"))
    reset_state()
    _, changed = inspect_request(_user_only("make a cube"))
    assert changed is False