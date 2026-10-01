# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Closed-loop runtime-error auto-retry (core/relay_autofix.py).

A real local HTTP server plays the fix model, so the full contract is
exercised: Blender-error detection, goal-keyed attempt budget, spin guard,
script validation of the forge payload, and the exact response envelope
Mixar accepts (tool_calls, no content, zero usage).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mixar.modules.local_models.core import relay_autofix
from mixar.modules.local_models.core import relay_stop

ORIGINAL_SCRIPT = (
    "import bpy\n"
    "\n"
    "bpy.ops.mesh.primitive_cube_add(size=1)\n"
    "obj = bpy.context.object\n"
    "obj.data.shade_smooth()"
)
FIXED_SCRIPT = (
    "import bpy\n"
    "\n"
    "bpy.ops.mesh.primitive_cube_add(size=1)\n"
    "bpy.ops.object.shade_smooth()"
)
TRACEBACK = (
    "Traceback (most recent call last):\n"
    "  File \"mixar_executor.py\", line 42, in <module>\n"
    "    obj.data.shade_smooth()\n"
    "AttributeError: 'Mesh' object has no attribute 'shade_smooth'\n"
)
GOAL = "Create a cube and smooth it"

EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_bpy_script",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
        },
    },
}
NO_EXEC_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_documentation",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
]


class _BaseFixHandler(BaseHTTPRequestHandler):
    """Replies to one chat-completion POST with ``type(self).fix_reply``."""

    fix_reply = "```python\n" + FIXED_SCRIPT + "```"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({
            "id": "chatcmpl-fix",
            "choices": [{
                "message": {"role": "assistant",
                            "content": type(self).fix_reply},
            }],
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def model_server():
    """Yields a factory: start a fix-model server replying ``reply`` and
    return its chat-completions URL."""
    servers = []

    def start(reply="```python\n" + FIXED_SCRIPT + "```"):
        class Handler(_BaseFixHandler):
            fix_reply = reply

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return (f"http://127.0.0.1:{server.server_address[1]}"
                "/v1/chat/completions")

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _clean_autofix_state():
    relay_autofix.reset_budget()
    relay_stop.clear_stop()
    yield
    relay_autofix.reset_budget()
    relay_stop.clear_stop()


def make_body(goal=GOAL, script=ORIGINAL_SCRIPT, error=TRACEBACK,
              tools=None, model="mixar-test"):
    """A relay body: user goal, one dispatched script, its tool result."""
    messages = [
        {"role": "user", "content": goal},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {
                "name": "execute_bpy_script",
                "arguments": json.dumps({"code": script}),
            },
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": error},
    ]
    obj = {"model": model, "messages": messages}
    if tools is not None:
        obj["tools"] = tools
    return json.dumps(obj).encode("utf-8")


def forge(url, body):
    return relay_autofix.maybe_forge(body, url, 5.0)


# ---------------------------------------------------------------------------
# Gate: when the forge is (not) attempted at all
# ---------------------------------------------------------------------------

def test_non_traceback_result_is_relayed_verbatim(model_server):
    url = model_server()
    assert forge(url, make_body(error="done, no errors")) is None
    assert forge(url, make_body(error="")) is None


def test_timeout_error_is_not_a_blender_crash(model_server):
    url = model_server()
    timed_out = TRACEBACK + "Operation timed out while executing\n"
    assert forge(url, make_body(error=timed_out)) is None


def test_no_exec_tool_in_catalogue_relays_unchanged(model_server):
    url = model_server()
    assert forge(url, make_body(tools=NO_EXEC_TOOL)) is None
    assert forge(url, make_body(tools=None)) is None


def test_malformed_body_fails_open(model_server):
    url = model_server()
    assert relay_autofix.maybe_forge(b"{not json", url, 5.0) is None
    assert relay_autofix.maybe_forge(b'{"messages": "nope"}', url, 5.0) is None


def test_stopped_relay_refuses_the_fix_roundtrip(model_server):
    url = model_server()
    relay_stop.request_stop()
    assert forge(url, make_body(tools=[EXEC_TOOL])) is None


# ---------------------------------------------------------------------------
# The fix roundtrip
# ---------------------------------------------------------------------------

def test_forges_valid_fix(model_server):
    url = model_server()
    out = forge(url, make_body(tools=[EXEC_TOOL]))
    assert out is not None
    parsed = json.loads(out.decode("utf-8"))
    assert parsed["id"] == "chatcmpl-auto-fix"
    assert parsed["model"] == "mixar-test"
    assert parsed["usage"] == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }
    message = parsed["choices"][0]["message"]
    assert parsed["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    tool_call = message["tool_calls"][0]
    assert tool_call["id"] == "call_auto_fix"
    assert tool_call["function"]["name"] == "execute_bpy_script"
    assert json.loads(tool_call["function"]["arguments"])["code"] == FIXED_SCRIPT


def test_reply_with_no_code_block_is_relayed_verbatim(model_server):
    url = model_server(reply="Sorry, I think that attribute does not exist.")
    assert forge(url, make_body(tools=[EXEC_TOOL])) is None


def test_reply_with_invalid_script_is_relayed_verbatim(model_server):
    url = model_server(reply="```python\nimport bpy\ndef broken(:\n```")
    assert forge(url, make_body(tools=[EXEC_TOOL])) is None


def test_spin_guard_rejects_the_same_script(model_server):
    """If the model 'fixes' the script back into itself, relaying that would
    loop forever — so the forge gives up."""
    url = model_server(reply="```python\n" + ORIGINAL_SCRIPT + "```")
    assert forge(url, make_body(tools=[EXEC_TOOL])) is None


# ---------------------------------------------------------------------------
# Attempt budget (per goal)
# ---------------------------------------------------------------------------

def test_budget_allows_two_dispatches_then_stops(model_server):
    url = model_server()
    body = make_body(tools=[EXEC_TOOL])
    assert forge(url, body) is not None      # attempt 1
    assert forge(url, body) is not None      # attempt 2 (last allowed)
    assert forge(url, body) is None          # budget exhausted
    assert forge(url, body) is None          # stays exhausted


def test_new_goal_gets_a_fresh_budget(model_server):
    url = model_server()
    body_a = make_body(goal="Goal A", tools=[EXEC_TOOL])
    body_b = make_body(goal="Goal B", tools=[EXEC_TOOL])
    assert forge(url, body_a) is not None
    assert forge(url, body_a) is not None
    assert forge(url, body_a) is None        # Goal A exhausted
    assert forge(url, body_b) is not None    # Goal B starts fresh