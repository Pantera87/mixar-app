# One-shot selftest for the native message gate (run with mixar.exe -b).
import json
import os
import sys

# Mixar's startup has already imported its OWN bundled copy of the `mixar`
# package; drop it and re-import from this repo's source tree instead, so
# the selftest exercises the code under development.
for _name in [n for n in sys.modules if n == "mixar" or n.startswith("mixar.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src", "scripts"))

failures = []


def check(label, cond):
    print(("PASS" if cond else "FAIL"), label)
    if not cond:
        failures.append(label)


# ---------------------------------------------------------------- validator
from mixar.modules.space_mixie_chat.core import script_validator as v

# NOTE: this Mixar build has NO bpy.ops.mesh.extrude (it has
# extrude_repeat / extrude_region / extrude_context) — the test data
# reflects that.
bogus = v.validate_bpy_properties("bpy.ops.mesh.extrude_forever()")
print("DEBUG bogus issues:", bogus)
real = v.validate_bpy_properties("bpy.ops.mesh.primitive_cube_add()")
print("DEBUG real issues:", real)
check("validator: bogus operator flagged (with suggestion)",
      any("extrude_forever" in i and "Did you mean" in i for i in bogus))
check("validator: real operator clean", real == [])
bogus_cat = v.validate_bpy_properties("bpy.ops.blehhhh.foo()")
check("validator: bogus category flagged",
      any("no such operator category" in i for i in bogus_cat))

# ------------------------------------------------------------------ gate
from mixar.modules.local_models.core import message_gate

message_gate.reset_state()

TOOLS = [{"function": {"name": "run_bpy", "parameters": {
    "type": "object",
    "properties": {"script": {"type": "string"}},
    "required": ["script"]}}}]

# 1) hallucinated call quarantined + its tool result dropped; valid kept
messages = [
    {"role": "user", "content": "make a cube"},
    {"role": "assistant", "content": "ok", "tool_calls": [
        {"id": "c1", "type": "function", "function": {
            "name": "teleport_user_to_moon",
            "arguments": "{\"a\":1}"}},
        {"id": "c2", "type": "function", "function": {
            "name": "run_bpy",
            "arguments": "{\"script\": \"import bpy\\nbpy.ops.mesh.primitive_cube_add()\"}"}},
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": "teleported"},
    {"role": "tool", "tool_call_id": "c2", "content": "done"},
]
body = json.dumps({"messages": messages, "tools": TOOLS}).encode()
out, changed = message_gate.inspect_request(body)
parsed = json.loads(out)
kept = parsed["messages"][1]["tool_calls"]
check("gate: quarantined call removed, valid kept",
      [c["function"]["name"] for c in kept] == ["run_bpy"])
check("gate: orphaned tool result dropped",
      all(m.get("tool_call_id") != "c1" for m in parsed["messages"]))
check("gate: valid tool result kept",
      any(m.get("role") == "tool" and m.get("tool_call_id") == "c2"
          for m in parsed["messages"]))
check("gate: no goal change on first turn", changed is False)

# 2) bad JSON arguments quarantined
body2 = json.dumps({"messages": [
    {"role": "user", "content": "x"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c9", "type": "function", "function": {
            "name": "run_bpy", "arguments": "{not json"}}]},
    {"role": "tool", "tool_call_id": "c9", "content": "r"}],
    "tools": TOOLS}).encode()
out2, _ = message_gate.inspect_request(body2)
parsed2 = json.loads(out2)
check("gate: non-JSON args quarantined + empty assistant dropped",
      len(parsed2["messages"]) == 1
      and parsed2["messages"][0].get("role") == "user")

# 3) goal boundary
body3 = json.dumps({"messages": [{"role": "user", "content": "build a bridge"}],
                    "tools": TOOLS}).encode()
_, changed3 = message_gate.inspect_request(body3)
check("gate: goal boundary detected on new user message", changed3 is True)
out3b, changed3b = message_gate.inspect_request(body3)
check("gate: same goal is not a boundary", changed3b is False)

# 4) clean history passes through byte-identical
body4 = json.dumps({"messages": [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi, what next?"}], "tools": TOOLS}).encode()
out4, changed4 = message_gate.inspect_request(body4)
check("gate: clean body untouched (same bytes)", out4 is body4)
check("gate: goal boundary on another new message", changed4 is True)

# 5) response audit (log-only, must not raise)
script_arg = json.dumps({"script": "bpy.ops.mesh.extrude_forever()"})
resp_body = json.dumps({
    "choices": [{"message": {
        "content": "sure:\n```python\nimport bpy\n"
                   "bpy.ops.mesh.extrude_forever()\n```",
        "tool_calls": [
            {"function": {"name": "run_bpy", "arguments": script_arg}},
            {"function": {"name": "run_bpy", "arguments": "{bad json"}},
        ]}}]})
message_gate.inspect_response(resp_body.encode())
check("gate: response audit never raises", True)

# 6) relay wiring
from mixar.modules.local_models.core import relay
check("relay: gate wired", hasattr(relay, "message_gate")
      and hasattr(relay, "_clear_remote_cache"))
from mixar.modules.local_models import constants as lm
check("relay: gate constants", lm.GATE_CACHE_CLEAR_ON_NEW_GOAL is True
      and lm.GATE_CACHE_CLEAR_TIMEOUT_S == 5.0)

print("\n%d failures" % len(failures))
sys.exit(1 if failures else 0)
