# One-shot: dump the bpy.ops operator table to rag/mixar_ops.json.
# Run with:  "C:\Program Files\Mixar\mixar.exe" -b -noaudio --factory-startup --python rag/_one_shot_ops_dump.py
import bpy
import json
import os

ops = {}
for m in dir(bpy.ops):
    if not m[:1].islower():
        continue
    try:
        ops[m] = sorted(set(n for n in dir(getattr(bpy.ops, m)) if not n.startswith("_")))
    except Exception:
        ops[m] = []

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mixar_ops.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(ops, f)
print("OPS_DUMP_DONE categories=%d ops=%d" % (len(ops), sum(len(v) for v in ops.values())))
