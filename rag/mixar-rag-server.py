import importlib
import os
import re
import sys
import subprocess
import hashlib

# ============================================================
# 0. AUTO-INSTALL MISSING DEPENDENCIES (first run only)
#
# NOTE: 'bpy' is deliberately NOT in this list — there is no
# pip wheel for Python 3.12, and an official wheel would describe
# stock Blender, not your Mixar 5.2 fork. Ground truth comes from
# the Mixar engine binary itself (Section 3), cached to
# mixar_props.json / mixar_types.json / mixar_sockets.json, and
# mirrored into the RAG collection via:
#     python mixar-rag-server.py --ingest-truth
# ============================================================
REQUIRED_PACKAGES = [  # (pip name, import name)
    ("fastapi", "fastapi"),
    ("pydantic", "pydantic"),
    ("uvicorn", "uvicorn"),
    ("chromadb", "chromadb"),
    ("openai", "openai"),
    ("torch", "torch"),
    ("sentence-transformers", "sentence_transformers"),
]


# ------------------------------------------------------------
# BANNER
# ------------------------------------------------------------
BANNER = r"""
  ███████████                        █████                                  ████████   ██████████
▒▒███▒▒▒▒▒███                      ▒▒███                                  ███▒▒▒▒███ ▒███▒▒▒▒███
 ▒███    ▒███  ██████   ████████   ███████    ██████  ████████   ██████  ▒███   ▒███ ▒▒▒    ███ 
 ▒██████████  ▒▒▒▒▒███ ▒▒███▒▒███ ▒▒▒███▒    ███▒▒███▒▒███▒▒███ ▒▒▒▒▒███ ▒▒████████        ███  
 ▒███▒▒▒▒▒▒    ███████  ▒███ ▒███   ▒███    ▒███████  ▒███ ▒▒▒   ███████  ███▒▒▒▒███      ███   
 ▒███         ███▒▒███  ▒███ ▒███   ▒███ ███▒███▒▒▒   ▒███      ███▒▒███ ▒███   ▒███     ███    
 █████       ▒▒████████ ████ █████  ▒▒█████ ▒▒██████  █████    ▒▒████████▒▒████████     ███     
▒▒▒▒▒         ▒▒▒▒▒▒▒▒ ▒▒▒▒ ▒▒▒▒▒    ▒▒▒▒▒   ▒▒▒▒▒▒  ▒▒▒▒▒      ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒     ▒▒▒      
                                                                                                
                                                                                                
                                                                                                
    ─────────────────────────────────────────────────────────────────
    RAG augmenting validator script  by Pantera87
    (v2: chain-walking attribute validation, truth-table only)
"""
print(BANNER)


def _ensure_dependencies():
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            print(f"✅ {pip_name}: present")
        except Exception:
            print(f"⬇️  {pip_name} is missing — auto-installing (this can take a while on first run)...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                print(f"✅ {pip_name}: installed")
            except Exception as e:
                print(f"❌ {pip_name}: auto-install FAILED ({e}). Continuing.")


_ensure_dependencies()

import ast
import html
import json
import urllib.request
from difflib import get_close_matches
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import openai
import uvicorn
import torch

app = FastAPI(title="Mixar RAG Dynamic Loop-Breaking Bridge")

# ------------------------------------------------------------
# THINKING TOGGLE (Qwen3 /no_think · /think control tokens)
#   "off"  = append /no_think to every user turn  ← default
#   "on"   = append /think to every user turn
#   "auto" = send nothing, model decides per request
# ------------------------------------------------------------
THINKING_MODE = "off"

# ------------------------------------------------------------
# PHASE FLAGS
#   SKIP_DOC_PHASE = True -> turns where Mixar declared only
#   search tools (search_documentation & co.) are answered
#   IMMEDIATELY with a canned continue-message — no LLM call,
#   no state pollution. Set False to restore the legacy
#   plain-text pass-through LLM call.
# ------------------------------------------------------------
SKIP_DOC_PHASE = True

# ------------------------------------------------------------
# 1. PATHS — resolved relative to THIS file's folder
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "blender_rag_db")
REF_JSON = os.path.join(SCRIPT_DIR, "mixar_props.json")        # v2: full dir()-based attribute sets
REF_TYPES_JSON = os.path.join(SCRIPT_DIR, "mixar_types.json")   # per-property type/min/max/enum
REF_SOCK_JSON = os.path.join(SCRIPT_DIR, "mixar_sockets.json")  # per-node-class socket names
REF_META_JSON = os.path.join(SCRIPT_DIR, "mixar_meta.json")     # cache format version
TRUTH_VERSION = 7   # v6: concrete socket classes via type(socket).__name__ + valid Mixar tree types

print(f"📁 RAG database location: {DB_PATH}")
if not os.path.isdir(DB_PATH):
    print("⚠️  WARNING: 'blender_rag_db' folder not found next to this script. "
          "Chroma will create an empty one — your RAG context will be blank "
          "until you move the database folder here (expected at: " + DB_PATH + ")")

# ------------------------------------------------------------
# 2. SETUP EMBEDDING DEVICE
# ------------------------------------------------------------
gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Chromadb Device Context: {gpu_device.upper()}")

print("⏳ Loading embedding model BAAI/bge-large-en-v1.5 (first run may download)...")
GLOBAL_EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5",
    device=gpu_device
)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="blender_api", embedding_function=GLOBAL_EMBEDDING_FN
)
print(f"📚 Collection 'blender_api' contains {collection.count()} documents.")

# ------------------------------------------------------------
# 3. GROUND TRUTH FOR ATTRIBUTE VALIDATION  (v2)
#    Layer 0: cached JSON (mixar_props/types/sockets + meta v2)
#    Layer 1: pip 'bpy' (rarely available on Python 3.12)
#    Layer 2: the Mixar binary (AUTHORITATIVE; dumped + cached)
#
# v2 dumps, per class:
#   * dir(cls) -> the FULL valid attribute set (properties AND
#     methods/functions — so method calls can't false-positive)
#   * per-property {type, min, max, enum items}
#   * per NODE class: input/output socket names (instantiated in
#     probe node trees)
# ------------------------------------------------------------
import shutil

BLENDER_EXE = r"C:\Program Files\Mixar\mixar.exe"  # ← set to YOURS


def _find_blender():
    if BLENDER_EXE and os.path.isfile(BLENDER_EXE):
        return BLENDER_EXE
    found = shutil.which("mixar")
    if found:
        return found
    for base in (r"C:\Program Files\Mixar",
                 r"C:\Program Files (x86)\Mixar"):
        if os.path.isdir(base):
            for entry in sorted(os.listdir(base), reverse=True):
                cand = os.path.join(base, entry, "mixar.exe")
                if os.path.isfile(cand):
                    return cand
    return None


BLENDER_PATH = _find_blender()
if BLENDER_PATH:
    print(f"✅ Mixar found for property reference: {BLENDER_PATH}")
else:
    print("⚠️ Mixar not found — set BLENDER_EXE at the top of this file. "
          "Attribute validation will have no truth table.")

BPy_ATTRS = {}     # lower-class-name -> sorted full attribute list (props + methods)
BPy_ATTR_SETS = {} # lower-class-name -> set(...) for O(1) checks
BPy_NAMES = {}     # lower-class-name -> canonical class name
BPy_TYPES = {}     # lower-class-name -> {prop: {"type","min","max","enum":[...]}}
BPy_SOCKETS = {}   # lower-node-class -> {"inputs": [names], "outputs": [names],
                   #                      "input_types": {name: cls}, "output_types": {name: cls}}
_BPy_REF_LOADED = False

_BPY_DUMP_EXPR = (
    "import bpy, json\n"
    "attrs, ptypes, sockets = {}, {}, {}\n"
    "node_clss = []\n"
    "for name in dir(bpy.types):\n"
    "    if name[:1].isupper():\n"
    "        cls = getattr(bpy.types, name, None)\n"
    "        rna = getattr(cls, 'bl_rna', None) if cls is not None else None\n"
    "        if rna is None:\n"
    "            continue\n"
    "        try:\n"
    "            rna_props = [p.identifier for p in rna.properties]\n"
    "            attrs[name] = sorted(set(d for d in dir(cls) if not d.startswith('__')) | set(rna_props))\n"
    "        except Exception:\n"
    "            attrs[name] = []\n"
    "        tm = {}\n"
    "        try:\n"
    "            for p in rna.properties:\n"
    "                info = {'type': p.type or '', 'min': None, 'max': None, 'enum': []}\n"
    "                try:\n"
    "                    info['min'] = p.min\n"
    "                    info['max'] = p.max\n"
    "                except Exception:\n"
    "                    pass\n"
    "                try:\n"
    "                    info['enum'] = [e.identifier for e in p.enum_items]\n"
    "                except Exception:\n"
    "                    pass\n"
    "                info['readonly'] = bool(getattr(p, 'is_readonly', False))\n"
    "                if p.type in ('FLOAT', 'INT') and getattr(p, 'is_array', False):\n"
    "                    info['type'] = p.type + '_ARRAY'\n"
    "                tm[p.identifier] = info\n"
    "        except Exception:\n"
    "            pass\n"
    "        ptypes[name] = tm\n"
    "        try:\n"
    "            if issubclass(cls, bpy.types.Node):\n"
    "                node_clss.append(name)\n"
    "        except Exception:\n"
    "            pass\n"
    "try:\n"
    "    trees = {}\n"
    "    for tname in ('ShaderNodeTree', 'GeometryNodeTree', 'CompositorNodeTree', 'TextureNodeTree'):\n"
    "        try:\n"
    "            trees[tname] = bpy.data.node_groups.new('__SOCK_PROBE__' + tname, tname)\n"
    "        except Exception:\n"
    "            pass\n"
    "    for nm in node_clss:\n"
    "        if nm in sockets:\n"
    "            continue\n"
    "        for tname, g in trees.items():\n"
    "            try:\n"
    "                node = g.nodes.new(nm)\n"
    "            except Exception:\n"
    "                continue\n"
    "            try:\n"
    "                ins = {}; [ins.setdefault(i.name or 'Slot%d' % len(ins), type(i).__name__) for i in node.inputs]\n"
    "            except Exception:\n"
    "                ins = {}\n"
    "            try:\n"
    "                outs = {}; [outs.setdefault(o.name or 'Slot%d' % len(outs), type(o).__name__) for o in node.outputs]\n"
    "            except Exception:\n"
    "                outs = {}\n"
    "            try:\n"
    "                g.nodes.remove(node)\n"
    "            except Exception:\n"
    "                pass\n"
    "            sockets[nm] = {'inputs': list(ins), 'outputs': list(outs),\n"
    "                           'input_types': ins, 'output_types': outs}\n"
    "            break\n"
    "    for tname, g in trees.items():\n"
    "        try:\n"
    "            bpy.data.node_groups.remove(g)\n"
    "        except Exception:\n"
    "            pass\n"
    "except Exception:\n"
    "    pass\n"
    "print('__BPy_ATTRS__' + json.dumps(attrs))\n"
    "print('__BPy_TYPES__' + json.dumps(ptypes))\n"
    "print('__BPy_SOCKETS__' + json.dumps(sockets))\n"
)


def _load_from_cache():
    with open(REF_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    for name, alist in data.items():
        key = name.lower()
        BPy_ATTRS[key] = alist
        BPy_ATTR_SETS[key] = set(alist)
        BPy_NAMES[key] = name
    if os.path.isfile(REF_TYPES_JSON):
        try:
            with open(REF_TYPES_JSON, "r", encoding="utf-8") as f:
                tdata = json.load(f)
            for name, tmap in tdata.items():
                BPy_TYPES[name.lower()] = tmap
            # The cached dir() lists were taken from the RNA TYPES, which only
            # expose boilerplate methods — the real property identifiers live in
            # the type maps. Merge them in so existence checks see real props.
            for key, tmap in BPy_TYPES.items():
                if not tmap:
                    continue
                merged = sorted(set(BPy_ATTRS.get(key, [])) | set(tmap.keys()))
                BPy_ATTRS[key] = merged
                BPy_ATTR_SETS[key] = set(merged)
            print(f"✅ Value-type reference loaded: {len(BPy_TYPES)} classes (type/enum checks ON).")
        except Exception as e:
            print(f"⚠️ Could not read {REF_TYPES_JSON} ({e}) — value-type checks OFF.")
    else:
        print("ℹ️  No mixar_types.json — value-type checks OFF. "
              "Re-dump with: python mixar-rag-server.py --refresh-truth")
    if os.path.isfile(REF_SOCK_JSON):
        try:
            with open(REF_SOCK_JSON, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            for name, smap in sdata.items():
                BPy_SOCKETS[name.lower()] = smap
            print(f"✅ Socket reference loaded: {len(BPy_SOCKETS)} node classes (socket checks ON).")
        except Exception as e:
            print(f"⚠️ Could not read {REF_SOCK_JSON} ({e}) — socket checks OFF.")
    print(f"✅ Attribute reference from cache: {len(BPy_ATTRS)} classes "
          f"({sum(len(v) for v in BPy_ATTR_SETS.values())} attributes total).")


def _load_property_reference():
    """One-time: full attribute + type + socket truth table.
    Layer 0: cache v2 -> Layer 1: pip bpy -> Layer 2: Mixar binary."""
    global _BPy_REF_LOADED
    if _BPy_REF_LOADED:
        return
    _BPy_REF_LOADED = True

    # Layer 0: cached v2 truth table
    if os.path.isfile(REF_JSON) and os.path.isfile(REF_META_JSON):
        try:
            with open(REF_META_JSON, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("version") == TRUTH_VERSION:
                _load_from_cache()
                return
            print("⚠️ Cache is an older format (v4 or earlier) — "
                  "re-dumping the v5 attribute+socket table.")
        except Exception:
            pass

    # Layer 1: in-process pip bpy (usually unavailable on Python 3.12)
    try:
        import bpy as _b
        for name in dir(_b.types):
            if name[:1].isupper():
                cls = getattr(_b.types, name, None)
                if cls is None:
                    continue
                try:
                    alist = sorted(set(d for d in dir(cls) if not d.startswith("__")))
                except Exception:
                    continue
                if not alist:
                    continue
                tm = {}
                rna = getattr(cls, "bl_rna", None)
                if rna is not None:
                    try:
                        for p in rna.properties:
                            info = {"type": p.type or "", "min": None, "max": None, "enum": []}
                            try:
                                info["min"], info["max"] = p.min, p.max
                            except Exception:
                                pass
                            try:
                                info["enum"] = [e.identifier for e in p.enum_items]
                            except Exception:
                                pass
                            info["readonly"] = bool(getattr(p, "is_readonly", False))
                            if p.type in ("FLOAT", "INT") and getattr(p, "is_array", False):
                                info["type"] = p.type + "_ARRAY"
                            tm[p.identifier] = info
                    except Exception:
                        pass
                # dir() on an RNA type misses instance properties — merge the
                # real property identifiers in before caching.
                alist = sorted(set(alist) | set(tm.keys()))
                BPy_ATTRS[name.lower()] = alist
                BPy_ATTR_SETS[name.lower()] = set(alist)
                BPy_NAMES[name.lower()] = name
                BPy_TYPES[name.lower()] = tm
        if BPy_ATTRS:
            print(f"✅ Attribute reference from pip bpy: {len(BPy_ATTRS)} classes (no socket table).")
            return
    except Exception:
        pass

    # Layer 2: ask the Mixar binary itself, then cache the dump to disk
    if BLENDER_PATH and not BPy_ATTRS:
        print("   Dumping v2 truth table from the Mixar binary "
              "(one-time; socket probe included; cached to JSON)...")
        try:
            proc = subprocess.run(
                [BLENDER_PATH, "-b", "-noaudio", "--factory-startup",
                 "--python-expr", _BPY_DUMP_EXPR],
                capture_output=True, text=True, timeout=600,
            )
            data, tdata, sdata = None, None, None
            for line in (proc.stdout or "").splitlines():
                if line.startswith("__BPy_ATTRS__"):
                    data = json.loads(line[len("__BPy_ATTRS__"):])
                elif line.startswith("__BPy_TYPES__"):
                    tdata = json.loads(line[len("__BPy_TYPES__"):])
                elif line.startswith("__BPy_SOCKETS__"):
                    sdata = json.loads(line[len("__BPy_SOCKETS__"):])
            if data is None:
                print(f"⚠️ Mixar dump produced no data. stderr: {(proc.stderr or '')[:500]}")
                return
            if tdata:
                # dir() on an RNA type misses instance properties — merge the
                # real property identifiers in before writing the cache.
                for cname, tm in tdata.items():
                    if tm and cname in data:
                        data[cname] = sorted(set(data[cname]) | set(tm.keys()))
            for name, alist in data.items():
                key = name.lower()
                BPy_ATTRS[key] = alist
                BPy_ATTR_SETS[key] = set(alist)
                BPy_NAMES[key] = name
            with open(REF_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f)
            if tdata is not None:
                for name, tmap in tdata.items():
                    BPy_TYPES[name.lower()] = tmap
                with open(REF_TYPES_JSON, "w", encoding="utf-8") as f:
                    json.dump(tdata, f)
            if sdata is not None:
                for name, smap in sdata.items():
                    BPy_SOCKETS[name.lower()] = smap
                with open(REF_SOCK_JSON, "w", encoding="utf-8") as f:
                    json.dump(sdata, f)
            with open(REF_META_JSON, "w", encoding="utf-8") as f:
                json.dump({"version": TRUTH_VERSION}, f)
            print(f"✅ Truth table v2 from Mixar binary: {len(BPy_ATTRS)} classes, "
                  f"{len(BPy_TYPES)} type maps, {len(BPy_SOCKETS)} node classes with sockets (cached)")
        except Exception as e:
            print(f"⚠️ Mixar binary introspection failed: {e}")

    if not BPy_ATTRS:
        print("🚨 No attribute reference available — validation will be blind. "
              "Fix BLENDER_EXE or start Mixar once so the dump can run.")


def _resolve_bpy_class(candidate: str):
    """Case-insensitive lookup against every known bpy.types class."""
    _load_property_reference()
    return BPy_NAMES.get(candidate.lower())


# ============================================================
# 4. LEMONADE LLM CLIENT
# ============================================================
LEMONADE_PORT = "13305"
# Hard cap on a single Lemonade call. The openai client's default (600 s)
# lets a slow/stalled local model hold the request until Mixar's own backend
# gives up with "OpenAI did not respond in time" long before it would fail
# on its own.
LLM_TIMEOUT_S = 120.0
# How many of the most recent conversation messages are forwarded to the
# local model (see _history_tail). Completed earlier rounds are dropped so
# the prompt — and the model server's cache — never grows across rounds.
HISTORY_TAIL = 6
llm_client = openai.OpenAI(
    base_url=f"http://127.0.0.1:{LEMONADE_PORT}/v1",
    api_key="not-needed",
    timeout=LLM_TIMEOUT_S
)

MAX_ATTEMPTS = 2       # max consecutive Blender runtime failures before stopping the chain
MAX_FIX_ROUNDS = 2     # attribute-fix rounds before the hard gate blocks the script
HARD_GATE = False      # True = never send a script with unconfirmed attribute names;
                       # False = warn in the log and ship anyway (fail-forward: the
                       # pipeline keeps going and Blender reports real errors at runtime)
MAX_DISPATCH_PER_GOAL = 3   # hard cap: max script dispatches per distinct user goal
_error_attempts = 0
_tools_logged = False
_goal_state = {}            # goal key -> {"dispatches": int, "scripts": set(sha1)}


# ============================================================
# THINKING CONTROL (Qwen3 /no_think · /think tokens)
# ============================================================
def _thinking_directive():
    if THINKING_MODE == "off":
        return "\n/no_think\n"
    if THINKING_MODE == "on":
        return "\n/think\n"
    return ""


def _inject_thinking(messages):
    """Append the thinking directive to system + every user turn."""
    directive = _thinking_directive()
    if not directive:
        return messages
    out = []
    for m in messages:
        if m.get("role") in ("system", "user"):
            content = m.get("content") or ""
            if directive.strip() not in content:
                m = dict(m)
                m["content"] = content + directive
        out.append(m)
    return out


def _reasoning_of(msg):
    """Grab vendor-specific reasoning fields if the response carries them."""
    try:
        extra = getattr(msg, "model_extra", None) or {}
        for key in ("reasoning_content", "reasoning", "thought"):
            val = extra.get(key)
            if val:
                return str(val)
        for key in ("reasoning_content", "reasoning", "thought"):
            val = getattr(msg, key, None)
            if val:
                return str(val)
    except Exception:
        pass
    return ""


# ============================================================
# SCHEMAS
# ============================================================
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


# ============================================================
# AST + CHAIN-WALKING TYPE RESOLVER  (the fundamental fix)
#
# Every attribute access in the script is resolved to a concrete
# bpy.types class by walking the expression left-to-right:
#
#   root          : variable (tracked) | bpy.data.<coll> | bpy.context.<m>
#                   | a call (x.modifiers.new(...), x.nodes.new(...),
#                             bpy.data.<coll>.new(...))
#   hops          : attribute / subscript steps
#
#   STRUCT_MAP      : (class, member) -> sub-struct class   (scene.render -> RenderSettings)
#   COLLECTION_MAP  : (class, member) -> element class of a
#                     subscript/iteration                    (Mesh.materials -> Material)
#
# The resolved (cls, pending_member) is then checked:
#   * attribute access  -> attr must be in the dir(cls) truth set
#   * socket subscript  -> name must be in the socket table for that node class
#   * constant RHS      -> value type/range/enum checked against the type map
# ============================================================
def _chain_names(node):
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return list(reversed(names))


BPy_DATA_CLASSES = {
    "objects": "Object", "meshes": "Mesh", "materials": "Material",
    "lights": "Light", "cameras": "Camera", "images": "Image",
    "textures": "Texture", "node_groups": "NodeTree", "curves": "Curve",
    "fonts": "Font", "armatures": "Armature", "actions": "Action",
    "scenes": "Scene", "worlds": "World", "sounds": "Sound",
    "collections": "Collection", "brushes": "Brush", "libraries": "Library",
    "masks": "Mask", "grease_pencils": "GreasePencil", "particles": "ParticleSettings",
}

# (class, member) -> class of the sub-struct you get by reading the member
STRUCT_MAP = {
    ("Object", "data"): "Mesh",               # best-effort (mesh is the common case)
    ("Object", "active_material"): "MaterialSlot",
    ("MaterialSlot", "material"): "Material",
    ("Scene", "render"): "RenderSettings",
    ("Scene", "display"): "SceneDisplay",
    ("Scene", "view_settings"): "ColorManagedViewSettings",
    ("Scene", "camera"): "Camera",
    ("Scene", "world"): "World",
    ("Scene", "collection"): "Collection",
    ("Camera", "data"): "Camera",
    ("Light", "data"): "Light",
    ("Material", "node_tree"): "NodeTree",
    ("World", "node_tree"): "NodeTree",
    ("Node", "node_tree"): "NodeTree",
    ("PoseBone", "bone"): "Bone",
    ("ParticleSystem", "settings"): "ParticleSettings",
    ("Window", "screen"): "Screen",
}

# (class, member) -> class of ELEMENTS when subscripting/iterating the member
COLLECTION_MAP = {
    ("Object", "modifiers"): "Modifier",
    ("Object", "constraints"): "Constraint",
    ("Object", "material_slots"): "MaterialSlot",
    ("Object", "pose_bones"): "PoseBone",
    ("Object", "particle_systems"): "ParticleSystem",
    ("Object", "vertex_groups"): "VertexGroup",
    ("Mesh", "materials"): "Material",
    # This Mixar build RENAMEs the element classes (verified against the
    # binary-dumped truth table): Vertex/Edge/Face/Loop/UVMap do NOT exist.
    ("Mesh", "vertices"): "MeshVertex",
    ("Mesh", "edges"): "MeshEdge",
    ("Mesh", "loops"): "MeshLoop",
    # MeshPolygon IS in the truth table (42 attrs) — mapping it makes
    # per-polygon loop variables (for p in mesh.polygons) verifiable
    # instead of silently unchecked.
    ("Mesh", "polygons"): "MeshPolygon",
    ("Scene", "objects"): "Object",
    ("Scene", "view_layers"): "ViewLayer",
    ("Collection", "all_objects"): "Object",
    ("Collection", "objects"): "Object",
    ("Collection", "children"): "Collection",
    ("ViewLayer", "objects"): "Object",
    ("Armature", "bones"): "Bone",
    ("NodeTree", "nodes"): "Node",
    ("Node", "inputs"): "NodeSocket",
    ("Node", "outputs"): "NodeSocket",
    ("WindowManager", "windows"): "Window",
}

# bpy.context.<member> -> class of the single value / element of a collection
CONTEXT_MAP = {
    "scene": "Scene", "object": "Object", "active_object": "Object",
    "camera": "Camera", "active_camera": "Camera",
    "selected_objects": "Object", "object_data": "Mesh",
    "active_object_data": "Mesh",
    "mesh": "Mesh", "material": "Material", "texture": "Texture",
    "image": "Image", "node": "Node",
    "bone": "Bone", "active_bone": "Bone",
    "pose_bone": "PoseBone", "active_pose_bone": "PoseBone",
    "armature": "Armature", "action": "Action", "world": "World",
    "area": "Area", "region": "Region",
    "window": "Window", "screen": "Screen",
    "view_layer": "ViewLayer", "render_layer": "RenderLayer",
    "tool_settings": "ToolSettings", "window_manager": "WindowManager",
}

# non-bpy classes we must never flag (mathutils & co.)
TRUSTED_CLASSES = {
    "vector", "floatvector", "intvector", "matrix", "quat", "euler",
    "color", "floatarray", "intarray", "string",
}

_COLLECTED = "__collection__"   # sentinel: expression is a bare collection
_GENERIC_SOCKET = "__generic_socket__"  # sentinel: socket slot whose node class is not in the socket table


# bpy.data collections that do NOT support .new() in this Mixar build
# (verified by probe against mixar.exe -b: "bpy_prop_collection: attribute 'new' not found").
NO_NEW_COLLECTIONS = {"sounds"}


def _parse_parent(expr):
    """Split an expression into (root, parts) where parts is a left->right
    list of ("attr", name) / ("sub", slice_node) steps above the root.
    root = Name id, a Call node, or None."""
    parts = []
    node = expr
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute):
            parts.append(("attr", node.attr))
        else:
            parts.append(("sub", node.slice))
        node = node.value
    parts.reverse()
    if isinstance(node, ast.Name):
        return node.id, parts
    if isinstance(node, ast.Call):
        return node, parts
    return None, parts


def _socket_elem_type(node_cls, side, slice_node):
    """CONCRETE socket class for  <node>.inputs['x'] / .outputs['x']  (or None).

    side is 'inputs' or 'outputs'. Resolves through the socket-type truth
    table (dumped with the concrete bpy class of every socket); None when
    the node class or that socket has no recorded type, in which case the
    caller falls back to the generic NodeSocket base class."""
    smap = BPy_SOCKETS.get(node_cls.lower())
    if not smap:
        return None
    if not (isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str)):
        return None
    stype = (smap.get(side[:-1] + "_types") or {}).get(slice_node.value)
    if stype and _resolve_bpy_class(stype):
        return stype
    return None


def _walk_members(cls, parts):
    """Walk hops on top of a known class. Returns (cls, pending_member):
    pending None  -> expression yields a value of class cls;
    pending 'm'   -> expression is cls.<member> read (property/collection)."""
    last = None
    for kind, val in parts:
        if kind == "attr":
            s = STRUCT_MAP.get((cls, val))
            if s is not None:
                cls, last = s, None      # struct hop: class advanced
            else:
                last = val                # property or collection member: pending
        else:  # subscript
            if last is None:
                return None, None
            el = COLLECTION_MAP.get((cls, last))
            if el is None and last in ("inputs", "outputs") and cls.lower() in BPy_SOCKETS:
                # concrete socket type when known (lets .default_value /
                # .hide_value etc. check against the REAL socket class, not
                # just the NodeSocket base)
                el = _socket_elem_type(cls, last, val) or "NodeSocket"
            if el is None:
                return None, None
            if el == "NodeSocket" and cls.lower() not in BPy_SOCKETS:
                # The node class was NOT in the socket table, so this
                # "NodeSocket" is a fallback guess — the concrete slot
                # type is unknowable. Property checks on the sentinel
                # defer to the whole-socket-family check.
                el = _GENERIC_SOCKET
            cls, last = el, None
    return cls, last


def _string_kw_or_arg(call, kw_name, positional_index):
    """Extract a string argument: keyword first, then Nth positional.
    SAFE: returns None for any non-Call node."""
    if not isinstance(call, ast.Call):
        return None
    for kw in call.keywords:
        if kw.arg == kw_name and isinstance(kw.value, ast.Constant) \
                and isinstance(kw.value.value, str):
            return kw.value.value
    if len(call.args) > positional_index \
            and isinstance(call.args[positional_index], ast.Constant) \
            and isinstance(call.args[positional_index].value, str):
        return call.args[positional_index].value
    return None


def _infer_call_class(call):
    """What bpy.types class does this call RETURN? (or None)"""
    if not isinstance(call, ast.Call):
        return None
    chain = _chain_names(call.func)
    if not chain:
        return None
    if chain[0] == "bpy" and "ops" in chain:
        return None
    # bpy.data.<collection>.new(...) -> element of that collection
    if chain[:2] == ["bpy", "data"] and len(chain) >= 3:
        return BPy_DATA_CLASSES.get(chain[2])
    # x.modifiers.new(type='<TYPE>') -> <Type>Modifier
    if "modifiers" in chain and chain[-1] in ("new", "append"):
        type_val = _string_kw_or_arg(call, "type", 1)
        if type_val:
            t = re.sub(r"[-_ ]", "_", type_val.strip().upper())
            cand = "".join(w[:1].upper() + w[1:].lower() for w in t.split("_") if w) + "Modifier"
            return _resolve_bpy_class(cand) or cand
        return None
    # x.nodes.new(type='<NodeTypeName>') -> that node class
    if "nodes" in chain and chain[-1] == "new":
        type_val = _string_kw_or_arg(call, "type", 0)
        if type_val:
            return _resolve_bpy_class(type_val) or type_val
        return None
    return None


def _state_of(root, parts, var_map=None):
    """(cls, pending_member) state for the expression root+parts."""
    if root is None:
        return None, None
    if isinstance(root, ast.Call):
        cls = _infer_call_class(root)
        if cls is None:
            return None, None
        return _walk_members(cls, parts)
    if root == "bpy":
        if not parts or parts[0][0] != "attr":
            return None, None
        first = parts[0][1]
        if first == "ops":
            return None, None              # operator tree is dynamic — never check
        if first == "data" and len(parts) >= 2 and parts[1][0] == "attr":
            el = BPy_DATA_CLASSES.get(parts[1][1])
            if el is None:
                return None, None
            if len(parts) == 2:
                return (_COLLECTED, el)    # bare collection; element class = el
            rest = parts[2:]
            if rest[0][0] == "sub":
                return _walk_members(el, rest[1:])
            return _walk_members(el, rest)
        if first == "context" and len(parts) >= 2 and parts[1][0] == "attr":
            el = CONTEXT_MAP.get(parts[1][1])
            if el is None:
                return None, None
            if len(parts) == 2:
                return (_COLLECTED, el)    # bare collection (selected_objects & co.)
            rest = parts[2:]
            if rest[0][0] == "sub":
                return _walk_members(el, rest[1:])
            return _walk_members(el, rest)
        return None, None
    # plain variable
    cls = (var_map or {}).get(root)
    if cls is None:
        return None, None
    return _walk_members(cls, parts)


def _expr_class(value, var_map):
    """bpy.types class produced by this expression (or None)."""
    if isinstance(value, ast.Call):
        cls = _infer_call_class(value)
        if cls:
            return cls
    root, parts = _parse_parent(value)
    if root is None:
        return None
    cls, last = _state_of(root, parts, var_map)
    if cls is None:
        return None
    if cls == _COLLECTED:
        return last                          # variable holds a collection -> element
    if last is None:
        return cls
    return COLLECTION_MAP.get((cls, last))


def _elem_class(value, var_map):
    """Class of the elements yielded when iterating this expression."""
    if isinstance(value, ast.Call):
        return _infer_call_class(value)
    root, parts = _parse_parent(value)
    if root is None:
        return None
    cls, last = _state_of(root, parts, var_map)
    if cls is None:
        return None
    if cls == _COLLECTED:
        return last
    if last is None:
        return cls                           # e.g. iterating bpy.context.selected_objects
    return COLLECTION_MAP.get((cls, last))


def _expr_is_collection(value, var_map):
    """True if this expression evaluates to a collection OBJECT ITSELF
    (bpy.data.objects, mat.node_tree.nodes, node.inputs, ...) rather
    than to a single element. .new/.remove/.clear/.get on such a variable
    are collection APIs — not element properties — so pass 1 must not
    validate them against the element class."""
    if isinstance(value, ast.Call):
        return False
    root, parts = _parse_parent(value)
    if root is None:
        return False
    cls, last = _state_of(root, parts, var_map)
    if cls == _COLLECTED:
        return True
    if cls is None or last is None:
        return False
    if COLLECTION_MAP.get((cls, last)) is not None:
        return True
    return last in ("inputs", "outputs") and cls.lower() in BPy_SOCKETS


# ============================================================
# VALUE-TYPE / RANGE / ENUM CHECKER
# ============================================================
def _type_issue(cls_name, attr, info, value_node):
    """Check a CONSTANT right-hand side against the property's declared
    type, range and enum values. Dynamic values are NOT checked.
    Returns an issue string or None."""
    if not info or "type" not in info:
        return None
    ptype = (info.get("type") or "").upper()
    if not ptype:
        return None
    if isinstance(value_node, (ast.Tuple, ast.List)):
        # sequence literal on the RHS (e.g. attr = (0, 1)).
        # Requires the v4 truth table: array properties are dumped as
        # FLOAT_ARRAY / INT_ARRAY, so a plain FLOAT/INT here is a true scalar.
        if ptype in ("INT_ARRAY", "FLOAT_ARRAY", "FLOAT_VECTOR", "CHAR_ARRAY"):
            return None          # a tuple is the right shape for an array/vector slot
        if ptype in ("INT", "FLOAT"):
            return (f"- {attr}: bpy.types.{cls_name}.{attr} expects a single "
                    f"{'integer' if ptype == 'INT' else 'number'} (you wrote a "
                    f"tuple). Pass one value, e.g. {attr} = 1.5")
        if ptype == "BOOLEAN":
            return (f"- {attr}: bpy.types.{cls_name}.{attr} expects True/False "
                    f"(you wrote a tuple).")
        if ptype in ("STRING", "ENUM"):
            return (f"- {attr}: bpy.types.{cls_name}.{attr} expects a string "
                    f"(you wrote a tuple).")
        return None
    if not isinstance(value_node, ast.Constant):
        return None
    v = value_node.value

    if ptype == "INT":
        if isinstance(v, bool):
            return f"- {attr}: bpy.types.{cls_name}.{attr} expects an int (you wrote a bool)."
        if isinstance(v, float):
            lo, hi = info.get("min"), info.get("max")
            rng = f" (legal range: {lo}..{hi})" if lo is not None and hi is not None else ""
            return (f"- {attr}: bpy.types.{cls_name}.{attr} is an INT in this Mixar build "
                    f"(you wrote the float {v!r}). Use an integer literal within the legal range"
                    + rng + ".")
    elif ptype == "FLOAT":
        pass  # int or float both accepted
    elif ptype in ("INT_ARRAY", "FLOAT_ARRAY", "FLOAT_VECTOR", "CHAR_ARRAY"):
        if not isinstance(v, (list, tuple)):
            hint = " (e.g. (x, y, z))" if ptype == "FLOAT_VECTOR" else " (e.g. (a, b))"
            return (f"- {attr}: bpy.types.{cls_name}.{attr} expects a sequence{hint}, "
                    f"not a single {type(v).__name__}.")
    elif ptype == "BOOL":
        if not isinstance(v, bool):
            return f"- {attr}: bpy.types.{cls_name}.{attr} expects True/False (you wrote {v!r})."
    elif ptype == "ENUM":
        if not isinstance(v, str):
            return f"- {attr}: bpy.types.{cls_name}.{attr} expects an enum string (you wrote {v!r})."
        enums = info.get("enum") or []
        if enums and v not in enums:
            close = get_close_matches(v, enums, n=3, cutoff=0.4)
            shown = ", ".join(enums[:12]) + ("..." if len(enums) > 12 else "")
            return (f"- {attr}: '{v}' is not a valid enum value for bpy.types.{cls_name}.{attr}. "
                    f"Valid: {shown}. "
                    + (f"Did you mean: {', '.join(close)}?" if close else ""))
        return None
    elif ptype == "STRING":
        if not isinstance(v, str):
            return f"- {attr}: bpy.types.{cls_name}.{attr} expects a string (you wrote {v!r})."

    # numeric range check for INT / FLOAT
    if ptype in ("INT", "FLOAT") and isinstance(v, (int, float)) and not isinstance(v, bool):
        lo, hi = info.get("min"), info.get("max")
        if lo is not None and v < lo:
            return f"- {attr}: value {v!r} is below the minimum {lo!r} allowed by bpy.types.{cls_name}."
        if hi is not None and v > hi:
            return f"- {attr}: value {v!r} is above the maximum {hi!r} allowed by bpy.types.{cls_name}."
    return None


# struct-pointer members that LEGALLY take ['name'] (e.g. obj.pose['Bone']),
# plus collection-typed members this build's type map records as plain
# methods (they are in dir() but not in the RNA property map).
_SUBSCRIPTABLE_POINTERS = {"pose"}
_NONPROP_COLLECTIONS = {"children", "children_recursive", "users_scene", "users_collection"}


def _subscript_target_issue(cls, member, node):
    """Validate WHAT a Subscript node is subscripting into — not just the key.

    The attribute pass already proved cls.<member> EXISTS; this checks whether
    the member can actually be indexed. Methods and scalars exist too, yet die
    at runtime ('builtin_function_or_method object does not support item
    assignment' / 'float object is not subscriptable'). member is None when a
    whole class instance is subscripted (mesh['k'] — legal custom-property
    access on data-blocks, illegal on non-Id structs). Returns an issue or None."""
    key = cls.lower()
    attrset = BPy_ATTR_SETS.get(key) or set()
    if member is None:
        # data-blocks in this build expose a dict API (get/keys/values/items/
        # pop) for custom properties, so  id['custom_prop'] = ...  is legal;
        # classes WITHOUT that API cannot be indexed at all
        if "get" in attrset:
            return None
        return (f"- {cls} does not support ['...'] item access — it has no "
                f"custom-property dict API in this Mixar build. Set its properties "
                f"directly (e.g. <{cls.lower()}>.<property> = value) instead of "
                f"indexing the object itself.")
    info = (BPy_TYPES.get(key) or {}).get(member)
    if info is None:
        if member in _NONPROP_COLLECTIONS:
            return None   # RNA collection the type map recorded as a method
        if member in attrset:
            return (f"- {member}: bpy.types.{cls}.{member} is a METHOD, not a "
                    f"collection or a property — methods can never be subscripted "
                    f"(the engine kills it at runtime with \"'builtin_function_or_method' "
                    f"object does not support item assignment\"). Call it with "
                    f"parentheses to use its return value, or set the data property "
                    f"it manages instead of indexing it.")
        return None   # the attribute pass already reported the missing member
    ptype = (info.get("type") or "").upper()
    if ptype == "COLLECTION":
        return None
    if ptype in ("FLOAT_ARRAY", "INT_ARRAY"):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return (f"- {member}: bpy.types.{cls}.{member} is a numeric array, not a "
                    f"dict — a string key is not a valid index. Use a numeric index "
                    f"(e.g. {member}[0]) or assign a tuple to the whole property "
                    f"({member} = (x, y, z)).")
        return None
    if ptype == "POINTER":
        if member in _SUBSCRIPTABLE_POINTERS:
            return None
        target = STRUCT_MAP.get((cls, member))
        if target is None:
            return None               # unknown target -> can't decide, don't guess
        if "get" in (BPy_ATTR_SETS.get(target.lower()) or set()):
            return None               # pointed-to data-block has the dict API
        return (f"- {member}: bpy.types.{cls}.{member} points at a {target} struct, "
                f"which has no ['...'] item access in this build. Reach it first and "
                f"set its properties directly (e.g. obj.{member}.<property> = value).")
    kind = {"FLOAT": "a number", "INT": "an integer", "BOOLEAN": "True/False",
            "STRING": "a string", "ENUM": "an enum string"}.get(ptype, "a single value")
    ex = {"FLOAT": "0.0", "INT": "0", "BOOLEAN": "True", "STRING": '"text"'}.get(ptype)
    if ptype == "ENUM" and (info.get("enum") or []):
        ex = repr(info["enum"][0])
    ex_s = f" (e.g. {member} = {ex})" if ex else ""
    return (f"- {member}: bpy.types.{cls}.{member} is {kind}, not a collection — "
            f"it can't be subscripted. Assign it directly{ex_s} "
            f"instead of {member}[...].")


# ============================================================
# WRITE-LEGALITY CHECK  (read-only RNA props + non-RNA attributes)
# ============================================================
# Probe of mixar.exe (2026-09-30, all 4041 bpy.types classes): ZERO
# dir()-level class attributes outside the RNA property map are
# writable — socket value aliases (values/vector/color/...), methods,
# descriptors and python properties are ALL read-only, and the engine
# rejects an assignment to any of them with
#   "'NodeSocketColor' object attribute 'values' is read-only".
# Whitelist of proven-writable non-RNA attributes (empty today).
_NONRNA_WRITABLE = set()


def _writable_props(cls_name, limit=10):
    tmap = BPy_TYPES.get(cls_name.lower()) or {}
    out = [a for a in sorted(tmap)
           if isinstance(tmap.get(a), dict) and not tmap[a].get("readonly")]
    return out[:limit]


def _write_issue(cls_name, attr):
    """Is  <cls_name>.<attr> = ...  legal in this build? Issue string or None.

    Two layers, both ground-truthed against the Mixar binary:
      * attr is an RNA property -> flagged only when it is readonly;
      * attr is NOT an RNA property (socket alias / method / descriptor —
        it passed the existence check because dir() lists it) -> flagged,
        because the engine rejects the assignment at runtime."""
    tmap = BPy_TYPES.get(cls_name.lower()) or {}
    if not tmap:
        return None          # no type map for this class -> can't decide
    info = tmap.get(attr)
    if info is None:
        if attr in _NONRNA_WRITABLE:
            return None
        wr = _writable_props(cls_name)
        return (f"- {attr}: bpy.types.{cls_name}.{attr} is NOT a writable property in this "
                f"Mixar build (read-only alias, method or descriptor — the engine kills the "
                f"assignment with \"'{cls_name}' object attribute '{attr}' is read-only\"). "
                f"Delete that line or set one of the writable properties instead: "
                f"{', '.join(wr) if wr else '(none on this class)'}")
    if info.get("readonly"):
        wr = _writable_props(cls_name)
        return (f"- {attr}: bpy.types.{cls_name}.{attr} is READ-ONLY in this Mixar build "
                f"(the engine computes it). Never assign to it — delete that line and "
                f"achieve the visual effect through the writable properties that drive it"
                + (f" (e.g. {', '.join(wr[:6])})." if wr else "."))
    return None


def _socket_iter_write_issue(node_cls, side, attr):
    """Write check for a socket variable whose concrete type is UNKNOWN
    (it comes from  for s in <node>.inputs/outputs ): legal only if EVERY
    recorded socket type on that node side accepts the write."""
    smap = BPy_SOCKETS.get(node_cls.lower()) or {}
    stypes = [t for t in (smap.get(side[:-1] + "_types") or {}).values() if t]
    tmaps = [tm for tm in (BPy_TYPES.get(t.lower()) for t in stypes) if tm]
    if not tmaps:
        return None          # socket table has no type info -> don't guess
    for tm in tmaps:
        info = tm.get(attr)
        if info is None or info.get("readonly"):
            break
    else:
        return None           # writable on every socket type of that side
    names = smap.get(side) or []
    return (f"- {attr}: a {node_cls} socket has no writable '{attr}' property in this Mixar "
            f"build (socket value slots like values/vector/color are read-only aliases — the "
            f"engine kills the assignment). Address the slot BY NAME and set a writable "
            f"property of it, e.g. {node_cls.lower()}.{side}['<socket>'].default_value = "
            f"(r, g, b, a) — but only if that socket's type lists default_value in its "
            f"valid properties"
            + (f". Sockets on {node_cls}: {', '.join(names[:12])}." if names else "."))


# ============================================================
# ATTRIBUTE VALIDATOR  (single definition — truth table only)
# ============================================================
_TRUTH_SEP = "| valid properties:"


def _generic_socket_access_issue(attr, is_write):
    """Check a property access on a socket slot whose CONCRETE type is
    unknown: the owning node's class is absent from the socket table
    (e.g. a plain  for node in nodes  — node is the generic Node class),
    so the slot resolved to the bare NodeSocket base. That base entry is
    a strict subset of the concrete types (it lacks default_value in this
    build), so the verdict goes against the WHOLE socket family:
      read:  legal if ANY socket type has the attribute;
      write: legal if SOME socket type accepts it (the engine resolves
             the slot's real type at runtime); flagged only when NO
             socket type makes the access legal.
    Returns an issue string or None."""
    sock_maps = [tm for k, tm in BPy_TYPES.items() if k.startswith("nodesocket")]
    generic_set = BPy_ATTR_SETS.get("nodesocket") or set()
    if attr not in generic_set:
        if not any(attr in tm for tm in sock_maps):
            return (f"- {attr}: no socket type in this Mixar build has a '{attr}' "
                    f"attribute - this slot access dies at runtime. Address the slot "
                    f"by name and use a property its type lists (e.g. default_value, "
                    f"hide, hide_value, enabled, description).")
        return None
    if not is_write:
        return None
    def _ok(tm):
        info = tm.get(attr)
        return isinstance(info, dict) and not info.get("readonly")
    if any(_ok(tm) for tm in sock_maps):
        return None
    return (f"- {attr}: no socket type in this Mixar build has a writable '{attr}' "
            f"property (socket value slots like values/vector/color are read-only "
            f"aliases - the engine kills the assignment). Address the slot BY NAME "
            f"and set a writable property of it, e.g. node.inputs['<socket>']"
            f".default_value = (r, g, b, a) - only if that socket's type lists "
            f"default_value in its valid properties.")


def validate_bpy_properties(script: str):
    """AST-scan the script; return a list of human-readable issues.

    EVERY attribute access (read or write) is resolved to a bpy.types class
    via the chain-walking resolver and checked against the binary-dumped
    dir() truth set. Socket-name subscripts are checked against the socket
    table; constant RHS values are checked for type/range/enum.
    No documentation search is involved — the truth table is the sole
    authority for this build."""
    _load_property_reference()
    issues, seen = [], set()

    def add(msg):
        if msg not in seen:
            seen.add(msg)
            issues.append(msg)

    try:
        tree = ast.parse(script)
    except SyntaxError:
        return []

    # ---- Pass 0: variable -> class tracking (source order) ----
    raw_events = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.value is not None:
            raw_events.append((node.lineno, node.col_offset, node.targets[0].id, "assign", node.value))
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            raw_events.append((node.lineno, node.col_offset, node.target.id, "iter", node.iter))
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            if node.generators and isinstance(node.generators[0].target, ast.Name):
                raw_events.append((node.lineno, node.col_offset,
                                   node.generators[0].target.id, "iter", node.generators[0].iter))
    var_class = {}
    var_colls = set()      # vars that hold a collection object (see _expr_is_collection)
    for _ln, _col, name, kind, value in sorted(raw_events):
        if kind == "assign":
            if _expr_is_collection(value, var_class):
                var_colls.add(name)
            else:
                var_colls.discard(name)
        cls = _elem_class(value, var_class) if kind == "iter" else _expr_class(value, var_class)
        if cls:
            var_class[name] = cls

    # socket variables born from  for s in <node>.inputs/outputs  (or from a
    # variable holding such a collection) — their concrete socket class is
    # unknowable, so their writes are checked against EVERY socket type of
    # that node side (see _socket_iter_write_issue).
    coll_socket_vars = {}    # var holding <node>.inputs/outputs -> (node_cls, side)
    for _ln, _col, name, kind, value in sorted(raw_events):
        if kind != "assign":
            continue
        root_v, parts_v = _parse_parent(value)
        if isinstance(root_v, str) and len(parts_v) == 1 and parts_v[0][0] == "attr" \
                and parts_v[0][1] in ("inputs", "outputs"):
            nclsv = var_class.get(root_v)
            if nclsv and nclsv.lower() in BPy_SOCKETS:
                coll_socket_vars[name] = (nclsv, parts_v[0][1])
    sock_iters = {}          # loop var name -> (node_cls, side)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.For) and isinstance(node.target, ast.Name)):
            continue
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) \
                and it.func.id == "list" and it.args:
            it = it.args[0]
        root, parts = _parse_parent(it)
        if not isinstance(root, str):
            continue
        if root in coll_socket_vars:
            sock_iters[node.target.id] = coll_socket_vars[root]
        elif len(parts) == 1 and parts[0][0] == "attr" and parts[0][1] in ("inputs", "outputs"):
            ncls = var_class.get(root)
            if ncls and ncls.lower() in BPy_SOCKETS:
                sock_iters[node.target.id] = (ncls, parts[0][1])

    # Map attribute-write targets -> their RHS value node (for type checks)
    assign_rhs = {}
    readonly_targets = {}   # id(leaf attr node of a WRITE TARGET) -> readonly issue text
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            tgt = node.targets[0] if len(node.targets) == 1 else None
        elif isinstance(node, ast.AugAssign):
            tgt = node.target
        else:
            tgt = None
        if tgt is not None and isinstance(tgt, ast.Attribute) and node.value is not None:
            assign_rhs[id(tgt)] = node.value
            try:
                root, parts = _parse_parent(tgt)
                cls_r, leaf_r = _state_of(root, parts, var_class)
                if cls_r and cls_r != _COLLECTED and leaf_r:
                    wi = _write_issue(cls_r, leaf_r)
                    if wi:
                        readonly_targets[id(tgt)] = wi
            except Exception:
                pass

    # ---- Pass 1: check every attribute access + socket subscript ----
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            root, parts = _parse_parent(node.value)
            if isinstance(root, str) and not parts and root in var_colls:
                continue   # holds a collection: .new/.remove/etc. are collection APIs
            if isinstance(root, str) and not parts and root in sock_iters:
                # socket from iteration: concrete type unknown -> only WRITEs
                # are checkable (against every socket type of that node side)
                if id(node) in assign_rhs or id(node) in readonly_targets:
                    ncls_i, side_i = sock_iters[root]
                    wi = _socket_iter_write_issue(ncls_i, side_i, node.attr)
                    if wi:
                        add(wi)
                continue
            cls, last = _state_of(root, parts, var_class)
            if cls is None or cls == _COLLECTED or last is not None:
                continue                     # parent not a known bpy class -> don't guess
            key = cls.lower()
            if key in TRUSTED_CLASSES:
                continue
            attr = node.attr
            if cls == _GENERIC_SOCKET:
                gi = _generic_socket_access_issue(attr, id(node) in assign_rhs)
                if gi:
                    add(gi)
                else:
                    checked += 1
                continue
            attrset = BPy_ATTR_SETS.get(key)
            if attrset is None:
                # The class name itself is not in the truth table — flag the class.
                close = get_close_matches(key, list(BPy_NAMES.keys()), n=3, cutoff=0.4)
                add(f"- bpy.types.{cls} is not a class in this Mixar build (cannot verify '{attr}')."
                    + (f" Did you mean: {', '.join(BPy_NAMES[c] for c in close)}?" if close else ""))
                continue
            if not attrset:
                continue   # class in truth table but zero recorded attrs -> nothing to verify
            if attr not in attrset:
                close = get_close_matches(attr, sorted(attrset), n=3, cutoff=0.45)
                add(f"- {attr}: bpy.types.{cls} has no attribute '{attr}' in this Mixar build."
                    + (f" Did you mean: {', '.join(close)}?" if close else ""))
                continue
            checked += 1
            # value type/range/enum on constant RHS
            rhs = assign_rhs.get(id(node))
            if rhs is not None:
                wi = _write_issue(cls, attr)
                if wi:
                    add(wi)
                    continue
                info = (BPy_TYPES.get(key) or {}).get(attr)
                if info:
                    t = _type_issue(cls, attr, info, rhs)
                    if t:
                        add(t)
            else:
                # deep write target (e.g. bpy.context.scene.display.shading = ...)
                if id(node) in readonly_targets:
                    add(readonly_targets[id(node)])
        elif isinstance(node, ast.For):
            # unscoped object-delete loop (the Mixar engine blocks it at runtime)
            it = node.iter
            if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) \
                    and it.func.id == "list" and it.args:
                it = it.args[0]
            names = _chain_names(it)
            if names in (["bpy", "data", "objects"],
                         ["bpy", "context", "view_layer", "objects"],
                         ["bpy", "context", "scene", "collection", "all_objects"]):
                removes = any(
                    isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "remove"
                    and _chain_names(c.func.value) == ["bpy", "data", "objects"]
                    for c in ast.walk(node))
                guarded = any(isinstance(s, ast.If) for s in node.body)
                if removes and not guarded:
                    add("- unscoped object-delete loop: the Mixar engine blocks it at runtime "
                         "with 'Blocked: unscoped delete' (bpy.data is shared by every scene in "
                         "this session). Rewrite it with a positive name guard - one of:\n"
                         "  (a) for n in ['Exact_Name_1', 'Exact_Name_2']:\n"
                         "        o = bpy.context.scene.objects.get(n)\n"
                         "        if o: bpy.data.objects.remove(o, do_unlink=True)\n"
                         "  (b) for obj in list(bpy.context.scene.collection.all_objects):\n"
                         "        if obj.name.startswith('Your_Prefix_'):\n"
                         "            bpy.data.objects.remove(obj, do_unlink=True)")
        elif isinstance(node, ast.Call):
            # .new() on collections this build does not support (bpy.data.sounds.new, ...)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "new":
                names = _chain_names(node.func.value)
                if len(names) >= 3 and names[0] == "bpy" and names[1] == "data" \
                        and names[2] in NO_NEW_COLLECTIONS:
                    add(f"- bpy.data.{names[2]}.new: this collection has no .new in this Mixar build "
                         f"(Mixar kills the call at runtime). DELETE those lines - there are no sound "
                         f"data-blocks or audio emitters in this build; configure audio at scene level: "
                         f"scene.use_audio, scene.audio_volume, scene.audio_distance_model, "
                         f"scene.audio_doppler_factor, scene.audio_doppler_speed, scene.use_audio_scrub.")
                # typed-collection .new(): the type STRING must name a class that exists in
                # this build, or Mixar dies with "Node type X undefined" / a bad modifier type
                type_str = None
                if "nodes" in names:
                    # x.nodes.new('ShaderNodeBump') or nodes.new(type='...')
                    for kw in node.keywords:
                        if kw.arg in ("type", "bl_idname") and isinstance(kw.value, ast.Constant) \
                                and isinstance(kw.value.value, str):
                            type_str = kw.value.value
                    if type_str is None and node.args and isinstance(node.args[0], ast.Constant) \
                            and isinstance(node.args[0].value, str):
                        type_str = node.args[0].value
                elif "modifiers" in names:
                    # x.modifiers.new('Name', 'OCEAN') or new(name, type='...')
                    for kw in node.keywords:
                        if kw.arg == "type" and isinstance(kw.value, ast.Constant) \
                                and isinstance(kw.value.value, str):
                            type_str = kw.value.value
                    if type_str is None and len(node.args) >= 2 \
                            and isinstance(node.args[1], ast.Constant) \
                            and isinstance(node.args[1].value, str):
                        type_str = node.args[1].value
                if type_str and type_str[:1].isupper():
                    if "modifiers" in names:
                        t = re.sub(r"[-_ ]", "_", type_str.strip().upper())
                        cand = "".join(w[:1].upper() + w[1:].lower() for w in t.split("_") if w) + "Modifier"
                        key, label = cand.lower(), cand
                    else:
                        key, label = type_str.lower(), type_str
                    if key not in BPy_NAMES:
                        close = get_close_matches(key, list(BPy_NAMES.keys()), n=3, cutoff=0.4)
                        add(f"- {label}: {label} is not a class in this Mixar build "
                             f"(.new('{type_str}') fails at runtime with \"type undefined\")."
                             + (f" Did you mean: {', '.join(BPy_NAMES[c] for c in close)}?" if close else ""))
        elif isinstance(node, ast.Subscript):
            # socket-name validation + subscript-TARGET validation
            root, parts = _parse_parent(node.value)
            cls, last = _state_of(root, parts, var_class)
            if cls and cls != _COLLECTED and last in ("inputs", "outputs"):
                # socket-name validation:  someNode.inputs['...'] / outputs['...']
                if not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
                    continue
                smap = BPy_SOCKETS.get(cls.lower())
                if not smap:
                    continue         # generic Node or unmapped class -> can't verify
                names = smap.get(last) or []
                if names and node.slice.value not in names:
                    close = get_close_matches(node.slice.value, names, n=3, cutoff=0.4)
                    add(f"- {node.slice.value}: not an input/output socket on {cls}. "
                        f"Available sockets: {', '.join(names)}. "
                        + (f"Did you mean: {', '.join(close)}?" if close else ""))
            elif cls and cls != _COLLECTED and cls != _GENERIC_SOCKET:
                # existence (checked by the attribute pass) is not subscriptability:
                # methods, scalars and non-Id structs die at runtime when indexed
                t = _subscript_target_issue(cls, last, node)
                if t:
                    add(t)

    print(f"🔍 Validated {checked} attribute access(es) against the Mixar truth table "
          f"({len(BPy_ATTRS)} classes, {len(BPy_TYPES)} type maps, "
          f"{len(BPy_SOCKETS)} socket tables).")
    return issues[:20]


# ============================================================
# MIXAR NORMALIZER — transform stock-Blender idioms into the
# equivalents this Mixar build actually accepts, BEFORE any gate
# runs. The gates below still validate the normalized script and
# catch whatever the normalizer cannot rewrite.
# ============================================================
def _socket_subscript(node):
    """True for <something>.inputs['name'] / <something>.outputs['name']."""
    return (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr in ("inputs", "outputs")
            and isinstance(node.slice, ast.Constant))


def _dead_new_call(node):
    """True for the Call node itself: bpy.data.<unsupported collection>.new(...)"""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "new":
        names = _chain_names(node.func.value)
        if len(names) >= 3 and names[:2] == ["bpy", "data"] \
                and names[2] in NO_NEW_COLLECTIONS:
            return True
    return False


def normalize_mixar_script(script):
    """Rewrite known stock-Blender idioms into Mixar-accepted equivalents.

    Returns (script, notes). notes are human-readable transform reports
    (printed to the server log); an empty notes list means the script was
    returned byte-identical. Every rule mirrors a real error reproduced
    against mixar.exe:

    T1  socket .values = ...           -> socket .default_value = ...
         ONLY when the socket's CONCRETE type is resolvable from the truth
         table (node class + socket name) and that type actually has
         default_value in this build. Untyped sockets keep .values (it
         exists on every socket); a blind rewrite produced default_value on
         sockets that lack it (NodeSocketGeometry/Matrix/Shader, ...) which
         the gate then rejected.
    T2  bpy.data.sounds.new(...) usage -> lines deleted, scene-level
        audio inserted (no sound data-blocks in this build:
         attribute 'new' not found)
    """
    notes = []
    if not script or not script.strip():
        return script, notes
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return script, notes        # Gate 1 owns syntax errors

    changed = False

    # ---- T1: socket .values -> .default_value ----------------------------
    # Only when the CONCRETE socket type is known AND its truth-table entry
    # actually has default_value. A blind rewrite produced .default_value on
    # sockets that lack it (NodeSocketGeometry/Matrix/Shader/...) and on
    # untyped sockets the gate resolves to the base NodeSocket — which also
    # lacks default_value in this build — so the rewrite only ever made
    # scripts MORE likely to be rejected.
    node_class_of = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)):
            c = _infer_call_class(node.value)
            if c:
                node_class_of[node.targets[0].id] = c

    def _node_io_info(expr):
        """(node_class, 'inputs'|'outputs') for  <obj>.inputs / <obj>.outputs."""
        if not (isinstance(expr, ast.Attribute)
                and expr.attr in ("inputs", "outputs")):
            return None
        c = None
        if isinstance(expr.value, ast.Call):
            c = _infer_call_class(expr.value)
        elif isinstance(expr.value, ast.Name):
            c = node_class_of.get(expr.value.id)
        return (c, expr.attr) if c else None

    def _socket_info(expr):
        """(node_class, side, socket_name) for a socket expression, or None."""
        if isinstance(expr, ast.Subscript) and _socket_subscript(expr):
            info = _node_io_info(expr.value)
            if info:
                return (info[0], info[1], expr.slice.value)
        elif isinstance(expr, ast.Name) and expr.id in socket_vars:
            return socket_vars[expr.id]
        return None

    socket_vars = {}    # var name -> (node_class|None, side, socket_name)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and _socket_subscript(node.value)):
            info = _node_io_info(node.value.value)
            if info:
                socket_vars[node.targets[0].id] = (info[0], info[1], node.value.slice.value)
            else:
                socket_vars[node.targets[0].id] = (None, None, None)

    # socket variables born from  for s in <node>.inputs/outputs  — the
    # concrete slot is unknown, so the rewrite is allowed only when EVERY
    # socket type on that node side has default_value in this build.
    sock_iters = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            it = node.iter
            if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) \
                    and it.func.id == "list" and it.args:
                it = it.args[0]
            if isinstance(it, ast.Attribute) and it.attr in ("inputs", "outputs") \
                    and isinstance(it.value, ast.Name):
                c = node_class_of.get(it.value.id)
                if c and c.lower() in BPy_SOCKETS:
                    sock_iters[node.target.id] = (c, it.attr)

    def _has_dv(t):
        return "default_value" in (BPy_ATTR_SETS.get(t.lower()) or set())

    n_vals = n_keep = 0
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for tgt in targets:
            if not (isinstance(tgt, ast.Attribute) and tgt.attr == "values"):
                continue
            stypes = []
            info = _socket_info(tgt.value)
            if info:
                ncls, side, name = info
                if ncls and BPy_ATTR_SETS:
                    smap = BPy_SOCKETS.get(ncls.lower())
                    if smap:
                        st = (smap.get(side[:-1] + "_types") or {}).get(name)
                        if st:
                            stypes = [st]
            elif isinstance(tgt.value, ast.Name) and tgt.value.id in sock_iters:
                ncls, side = sock_iters[tgt.value.id]
                smap = BPy_SOCKETS.get(ncls.lower()) or {}
                stypes = [t for t in (smap.get(side[:-1] + "_types") or {}).values() if t]
            if not stypes:
                n_keep += 1     # type unconfirmed -> keep .values (always exists)
                continue
            if all(_has_dv(t) for t in stypes):
                tgt.attr = "default_value"
                n_vals += 1
            else:
                n_keep += 1     # some socket type on this side lacks default_value
    if n_vals:
        changed = True
        notes.append(f"{n_vals} socket .values assignment(s) rewritten to "
                     f".default_value (confirmed present on the socket type)")
    if n_keep:
        notes.append(f"{n_keep} socket .values assignment(s) left untouched "
                     f"(socket type not confirmed to have default_value)")

    # ---- T2: sound data-blocks -> scene-level audio -----------------------
    sounds_vars = set()
    bare_sounds = False
    for stmt in tree.body:
        for sub in ast.walk(stmt):
            if _dead_new_call(sub):
                bare_sounds = True
                if isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Name):
                            sounds_vars.add(t.id)
    if bare_sounds:
        def _kills(stmt):
            if any(_dead_new_call(sub) for sub in ast.walk(stmt)):
                return True
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Attribute) \
                            and isinstance(t.value, ast.Name) \
                            and t.value.id in sounds_vars:
                        return True
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute) \
                        and isinstance(func.value, ast.Name) \
                        and func.value.id in sounds_vars:
                    return True
            return False

        kept, n_dead = [], 0
        for stmt in tree.body:
            if _kills(stmt):
                n_dead += 1
                continue
            kept.append(stmt)
        if n_dead:
            if not re.search(r"\buse_audio\b", script):
                kept.append(ast.parse("bpy.context.scene.use_audio = True").body[0])
            changed = True
            notes.append(f"{n_dead} sound-data line(s) removed (no sound data-blocks "
                         f"in this build) - replaced with scene-level audio "
                         f"(bpy.context.scene.use_audio = True)")
        tree.body = kept

    # ---- T3: use_nodes (deprecated shim - node trees are always on) ------
    class _StripUseNodes(ast.NodeTransformer):
        def __init__(self):
            super().__init__()
            self.n = 0
        def _fix(self, st):
            if (isinstance(st, ast.Assign) and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Attribute)
                    and st.targets[0].attr == "use_nodes"):
                self.n += 1
                return None
            # containers that lost their only statement need repair
            if isinstance(st, ast.If) and not st.body:
                if st.orelse:
                    st.body = [ast.parse("pass").body[0]]
                else:
                    return None            # no-op if -> drop entirely
            if isinstance(st, (ast.For, ast.While, ast.Try)) and not st.body:
                st.body = [ast.parse("pass").body[0]]
            return st
        def generic_visit(self, node):
            node = super().generic_visit(node)
            for field, value in ast.iter_fields(node):
                if (isinstance(value, list) and value
                        and isinstance(value[0], ast.stmt)):
                    setattr(node, field,
                            [k for k in (self._fix(st) for st in value)
                             if k is not None])
            return node

    stripper = _StripUseNodes()
    tree = stripper.visit(tree)
    if stripper.n:
        changed = True
        notes.append(f"{stripper.n} use_nodes assignment(s) removed (node trees are "
                     f"always enabled in this build; the property is deprecated and warns "
                     f"at runtime)")

    # ---- T4: ocean modifier filmable floor -------------------------------
    # The small generation model copies the manual's example values
    # verbatim (resolution ~7, time 0, geometry_mode 'DISPLACE'), which
    # renders as a flat, nearly motionless quad. Enforce the filmable
    # floor deterministically:
    #   geometry_mode 'GENERATE'  ('DISPLACE' on a plain plane never waves)
    #   resolution >= 24          (the documented default is a preview grid)
    #   time > 0                  (at t=0 the wave field is exactly flat)
    ocean_fixes = []
    ocean_inserts = {}    # creation-stmt body index -> [stmts inserted after]
    for i, stmt in enumerate(tree.body):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call)):
            continue
        f = stmt.value.func
        if not (isinstance(f, ast.Attribute) and f.attr == "new"
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "modifiers"):
            continue
        if not any(isinstance(a, ast.Constant) and str(a.value) == "OCEAN"
                   for a in list(stmt.value.args)
                   + [k.value for k in stmt.value.keywords]):
            continue
        var = stmt.targets[0].id
        seen = {"geometry_mode": False, "time": False}
        for s2 in tree.body:
            if not (isinstance(s2, ast.Assign) and len(s2.targets) == 1):
                continue
            t = s2.targets[0]
            if not (isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == var):
                continue
            if t.attr in seen:
                seen[t.attr] = True
            if not isinstance(s2.value, ast.Constant):
                continue
            if t.attr == "geometry_mode" and s2.value.value != "GENERATE":
                s2.value.value = "GENERATE"
                ocean_fixes.append(f"{var}: geometry_mode -> 'GENERATE' "
                                   f"('DISPLACE' on a plain plane never waves)")
            elif t.attr == "time" and isinstance(s2.value.value, (int, float)) \
                    and not isinstance(s2.value.value, bool) \
                    and s2.value.value <= 0:
                s2.value.value = 5.0
                ocean_fixes.append(f"{var}: time set to 5.0 s "
                                   f"(t=0 renders a flat surface)")
            elif t.attr == "resolution" and isinstance(s2.value.value, (int, float)) \
                    and not isinstance(s2.value.value, bool) \
                    and s2.value.value < 24:
                s2.value.value = 24
                ocean_fixes.append(f"{var}: resolution raised to 24 "
                                   f"(the documented default renders a coarse grid)")
        insert = []
        if not seen["geometry_mode"]:
            insert.append(ast.parse(f"{var}.geometry_mode = 'GENERATE'").body[0])
            ocean_fixes.append(f"{var}: geometry_mode 'GENERATE' inserted")
        if not seen["time"]:
            insert.append(ast.parse(f"{var}.time = 5.0").body[0])
            ocean_fixes.append(f"{var}: time 5.0 s inserted "
                               f"(t=0 renders a flat surface)")
        if insert:
            ocean_inserts[i] = insert

    if ocean_fixes:
        new_body = []
        for i, stmt in enumerate(tree.body):
            new_body.append(stmt)
            new_body.extend(ocean_inserts.get(i, []))
        tree.body = new_body
        changed = True
        notes.extend(ocean_fixes)

    if not changed:
        return script, notes
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree), notes
    except Exception:
        return script, notes        # never break a script we cannot re-emit


# ============================================================
# HELPERS
# ============================================================
@app.get("/v1/models")
async def list_models():
    return {"object": "list",
            "data": [{"id": "qwen3.8:27b", "object": "model",
                     "created": 1686935002, "owned_by": "lemonade"}]}


def _looks_like_code(text: str) -> bool:
    """Does unfenced text actually look like a python script (vs. planning prose)?"""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    # a script starts with a code statement, not a sentence
    if not re.match(r'^\s*(import|from|bpy|obj|scene|mat|mesh|for|if|def|return|#)\b', lines[0]):
        return False
    long_lines = sum(1 for l in lines if len(l) > 120)
    prose_lines = sum(
        1 for l in lines
        if re.search(r'\b(?:the user|let me|here (?:is|are)|i will|we will|step \d|this will)\b', l, re.I)
    )
    return long_lines <= 1 and prose_lines <= 1


def extract_pure_script(text: str) -> str:
    """Pull the script out of a model response. STRICT:
    - </think> blocks are stripped first
    - a fenced block (```python or ```) wins
    - with NO fence, the text is only accepted if it provably looks like
      code; prose returns "" so the self-correction loop demands a fence."""
    if not text:
        return ""
    text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    m = re.search(r'```python(.*?)```', text_no_think, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```(.*?)```', text_no_think, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    stripped = text_no_think.strip()
    m = re.search(r'```(?:python)?\s*\n', text_no_think)
    if m:
        # opening fence with no closing fence -> truncated completion:
        # salvage the code up to the cut (Gate-1 will report any syntax gap)
        tail = text_no_think[m.end():].strip()
        if tail:
            return tail
    if stripped and _looks_like_code(stripped):
        return stripped
    return ""


def looks_like_error(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in (
        "traceback", "error", "exception", "failed",
        "field required", "not a valid tool", "invalid invocation",
    ))


def _break_response(model: str, content: str, rid: str) -> dict:
    return {
        "id": rid,
        "object": "chat.completion",
        "created": 1686935002,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
    }


def _history_tail(messages, tail=HISTORY_TAIL):
    """Bound what the local model sees on every request.

    Mixar's agent resends the ENTIRE conversation on every turn, so without
    this the local model re-prefills every earlier round on every request:
    round 2 carries round 1's messages, round 3 carries both, and so on —
    that growth is exactly what makes each round slower than the last,
    until Mixar's backend gives up with 'did not respond in time'.

    Generation only needs the current turn: the latest goal plus the newest
    tool results, which are always at the END of the list. Everything from
    completed earlier rounds is dropped here — effectively clearing the
    conversation cache after every finished inference — while leading system
    messages are kept (they are identical every round, so the model server's
    prompt cache still hits on them).
    """
    non_system = [m for m in messages if m.role != "system"]
    if len(non_system) <= tail:
        return list(messages)
    systems = [m for m in messages if m.role == "system"]
    return systems + non_system[-tail:]


def _llm_create(model_id, messages, timeout=LLM_TIMEOUT_S):
    """One OpenAI-compatible completion call against the local model.

    Every call goes through here so the hard per-call timeout
    (LLM_TIMEOUT_S) can never be bypassed, and so a stalled local model
    fails fast with a clear 504 instead of hanging until Mixar's own
    timeout fires ('OpenAI did not respond in time').
    """
    try:
        return llm_client.chat.completions.create(
            model=model_id, messages=_inject_thinking(messages),
            temperature=0.0, stream=False, timeout=timeout
        )
    except (openai.APITimeoutError, openai.APIConnectionError) as e:
        print(f"❌ Lemonade (port {LEMONADE_PORT}) — {e.__class__.__name__} after {timeout}s")
        raise HTTPException(
            status_code=504,
            detail=(f"Local model did not respond within {timeout:g}s. "
                    "Check that Lemonade / llama-server is running and not stuck on "
                    "another request, then try again."),
        ) from e


def _llm_chat(messages, model_id, label):
    """One Lemonade call with thinking control injected + no-script retry.
    Returns (content, reasoning)."""
    print(f"👉 {label} (model: {model_id}, thinking: {THINKING_MODE})...")
    resp = _llm_create(model_id, messages)
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = _reasoning_of(msg)
    if not extract_pure_script(content) and reasoning and extract_pure_script(reasoning):
        print("   🧠 (code was inside the reasoning field — salvaging it)")
        content, reasoning = reasoning, ""

    if not extract_pure_script(content):
        # model returned no usable code (prose, thinking overflow, or empty) —
        # ask once more, explicitly demanding the code this time
        print("   ⚠️ No script in response — retrying with an explicit demand...")
        retry_msgs = list(messages) + [
            {"role": "assistant", "content": content or "(no output)"},
            {"role": "user", "content":
                "Your last response contained NO SCRIPT — only planning text. Do not list "
                "properties, steps or a plan. Output ONLY the complete final script inside one "
                "```python fence, starting with `import bpy`. Code only."}
        ]
        resp2 = _llm_create(model_id, retry_msgs)
        msg2 = resp2.choices[0].message
        content2 = msg2.content or ""
        reasoning2 = _reasoning_of(msg2)
        if extract_pure_script(content2):
            return content2, reasoning
        if extract_pure_script(reasoning2):
            print("   🧠 (salvaged from reasoning on retry)")
            return reasoning2, ""
        return content2, reasoning2
    return content, reasoning


def _is_nav_junk(text: str) -> bool:
    """Detect manual-page navigation/TOC chunks (sidebar link lists) that poison
    search results with a wall of titles and zero property information."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False
    if any(ch.isdigit() for l in lines for ch in l):
        return False
    title_like = sum(1 for l in lines
                     if 2 < len(l) <= 48 and re.fullmatch(r"[A-Za-z][A-Za-z0-9 ()'\-]*", l))
    return (title_like / len(lines)) >= 0.8


def _rag_howto_lookup(query: str):
    """Look up HOW-TO documentation for the requested task in blender_rag_db.
    Two complementary queries (the task as-is + a 'how to ... in bpy' phrasing),
    merged, de-duplicated, distance-sorted. Returns chunk text or ''."""
    q = " ".join((query or "").split())
    queries = [q] if q else []
    core = re.sub(
        r'^(please|can you|could you|hi|hey|help me|show me|i want to|i want|i need to|i need|make|create|do)\s+',
        "", q.lower()).strip()
    if len(core) > 8 and core not in queries:
        queries.append(f"how to {core} in blender using python bpy script example")
    if not queries:
        return ""

    seen, chunks = set(), []
    for qq in queries:
        try:
            res = collection.query(query_texts=[qq], n_results=8)
            docs = (res.get("documents") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            for d, dist in zip(docs, dists):
                if not d or not str(d).strip():
                    continue
                d = str(d).strip()
                if d[:120] in seen:
                    continue
                seen.add(d[:120])
                if _is_nav_junk(d):
                    print(f"   └─ (distance {dist:.3f}) SKIPPED nav/TOC junk chunk")
                    continue
                chunks.append((dist, d))
        except Exception as e:
            print(f"⚠️ RAG lookup failed for {qq!r}: {e}")

    if not chunks:
        print(f"📭 RAG DOC LOOKUP — no manual chunks found for: {q[:80]}")
        return ""
    chunks.sort(key=lambda c: c[0])
    chunks = chunks[:6]
    print(f"📖 RAG DOC LOOKUP — {len(chunks)} manual chunk(s) for: {q[:80]}")
    for dist, d in chunks:
        print(f"   └─ (distance {dist:.3f}) {d[:160]!r}")
    return "\n\n".join(d for _, d in chunks)


def _rag_attribute_context(issues, max_chars=6000):
    """Manual + truth-table context for a list of validator issues.

    For every flagged class this pulls:
      1. the AUTHORITATIVE 'valid properties' truth doc (fetched by exact id,
         so it never misses), and
      2. the closest manual chunks for the specific flagged attributes.
    Returns '' when nothing could be retrieved."""
    if not issues:
        return ""
    cls_re = re.compile(r"bpy\.types\.(\w+)")
    attr_re = re.compile(r"bpy\.types\.(\w+) has no attribute '([^']+)'")
    classes, seen_cls = [], set()
    pairs, seen_pairs = [], set()
    for issue in issues:
        m = cls_re.search(issue)
        if m and m.group(1).lower() not in seen_cls:
            seen_cls.add(m.group(1).lower())
            classes.append(m.group(1))
        m2 = attr_re.search(issue)
        if m2 and (m2.group(1), m2.group(2)) not in seen_pairs:
            seen_pairs.add((m2.group(1), m2.group(2)))
            pairs.append((m2.group(1), m2.group(2)))

    parts = []
    # 1. Authoritative valid-property lists (exact-id fetch, no vector miss)
    truth_hits = []
    for cls in classes[:4]:
        try:
            got = collection.get(ids=[f"truth:{cls.lower()}"], include=["documents"])
            docs = got.get("documents") or []
            if docs and isinstance(docs[0], list):
                docs = docs[0]
            for d in docs:
                if d and str(d).strip():
                    truth_hits.append(str(d).strip())
                    break
        except Exception as e:
            print(f"⚠️ truth-doc fetch failed for {cls}: {e}")
    if truth_hits:
        parts.append("--- AUTHORITATIVE VALID-PROPERTY LISTS FOR THIS MIXAR BUILD (use ONLY names from these lists) ---\n"
                     + "\n\n".join(truth_hits[:3]))

    # 2. Manual chunks for the specific flagged attributes
    manual_chunks, manual_seen = [], set()
    for cls, attr in pairs[:3]:
        q = f"how to set {attr} on {cls} in blender manual python example"
        try:
            res = collection.query(query_texts=[q], n_results=3)
            for d in (res.get("documents") or [[]])[0]:
                d = (d or "").strip()
                if not d or d[:120] in manual_seen:
                    continue
                manual_seen.add(d[:120])
                manual_chunks.append(d)
        except Exception as e:
            print(f"⚠️ RAG attribute lookup failed for {cls}.{attr}: {e}")
    if manual_chunks:
        parts.append("--- MANUAL CHUNKS LOOKED UP FOR THE FLAGGED ATTRIBUTES ---\n"
                     + "\n\n".join(manual_chunks[:4]))

    ctx = "\n\n".join(parts)
    if not ctx:
        print("📭 RAG ATTRIBUTE CONTEXT — nothing retrieved (has --ingest-truth been run?).")
        return ""
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + " ...(truncated)"
    print(f"📖 RAG ATTRIBUTE CONTEXT — {len(ctx)} chars for {len(classes)} class(es), {len(pairs)} attribute(s)")
    return ctx


# ============================================================
# ONLINE CROSS-CHECK — when the LOCAL truth table cannot resolve
# a flagged reference (unknown class / missing attribute / read-only
# write), fetch the official Blender API page for that class from
# docs.blender.org (disk-cached in docs_cache/) so the fix round is
# answered from an authoritative online source instead of the model's
# memory. Best-effort: offline or unknown pages degrade to "no context".
# ============================================================
_DOCS_CACHE_DIR = os.path.join(SCRIPT_DIR, "docs_cache")
_ONLINE_FETCHED = set()


def _fetch_online_api_ref(cls_name, max_chars=2500):
    url = f"https://docs.blender.org/api/current/bpy.types.{cls_name}.html"
    path = os.path.join(_DOCS_CACHE_DIR, cls_name + ".html")
    raw = None
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                raw = f.read(400000)
        except OSError:
            raw = None
    if raw is None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MixarRAG/1.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                raw = r.read(400000)
            try:
                os.makedirs(_DOCS_CACHE_DIR, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(raw)
            except OSError:
                pass
        except Exception as e:
            print(f"ℹ️ Online API page unavailable for {cls_name} ({e.__class__.__name__})")
            return None
    if not raw:
        return None
    txt = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw.decode("utf-8", "ignore"))
    txt = re.sub(r"(?s)<[^>]+>", "\n", txt)
    txt = html.unescape(txt)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    out, started = [], False
    for l in lines:
        if re.match(r"^(Attributes|Properties|Methods)$", l):
            started = True
        if started:
            out.append(l)
        if len(out) > 150:
            break
    excerpt = "\n".join(out).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + " ..."
    return excerpt if len(excerpt) > 80 else None


def _online_crosscheck(issues, limit=3):
    """For each flagged reference the LOCAL truth table could not resolve,
    pull the official Blender API page for that class. Returns text or ''."""
    cls_re = re.compile(r"bpy\.types\.(\w+)")
    out = []
    for issue in issues[:15]:
        m = cls_re.search(issue)
        if not m:
            continue
        cls = m.group(1)
        if not any(k in issue for k in
                   ("is not a class", "has no attribute", "READ-ONLY", "NOT a writable")):
            continue          # resolved locally -> no online source needed
        if cls.lower() in _ONLINE_FETCHED:
            continue
        _ONLINE_FETCHED.add(cls.lower())
        txt = _fetch_online_api_ref(cls)
        if txt:
            print(f"🌐 ONLINE CROSS-CHECK: official API page for bpy.types.{cls} "
                  f"({len(txt)} chars, cached in docs_cache/)")
            out.append(f"### bpy.types.{cls}\n{txt}")
        if len(out) >= limit:
            break
    return "\n\n".join(out)


# ============================================================
# TRUTH-TABLE INGESTION — feed the validated attribute lists into
# the RAG collection so every retrieval path sees them.
# Run once after the v2 truth table exists:
#     python mixar-rag-server.py --ingest-truth
# ============================================================
def ingest_truth_table():
    _load_property_reference()
    if not BPy_ATTRS:
        print("❌ No truth table available.")
        print("   Start the server once (it dumps the table from the Mixar binary), "
              "then re-run:  python mixar-rag-server.py --ingest-truth")
        return
    ids, docs = [], []
    for key in sorted(BPy_ATTRS.keys()):
        name = BPy_NAMES.get(key, key)
        ids.append(f"truth:{key}")
        docs.append(f"bpy.types.{name} {_TRUTH_SEP} " + ", ".join(BPy_ATTRS[key]))
    for i in range(0, len(ids), 500):  # Chroma batch limit
        collection.upsert(ids=ids[i:i + 500], documents=docs[i:i + 500])
    print(f"✅ Ingested {len(ids)} truth-table documents into 'blender_api' "
          f"(collection now holds {collection.count()} documents).")


# ============================================================
# DEBUG ENDPOINTS
#   RAG probe:  http://127.0.0.1:8000/rag-test?q=YOUR+QUERY
#   Truth:      http://127.0.0.1:8000/truth?cls=Material
# ============================================================
@app.get("/rag-test")
async def rag_test(q: str = "create a material"):
    results = collection.query(query_texts=[q], n_results=3)
    docs = (results.get("documents") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    return {
        "query": q,
        "collection_count": collection.count(),
        "results": [
            {"distance": round(d, 4), "document": str(doc)[:500]}
            for d, doc in zip(dists, docs)
        ],
    }


@app.get("/truth")
async def truth_probe(cls: str = ""):
    _load_property_reference()
    key = cls.lower()
    if key:
        attrs = BPy_ATTRS.get(key)
        if attrs is None:
            close = get_close_matches(key, list(BPy_NAMES.keys()), n=5, cutoff=0.4)
            return {"class": cls, "found": False,
                    "hint": f"Closest classes: {', '.join(BPy_NAMES[c] for c in close)}" if close else None,
                    "total_classes": len(BPy_ATTRS)}
        return {"class": BPy_NAMES.get(key, key), "found": True,
                "count": len(attrs), "attributes": attrs,
                "types": BPy_TYPES.get(key, {}),
                "sockets": BPy_SOCKETS.get(key)}
    return {"total_classes": len(BPy_ATTRS), "sample": sorted(BPy_ATTRS.keys())[:30]}


# ============================================================
# TOOL-RESULT DETECTION + TOOL-NAME RESOLUTION
# ============================================================
def _is_tool_result(msg):
    if msg.role in ("tool", "function"):
        return True
    # some clients echo tool output back as a named assistant message
    if msg.role == "assistant" and msg.name and msg.name not in ("assistant", "user", "system"):
        return True
    return False


_EXEC_NAME_HINTS = ("execute", "run_", "python", "bpy", "script", "eval", "console", "code")
_EXEC_PARAM_HINTS = ("code", "script", "python_code", "bpy_code", "source", "snippet", "command")
_CODE_PARAM_KEYS = ("code", "script", "python_code", "bpy_code", "source", "snippet")


def _pick_param(params):
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
    str_keys = [k for k, v in params.items() if isinstance(v, dict) and v.get("type") == "string"]
    if len(str_keys) == 1:
        return str_keys[0]
    return "code"


def _pick_exec_tool(tools):
    """Derive (tool_name, param_name) from Mixar's declared schema.
    Returns (None, None) when declared tools exist but NONE of them can
    execute a script (e.g. only search_documentation)."""
    if not tools:
        print("⚠️ No tools declared by Mixar — using legacy execute_bpy_script/code.")
        return "execute_bpy_script", "code"
    names = []
    # pass 1: execution-looking name
    for t in tools:
        fn = (t or {}).get("function") or {}
        nm = fn.get("name")
        if not nm:
            continue
        names.append(nm)
        low = nm.lower()
        if any(h in low for h in _EXEC_NAME_HINTS) and "search" not in low:
            param = _pick_param((fn.get("parameters") or {}).get("properties") or {})
            print(f"✅ Using declared execution tool: '{nm}' (param: '{param}')")
            return nm, param
    # pass 2: a parameter that ACTUALLY looks like it carries code
    for t in tools:
        fn = (t or {}).get("function") or {}
        nm = fn.get("name")
        if not nm:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        code_keys = [k for k in props if str(k).lower() in _CODE_PARAM_KEYS]
        if code_keys:
            print(f"⚠️ No execution-named tool in {names}; using '{nm}' (param: '{code_keys[0]}').")
            return nm, code_keys[0]
    print(f"🔀 Declared tools {names} contain no script-execution tool — "
          "doc-phase skip (no tool payload will be forged).")
    return None, None


def _doc_phase_bypass(request_model: str, messages) -> dict:
    """LEGACY (SKIP_DOC_PHASE = False): turns where Mixar declared no execution
    tool are handed to the model as plain text — no gates, no checks, no tool
    payload, no goal-state pollution."""
    print("🔀 DOC-PHASE BYPASS — no script-execution tool declared this turn. "
          "Passing the turn to the model as plain text (no checks, no tool call).")
    content = ""
    try:
        plain = [{"role": m.role, "content": m.content or ""}
                 for m in _history_tail(messages)
                 if m.role in ("system", "user", "assistant") and (m.content or "")]
        if plain:
            resp = _llm_create(request_model, plain)
            content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠️ Doc-phase pass-through call failed: {e}")
    if not content:
        content = ("Understood — preparing the Blender work. The script will be "
                   "executed via the script-execution tool on the next step.")
    print("🔀 DOC-PHASE BYPASS response (plain text, no tool call):")
    print(content[:500])
    return {
        "id": "chatcmpl-doc-phase",
        "object": "chat.completion",
        "created": 1686935002,
        "model": request_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
    }


# ============================================================
# MAIN ENDPOINT
# ============================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global _tools_logged, _error_attempts
    try:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=false is required by this bridge")

        _last_in = request.messages[-1]
        _tool_names = [(t.get("function") or {}).get("name") for t in (request.tools or [])]
        print(f"📨 INCOMING — {len(request.messages)} message(s), last role: {_last_in.role}"
              + (f" (name={_last_in.name})" if _last_in.name else "")
              + f", tools: {_tool_names or 'none'}")

        if request.messages and request.messages[-1].role == "user":
            _error_attempts = 0

        if not _tools_logged and request.tools:
            _tools_logged = True
            print("🔎 MIXAR TOOL SCHEMA (declared tools):")
            print(json.dumps(request.tools, indent=2))

        # ========================================================
        # LOOP BREAKER — READ WHAT BLENDER ACTUALLY REPORTED
        # ========================================================
        if request.messages and _is_tool_result(request.messages[-1]):
            tool_result = request.messages[-1].content or ""
            print("📩 BLENDER TOOL RESULT RECEIVED:")
            print(tool_result[:3000])
            print("-" * 60)

            if looks_like_error(tool_result):
                hint = ""
                if "timed out" in (tool_result or "").lower():
                    hint += (" This run EXCEEDED THE HARD 60-SECOND EXECUTION BUDGET. WARNING: the timed-out "
                              "script may have PARTIALLY EXECUTED - heavy leftovers are likely in the scene "
                              "and will make your next attempt time out too. FIRST send a small cleanup-only "
                              "script that removes the objects/meshes your last script created (find them by "
                              "name and bpy.data.objects.remove each, then purge orphaned bpy.data meshes). "
                              "THEN rebuild much lighter across MULTIPLE small calls: no render calls, far "
                              "fewer and lower-poly objects (aim for a lean representative version - grid "
                              "resolutions under ~150, particle/instance counts under ~300), lower texture "
                              "iteration counts. Every call must finish well under 60 seconds.")
                if "is read-only" in (tool_result or "").lower():
                    hint += (" That attribute is READ-ONLY in this Mixar build — do not try another value "
                             "or a different route to the same state. DELETE that assignment line entirely, "
                             "keep every other line of the script identical, and resubmit.")
                if "builtin_function_or_method" in (tool_result or "") or \
                        "does not support item assignment" in (tool_result or ""):
                    hint += (" That line is subscripting a method or a scalar property — only "
                             "collections (modifiers, constraints, nodes, inputs, outputs, "
                             "vertex_groups, material_slots, bpy.data.* collections) and numeric "
                             "arrays accept ['name'] / [i] indexing. Remove the [ ... ] brackets and "
                             "set the property directly (e.g. obj.prop = value), keeping every other "
                             "line of the script identical.")
                if "bpy_prop_collection: attribute" in (tool_result or ""):
                    hint += (" That bpy.data collection does not have that method in the Mixar build - "
                              "stock-Blender data-creation APIs are not all available here. IMPORTANT: there "
                              "are NO sound data-blocks in this build (bpy.data.sounds has no .new, no Sound "
                              "objects, no audio emitters to attach). Audio is configured at SCENE level only, "
                              "via the writable scene properties: use_audio, audio_volume, "
                              "audio_distance_model, audio_doppler_factor, audio_doppler_speed and "
                              "use_audio_scrub. DELETE the sound-data creation lines entirely, set the scene "
                              "audio properties instead, and keep every other line identical.")
                if "required parameter" in (tool_result or ""):
                    hint += (" The .new() call is missing an argument this build requires (the error names "
                              "it). In this build several data-block creators need their full argument list, "
                              "e.g. bpy.data.lights.new(name, type), bpy.data.curves.new(name, type), "
                              "bpy.data.images.new(name, width, height), bpy.data.node_groups.new(name, type). "
                              "Add the required argument(s) and keep every other line identical.")
                if _error_attempts < MAX_ATTEMPTS:
                    _error_attempts += 1
                    print(f"⚠️ Blender reported an error (attempt {_error_attempts}/{MAX_ATTEMPTS}). Relaying the real error.")
                    return _break_response(
                        request.model,
                        f"Script execution failed in Blender (attempt {_error_attempts}/{MAX_ATTEMPTS}). "
                        f"Blender reported: {tool_result[:1500]}. "
                        f"Please fix the error and generate a corrected script." + hint,
                        "chatcmpl-error-relay"
                    )
                _error_attempts = 0
                print("🛑 Error limit reached. Stopping the retry chain.")
                return _break_response(
                    request.model,
                    f"Blender script failed repeatedly. Last error: {tool_result[:800]}",
                    "chatcmpl-error-limit"
                )

            _error_attempts = 0
            print("🔄 Blender reported success. Breaking the execution loop.")
            return _break_response(
                request.model,
                "The blender script execution pass has run successfully.",
                "chatcmpl-loop-break"
            )

        # ========================================================
        # GENERATION PATH
        # ========================================================
        user_query = ""
        for message in reversed(request.messages):
            if message.role == "user":
                user_query = message.content or ""
                break

        if not user_query:
            raise HTTPException(status_code=400, detail="No user query detected.")

        # ---------- DOC PHASE (search_documentation & co.) ----------
        # A turn that declares no execution tool is answered immediately —
        # no LLM call, no state pollution (SKIP_DOC_PHASE), or passed
        # through as plain text (legacy). The turn that DECLARES the
        # execution tool carries the fully validated script.
        exec_tool, exec_param = _pick_exec_tool(request.tools)
        if exec_tool is None:
            if SKIP_DOC_PHASE:
                print("🔀 DOC PHASE SKIPPED — search_documentation-only turn, no LLM call.")
                return _break_response(
                    request.model,
                    "Documentation lookup skipped. Proceed directly to the script-execution step.",
                    "chatcmpl-doc-phase-skip"
                )
            return _doc_phase_bypass(request.model, request.messages)

        # ---------- GOAL LOOP BREAKER ----------
        if len(_goal_state) > 500:
            _goal_state.clear()
        goal_key = re.sub(r"\s+", " ", user_query.lower()).strip()[:200]
        g_state = _goal_state.setdefault(goal_key, {"dispatches": 0, "scripts": set()})

        if g_state["dispatches"] >= MAX_DISPATCH_PER_GOAL:
            print(f"🛑 GOAL LOOP BREAKER — '{user_query[:60]}' already dispatched "
                  f"{g_state['dispatches']}× — stopping the agent chain.")
            return _break_response(
                request.model,
                f"I have already run {g_state['dispatches']} Blender passes for "
                f"'{user_query.strip()[:80]}'. Stopping further automatic iterations on this goal. "
                "Tell me a specific change if you want another pass.",
                "chatcmpl-goal-loop"
            )

        # Spin detector: this goal was already dispatched at least once, and the
        # incoming history contains NO tool result anywhere.
        has_tool_in_history = any(_is_tool_result(m) for m in request.messages)
        if g_state["dispatches"] >= 1 and not has_tool_in_history:
            print("🛑 GOAL LOOP BREAKER — goal already dispatched, no tool result in history "
                  "(agent spin). Stopping without regenerating.")
            return _break_response(
                request.model,
                "This goal was already sent to Blender and no execution result has been reported "
                "since. I will not re-run the same pass in a loop. If the scene did not change, "
                "check that a script-execution tool is enabled for this agent, or give me a "
                "specific new instruction.",
                "chatcmpl-goal-spin"
            )

        # ---------- RAG DOCUMENTATION LOOKUP (blender_rag_db) ----------
        # Look up the manual for HOW the requested task is done, so the
        # model implements it the documented way.
        retrieved_context = _rag_howto_lookup(user_query)

        rag_system_prompt = (
            "You are Mixar's internal automated code compiler. Output raw python code strings for Blender.\n"
            "CRITICAL FORMATTING BOUNDARY:\n"
            "- Do not write pleasantries, explanations, or dialogue.\n"
            "- Never output numbered plans, bullet lists, or property inventories. Code only.\n"
            "- Your response must immediately start with ```python and end with ```.\n"
            "- Do NOT write thinking blocks or reasoning text outside the code fence.\n"
            "- Only output valid Blender bpy Python code.\n\n"
            "BLENDER SESSION SAFETY RULES (a hard guard enforces these — violating them blocks execution):\n"
            "1. bpy.data is ONE namespace shared by every scene in this Blender session. Other agents may be working in sibling scenes.\n"
            "2. NEVER delete objects with an unscoped loop. No 'for obj in list(bpy.data.objects)' or "
            "'for obj in bpy.context.scene.collection.all_objects' without a positive name guard.\n"
            "3. To remove things you just created, delete BY EXACT NAME:\n"
            "     for n in ['MyObj_1', 'MyObj_2']:\n"
            "         o = bpy.context.scene.objects.get(n)\n"
            "         if o: bpy.data.objects.remove(o, do_unlink=True)\n"
            "4. Give every object you create a unique prefixed name (e.g. 'Chair_') and only ever target your own prefix:\n"
            "     for obj in list(bpy.context.scene.collection.all_objects):\n"
            "         if obj.name.startswith('Chair_'):\n"
            "             bpy.data.objects.remove(obj, do_unlink=True)\n"
            "5. Only if the user EXPLICITLY asked to clear the scene, use this guarded pattern:\n"
            "     for obj in list(bpy.context.scene.collection.all_objects):\n"
            "         if obj.type in {'CAMERA', 'LIGHT'} or len(obj.users_scene) > 1:\n"
            "             continue\n"
            "         bpy.data.objects.remove(obj, do_unlink=True)\n"
            "6. NEVER invent bpy attribute names from UI labels — UI section titles are not attribute names. "
            "A reference line listing 'valid properties' is AUTHORITATIVE for this build: only use "
            "attributes from that list. If you are not certain an attribute exists, omit that setting "
            "rather than guess.\n"
            "7. ATTRIBUTE VALUE TYPES: integer attributes must be assigned integer literals (never floats — "
            "use the stated minimum/maximum range, do not wrap a float in int() blindly); vector/array "
            "attributes must be assigned tuples like (x, y, z); booleans must be True/False; enums must be "
            "strings.\n"
            "8. NODE SOCKETS: when documentation lists a node type's sockets, use those exact socket names "
            "for node.inputs['...'] / node.outputs['...']. Never invent socket names. Some nodes expose SEVERAL sockets with the SAME name (Mix, Math, VectorMath, MapRange, AddShader): there are no distinct keys - node.inputs['A'] returns the FIRST socket named 'A'; reach the other variants of the same name by integer index (e.g. node.inputs[3]).\n"
            "9. MANUAL DOCUMENTATION: the final section below was looked up in the Blender manual "
            "specifically for THIS request. If it shows how the requested task is done (recipe, "
            "property sequence, operator, node setup), implement it exactly that way — the "
            "documentation overrides your own assumptions about how to do it.\n"
            "10. EXECUTION TIME BUDGET: the script runs in the live Mixar editor under a HARD 60-SECOND "
            "TIMEOUT — exceeding it kills the run with 'Request timed out after 60.0s'. Budget every "
            "second of it:\n"
            "    - NEVER call bpy.ops.render.render() or any render operator inside the script — "
            "previewing is handled separately by the viewport; set up the camera and lighting and stop.\n"
            "    - Keep the object count lean (roughly <= 100 objects). For repeated elements use "
            "linked-duplicate instances instead of hundreds of separate objects.\n"
            "    - Use low-resolution geometry: keep mesh subdivisions and displace/noise texture "
            "iterations low (e.g. a 128-segment grid max, iteration counts in single digits).\n"
            "    - Keep particle system counts modest (<= 500) and never start a simulation or "
            "playback frames.\n"
            "    - Call bpy.context.view_layer.update() once at the end, never inside loops.\n"
            "    - If the goal implies a huge scene, build a representative lean version that "
            "captures the look (e.g. 40 cloud puffs instead of 400).\n"
            "11. READ-ONLY PROPERTIES: some properties exist but are READ-ONLY in this Mixar build — "
            "the editor itself owns them. Example: scene.display.shading (viewport shading is driven by "
            "Mixar, not by your script). Never assign to such a property; if the desired effect is visual, "
            "set the writable data that drives it (materials, lights, world, camera) instead.\n12. EXECUTION TIME BUDGET: each script is killed after a HARD 60-SECOND LIMIT - size the work to finish well under it. If a previous script TIMED OUT it may have left partially-built objects behind: FIRST remove those leftovers in a small cleanup script, then continue with a lighter plan.\n13. NO SOUND DATA-BLOCKS in this build: never call bpy.data.sounds.new or create audio/sound objects or emitters - they do not exist here. Audio is scene-level only: scene.use_audio, scene.audio_volume, scene.audio_distance_model, scene.audio_doppler_factor, scene.audio_doppler_speed, scene.use_audio_scrub.\n\n"
            "14. ONE ACTION PER SCRIPT: each script does exactly ONE kind of work - either delete the leftovers of a failed run, OR create one small group of objects, OR apply materials or lighting to existing objects. NEVER combine cleanup + build + lighting + animation in a single script - finish this phase only, the next pass continues the rest.\n\n"
            "15. SUBSCRIPTING: only COLLECTIONS (modifiers, constraints, nodes, inputs, outputs, "
            "vertex_groups, material_slots, bpy.data.* collections) and NUMERIC ARRAYS (co[0], "
            "diffuse_color[0]) accept ['name'] / [i] indexing. Methods and scalar properties can "
            "NEVER be subscripted - assign them directly. The one exception: obj.pose['BoneName'].\n\n"
            "16. SOCKET SLOTS: never assign to a socket's .values / .vector / .color / .number - "
            "those are READ-ONLY aliases in this build and the engine rejects them ('NodeSocketColor' "
            "object attribute 'values' is read-only). Set the slot's writable property instead, and "
            "only when the socket's type lists it in the valid-property reference: "
            "node.outputs['Color'].default_value = (r, g, b, a), plus .hide / .enabled / .description. "
            "When iterating over node.inputs / node.outputs you do NOT know the slot's type - only "
            "set properties valid on EVERY socket type (e.g. .hide), or first index the exact socket "
            "by name and set its property there.\n\n"
            "17. FILMABLE COMPOSITION — the user judges the result in the viewport, so the scene "
            "must read as a shot, not a parts bin:\n"
            "    - Frame it: if the request implies a view, ensure the scene has a camera "
            "(create one and set scene.camera if missing) and point it AT the subject "
            "(rotation_euler or a TRACK_TO constraint on an empty target).\n"
            "    - Scale & distance: place the camera ~2-5x the subject's size away and keep the "
            "subject inside the frustum; every effect must be visible AT THAT DISTANCE "
            "(particle sizes >= ~0.3, no 0.1 m detail 100 m away).\n"
            "    - Materials: every visible surface gets an intentional Principled BSDF "
            "(Base Color + Roughness set) — never default grey. Build defensively: "
            "node = tree.nodes.get('Principled BSDF'); if None: tree.nodes.new("
            "'ShaderNodeBsdfPrincipled'); same pattern for the 'Material Output' node. A scene of "
            "untinted white planes is a FAILURE, not a success.\n"
            "    - Ocean modifier: use geometry_mode 'GENERATE' so the modifier builds its own "
            "wave grid; 'DISPLACE' on a plain single-quad plane stays a perfectly flat "
            "rectangle. Use resolution >= 24, spatial_size covering the visible area, and a "
            "time value > 0 so the waves are mid-motion.\n"
            "    - Layer depth: sky/world background + main subject + foreground accents, with a "
            "dim key light and a cool fill so nothing is pure black.\n\n"
            f"--- MANUAL DOCUMENTATION LOOKED UP FOR THIS TASK (blender_rag_db) ---\n"
            f"{retrieved_context if retrieved_context else '(no manual chunks matched this request — use only stable, well-known APIs and keep the script minimal)'}\n"
        )

        final_messages = [{"role": "system", "content": rag_system_prompt}]
        # Only the tail of the conversation reaches the local model: completed
        # earlier rounds are dropped (see _history_tail) so round 5 is no
        # slower than round 1.
        tail_msgs = _history_tail(request.messages)
        for msg in tail_msgs:
            if msg.role != "system":
                final_messages.append({"role": msg.role, "content": msg.content or ""})
        print(f"✂️ HISTORY TRIM — {len(request.messages)} incoming message(s), "
              f"{sum(1 for m in tail_msgs if m.role != 'system')} kept for the local model "
              f"(tail limit: {HISTORY_TAIL})")

        generated_text, _ = _llm_chat(final_messages, request.model, "GENERATION")
        pure_script, norm_notes = normalize_mixar_script(extract_pure_script(generated_text))
        for _nn in norm_notes:
            print(f"🔁 NORMALIZED: {_nn}")

        print("📄 GENERATED SCRIPT FOR BLENDER:")
        print(pure_script)
        print("=" * 60)

        # ---------- GATE 1: validate PYTHON SYNTAX ----------
        syntax_error = None
        if not pure_script.strip():
            syntax_error = "no code block found in the model reply"
        else:
            try:
                compile(pure_script, "<agent_script>", "exec")
            except SyntaxError as e:
                syntax_error = f"{e.msg} (line {e.lineno})"

        if syntax_error:
            print(f"⚠️ Generated script is invalid Python: {syntax_error}. Asking Lemonade to self-correct...")
            if "no code" in syntax_error:
                fix_note = (
                    "Your reply contained NO code at all — only planning text. "
                    "Do NOT write a plan, bullet list, or property inventory. "
                    "Output ONLY the complete script inside a single ```python fence, "
                    "starting with `import bpy`. Code only — zero prose.")
            else:
                fix_note = (
                    f"Your output was NOT valid Python (error: {syntax_error}). "
                    "Output the corrected script again. Start with ```python, end with ```. "
                    "Code only — no explanation.")
            retry_messages = final_messages + [
                {"role": "assistant", "content": generated_text or "(no output)"},
                {"role": "user", "content": fix_note}
            ]
            retry_text, _ = _llm_chat(retry_messages, request.model, "SYNTAX SELF-CORRECTION")
            pure_script = extract_pure_script(retry_text)
            print("📄 CORRECTED SCRIPT:")
            print(pure_script)
            print("=" * 60)
            try:
                compile(pure_script, "<agent_script>", "exec")
            except SyntaxError as e2:
                print(f"🛑 Self-correction failed: {e2.msg} (line {e2.lineno}).")
                return _break_response(
                    request.model,
                    f"Script generation failed — invalid Python even after retry: {e2.msg} (line {e2.lineno}). Please rephrase your request.",
                    "chatcmpl-syntax-retry-failed"
                )

        # ---------- GATE 2: validate EVERY attribute vs the Mixar v2 truth table ----------
        prop_issues = validate_bpy_properties(pure_script)
        for fix_round in range(MAX_FIX_ROUNDS):
            if not prop_issues:
                break
            print(f"🔍 ATTRIBUTE VALIDATION ISSUES (round {fix_round + 1}/{MAX_FIX_ROUNDS}):")
            for i in prop_issues:
                print("   ", i)
            print("   → Asking Lemonade to fix attribute names / value types...")
            report = "\n".join(prop_issues)
            fix_prompt = (
                "Attribute validation against the AUTHORITATIVE attribute table of this Mixar build "
                "flagged these problems:\n"
                f"{report}\n"
                "The lists above are AUTHORITATIVE and override anything in the API context or in "
                "your training data. Fix ONLY the flagged lines using the suggested names (if no "
                "valid attribute provides the same effect, simply drop that line). Value-type rules: "
                "if an attribute 'expects an INT', use an integer literal within the stated "
                "minimum/maximum — and PRESERVE THE AUTHOR'S INTENDED MAGNITUDE: if the context "
                "below lists a documented default for that property, prefer that default; otherwise "
                "pick a value near the original value's scale. Never emit the bare range minimum or "
                "maximum just because it is legal. If it 'expects a sequence', pass a tuple like "
                "(x, y, z); if it 'expects True/False', use True/False; if it 'expects an enum "
                "string', use one of the documented enum values; if a socket name is flagged, use "
                "one of the listed sockets. Keep everything else identical. Output the full "
                "corrected script: "
                "```python ... ``` only."
            )
            # Look the flagged classes up in the manual + truth table so each
            # fix round is answered by the documentation, not the model's memory.
            attr_context = _rag_attribute_context(prop_issues)
            if attr_context:
                fix_prompt += (
                    "\n\nThe sections below were looked up in the Mixar manual and the authoritative "
                    "attribute table specifically for the flagged classes. Use them to pick the "
                    "correct names:\n\n"
                    f"{attr_context}"
                )
            # Anything the local truth table could not resolve is cross-checked
            # against the official Blender API pages (docs.blender.org, cached).
            online_ctx = _online_crosscheck(prop_issues)
            if online_ctx:
                fix_prompt += (
                    "\n\nOFFICIAL BLENDER API PAGES (fetched from docs.blender.org because the local "
                    "truth table could not fully resolve the flagged references; they describe STOCK "
                    "Blender and may differ from this Mixar build — the authoritative attribute lists "
                    "above ALWAYS win on conflicts, but use the pages to find the canonical "
                    "replacement for a blocked attribute):\n\n" + online_ctx
                )
            fix_messages = final_messages + [
                {"role": "assistant", "content": "```python\n" + pure_script + "\n```"},
                {"role": "user", "content": fix_prompt}
            ]
            fix_text, _ = _llm_chat(fix_messages, request.model, "ATTRIBUTE FIX")
            fixed_script = extract_pure_script(fix_text)
            print("📄 ATTRIBUTE-FIXED SCRIPT:")
            print(fixed_script)
            print("=" * 60)
            if fixed_script.strip():
                try:
                    compile(fixed_script, "<agent_script>", "exec")
                    pure_script, _nn2 = normalize_mixar_script(fixed_script)
                    for _nn in _nn2:
                        print(f"🔁 NORMALIZED (post-fix): {_nn}")
                except SyntaxError:
                    print("⚠️ Attribute fix broke syntax — keeping previous script.")
            prop_issues = validate_bpy_properties(pure_script)

        if not prop_issues:
            print("🔍 ATTRIBUTE VALIDATION: every attribute access checks out against the Mixar truth table.")
        elif HARD_GATE:
            print("🛑 HARD GATE: unresolved attribute issues — NOT sending to Mixar.")
            return _break_response(
                request.model,
                "Script was NOT executed: the following bpy attributes do not exist in this Mixar build "
                f"and could not be auto-corrected:\n" + "\n".join(prop_issues) +
                "\n(If Mixar was recently updated, re-dump the truth table: "
                "python mixar-rag-server.py --refresh-truth then --ingest-truth.)",
                "chatcmpl-attribute-gate"
            )
        else:
            print(f"⚠️ {len(prop_issues)} attribute issue(s) remain — shipping anyway.")

        # -----------------------------------------------------------------
        if not pure_script.strip():
            print(f"⚠️ EMPTY SCRIPT — raw model output tail: {generated_text[-400:]!r}")
            return _break_response(
                request.model,
                "Script generation failed (model returned empty code). Please try again.",
                "chatcmpl-empty"
            )

        # ---------- duplicate-script guard (same goal, same code = spin) ----------
        script_sha = hashlib.sha1(pure_script.strip().encode("utf-8")).hexdigest()
        if g_state and script_sha in g_state["scripts"]:
            print("🛑 GOAL LOOP BREAKER — identical script already executed for this goal.")
            return _break_response(
                request.model,
                "The same Blender script was already executed for this goal. Stopping to avoid an "
                "infinite loop — ask for a specific change to proceed.",
                "chatcmpl-duplicate-script"
            )

        # ---------- dispatch ----------
        tool_arguments = json.dumps({exec_param: pure_script})

        if g_state:
            g_state["dispatches"] += 1
            g_state["scripts"].add(script_sha)

        print(f"⚡ Forcing single-pass Tool Payload down to Mixar via tool '{exec_tool}'...")
        return {
            "id": "chatcmpl-mixar-bridge",
            "object": "chat.completion",
            "created": 1686935002,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Running requested generation action directly inside the Blender viewport layout workspace...",
                    "tool_calls": [{
                        "id": "call_mixar_auto_run_01",
                        "type": "function",
                        "function": {
                            "name": exec_tool,
                            "arguments": tool_arguments
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ DETECTED ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bridge Error: {str(e)}")


# ============================================================
# ENTRY POINT
#   normal:     python mixar-rag-server.py
#   ingest:    python mixar-rag-server.py --ingest-truth
#   re-dump:   python mixar-rag-server.py --refresh-truth
# ============================================================
if __name__ == "__main__":
    args = set(sys.argv[1:])
    if "--refresh-truth" in args:
        for path in (REF_JSON, REF_TYPES_JSON, REF_SOCK_JSON, REF_META_JSON):
            try:
                os.remove(path)
                print(f"🗑️  Removed {os.path.basename(path)} — re-dumping v2 truth table from Mixar.")
            except OSError:
                pass
    _load_property_reference()   # load cache, or dump from Mixar once (visible at startup)
    if "--ingest-truth" in args:
        ingest_truth_table()
    uvicorn.run(app, host="127.0.0.1", port=8000)
