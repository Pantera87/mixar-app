import bpy, json

out = {}

c = getattr(bpy.types, "NodeSocketColor", None)
spec = {}
if c is not None:
    for n in ("values", "vector", "color", "default_value"):
        try:
            d = getattr(c, n, "__MISS__")
            if d == "__MISS__":
                spec[n] = "missing"
            elif isinstance(d, property):
                spec[n] = f"property fset={bool(d.fset)}"
            elif hasattr(d, "is_readonly"):
                spec[n] = f"BPyProp ro={d.is_readonly} type={d.type}"
            elif callable(d):
                spec[n] = f"callable {type(d).__name__}"
            else:
                spec[n] = type(d).__name__
        except Exception as e:
            spec[n] = f"err:{type(e).__name__}:{e}"
out["NodeSocketColor"] = spec


def ids_of(rna):
    try:
        s = set()
        for p in rna.properties:
            i = getattr(p, "identifier", None)
            if i is None:
                i = getattr(p, "name", None)
            if i:
                s.add(i)
        return s
    except Exception:
        return None


kinds = {}
wnon = []
prop_names = {}
bpynon = {}
other_names = {}
ncls = 0
for name in dir(bpy.types):
    if not name[:1].isupper():
        continue
    try:
        cls = getattr(bpy.types, name, None)
        if cls is None:
            continue
        rna = getattr(cls, "bl_rna", None)
        if rna is None:
            continue
        props = ids_of(rna)
        if props is None:
            continue
        ncls += 1
        for m in dir(cls):
            if m[:2] == "__" or m in props:
                continue
            try:
                d = getattr(cls, m)
            except Exception:
                kinds["err"] = kinds.get("err", 0) + 1
                continue
            if isinstance(d, property):
                kinds["property"] = kinds.get("property", 0) + 1
                if d.fset is not None:
                    wnon.append(f"{name}.{m}")
                if m[:1].islower():
                    prop_names[m] = prop_names.get(m, 0) + 1
            elif hasattr(d, "is_readonly"):
                kinds["BPyProp"] = kinds.get("BPyProp", 0) + 1
                bpynon[m] = bpynon.get(m, 0) + 1
            elif callable(d):
                kinds["method"] = kinds.get("method", 0) + 1
            else:
                k = "other:" + type(d).__name__
                kinds[k] = kinds.get(k, 0) + 1
                other_names[m] = other_names.get(m, 0) + 1
    except Exception:
        kinds["class_err"] = kinds.get("class_err", 0) + 1

out["classes"] = ncls
out["kinds"] = kinds
out["writable_nonrna"] = wnon
out["nonrna_property_names"] = prop_names
out["nonrna_bpyprops"] = bpynon
out["nonrna_other_names"] = other_names
print("__PROBE__" + json.dumps(out))
