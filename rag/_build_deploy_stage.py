"""One-shot: stage install-ready copies of the message-gate feature.

The installed Mixar build (C:\\Program Files\\Mixar) predates the i18n `n_`
wrapper in local_models, so the staged relay.py / constants.py are the repo
versions with ONLY the i18n lines stripped — everything else (the gate) is
verbatim. Run: python rag/_build_deploy_stage.py
"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
repo = HERE.parent / "src" / "scripts" / "mixar" / "modules"
stage = HERE / "_deploy_stage"
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir()

# 1. script_validator.py — verbatim (no i18n in this file)
shutil.copyfile(repo / "space_mixie_chat" / "core" / "script_validator.py",
                stage / "script_validator.py")

# 2. message_gate.py — verbatim (no i18n)
shutil.copyfile(repo / "local_models" / "core" / "message_gate.py",
                stage / "message_gate.py")

# 3. relay.py — strip i18n import + unwrap the two pre-existing messages
s = (repo / "local_models" / "core" / "relay.py").read_text(encoding="utf-8")
s = s.replace("from mixar.modules.common.i18n import n_\n\n", "")
s = s.replace('n_("Enter a valid http(s) URL, e.g. http://127.0.0.1:11434")',
              '"Enter a valid http(s) URL, e.g. http://127.0.0.1:11434"')
s = s.replace('n_("Only servers on this computer or your local network are allowed")',
              '"Only servers on this computer or your local network are allowed"')
assert "n_(" not in s and "i18n" not in s, "relay still has i18n remnants"
(stage / "relay.py").write_text(s, encoding="utf-8")

# 4. constants.py — strip i18n import, unwrap the 5 descriptions
s = (repo / "local_models" / "constants.py").read_text(encoding="utf-8")
s = s.replace("from mixar.modules.common.i18n import n_\n\n", "")
descriptions = [
    "Smallest vision-capable model — fast on any machine, good tool use.",
    "Recommended default — balanced speed/quality, vision + very good tool use.",
    "High-quality vision model for 16 GB machines — excellent tool use.",
    "Text-only reasoning/agentic model (no image understanding).",
    "Agentic-coding flagship for 32 GB machines — vision + best tool use.",
]
for d in descriptions:
    s = s.replace(f'n_("{d}")', f'"{d}"')
assert "n_(" not in s and "i18n" not in s, "constants still has i18n remnants"
(stage / "constants.py").write_text(s, encoding="utf-8")

# 5. bundled truth tables
truth = stage / "data" / "mixar_truth"
truth.mkdir(parents=True)
for f in (repo / "space_mixie_chat" / "data" / "mixar_truth").glob("*.json"):
    shutil.copyfile(f, truth / f.name)

print("STAGED:")
for p in sorted(stage.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(HERE), p.stat().st_size)
