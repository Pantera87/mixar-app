"""One-shot: stage install-ready copies of the message-gate feature.

Stages verbatim copies of the feature-branch files (the current installed
build HAS the i18n `n_` module, so no porting is needed).
Run: python rag/_build_deploy_stage.py
"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
repo = HERE.parent / "src" / "scripts" / "mixar" / "modules"
stage = HERE / "_deploy_stage"
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir()

# script_validator.py, message_gate.py, relay.py, constants.py — verbatim
shutil.copyfile(repo / "space_mixie_chat" / "core" / "script_validator.py",
                stage / "script_validator.py")
shutil.copyfile(repo / "local_models" / "core" / "message_gate.py",
                stage / "message_gate.py")
shutil.copyfile(repo / "local_models" / "core" / "relay.py",
                stage / "relay.py")
shutil.copyfile(repo / "local_models" / "constants.py",
                stage / "constants.py")

# bundled truth tables
truth = stage / "data" / "mixar_truth"
truth.mkdir(parents=True)
for f in (repo / "space_mixie_chat" / "data" / "mixar_truth").glob("*.json"):
    shutil.copyfile(f, truth / f.name)

print("STAGED:")
for p in sorted(stage.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(HERE), p.stat().st_size)

