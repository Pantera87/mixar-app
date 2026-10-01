# One-shot driver: run pytest inside the Mixar python.
# Usage (from the repo root):
#   mixar.exe -b -noaudio --factory-startup --python rag/_run_pytest.py [paths...]
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src", "scripts"))

import pytest

args = list(sys.argv[1:]) or [
    "src/scripts/mixar/modules/space_mixie_chat/tests"]
args = ["-q", "--no-header", "-p", "no:cacheprovider"] + args
raise SystemExit(pytest.main(args))
