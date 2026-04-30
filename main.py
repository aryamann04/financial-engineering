from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def _bootstrap_project_venv() -> None:
    if os.environ.get("FE_SKIP_VENV_BOOTSTRAP") == "1":
        return

    repo_root = Path(__file__).resolve().parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    current = Path(sys.executable)
    if current == venv_python:
        return

    env = os.environ.copy()
    env["FE_SKIP_VENV_BOOTSTRAP"] = "1"
    os.execve(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]], env)


def main() -> None:
    from analyzer.unified import run_unified_analyzer

    run_unified_analyzer()


if __name__ == "__main__":
    _bootstrap_project_venv()
    main()
