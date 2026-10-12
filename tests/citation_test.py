import os
import subprocess
import sys
from pathlib import Path


def _run_python(code: str, env: dict | None = None) -> str:
    run_env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    existing_pythonpath = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
    if env:
        run_env.update(env)
    out = subprocess.check_output([sys.executable, "-c", code], env=run_env)
    return out.decode("utf-8", errors="replace")


def test_lt_citation_function_available():
    import linktransformer as lt

    cite = lt.citation()
    assert isinstance(cite, str)
    assert "doi.org/10.18653/v1/2024.acl-demos.21" in cite


def test_lt_import_prints_citation_once_per_process():
    out = _run_python(
        "import linktransformer\nimport linktransformer\n",
        env={"LINKTRANSFORMER_SILENCE_CITATION": ""},
    )
    assert out.count("doi.org/10.18653/v1/2024.acl-demos.21") == 1


def test_lt_import_citation_can_be_silenced():
    out = _run_python(
        "import linktransformer\n",
        env={"LINKTRANSFORMER_SILENCE_CITATION": "1"},
    )
    assert "doi.org/10.18653/v1/2024.acl-demos.21" not in out
