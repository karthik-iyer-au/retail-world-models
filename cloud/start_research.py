"""Start the research loop on Modal so that it runs to the end on its own,
whatever happens to the computer that started it.

  python cloud/start_research.py configs/research/<name>.yaml

Deploys the app (`modal deploy cloud/modal_app.py`), fetches the data the spec
needs, and starts `research_loop` on the deployed app. A loop started from a
`modal run` session stops when that session's computer disconnects, even with
--detach, because the loop hands its tries to other functions. The log is
saved after every row to the results volume as research/<name>/log.jsonl.
"""
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))


def main(spec: str) -> None:
    import modal

    import modal_app
    from rwm.research import loop

    commit = modal_app._local_commit()
    s = loop.load_spec(spec)
    base = yaml.safe_load((modal_app.LOCAL_REPO / s["base"]).read_text())
    subprocess.run(["modal", "deploy", str(HERE / "modal_app.py")], check=True)
    modal.Function.from_name("retail-world-models", "fetch").remote(modal_app._dataset_job(base), commit)
    call = modal.Function.from_name("retail-world-models", "research_loop").spawn(spec, commit, 0)
    print(f"research loop started at commit {commit[:8]} ({call.object_id}); log at rwm-results research/{s['name']}/log.jsonl")


if __name__ == "__main__":
    main(sys.argv[1])
