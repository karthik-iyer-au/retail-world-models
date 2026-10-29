"""Start the research loop on Modal so that it runs to the end on its own,
whatever happens to the computer that started it.

  python cloud/start_research.py configs/research/<name>.yaml
  python cloud/start_research.py configs/research/<name>.yaml --start-only   # the starting settings only

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


def start_only(spec: str) -> None:
    """Run only the loop's starting settings, on every seed, under the names the
    loop gives them, so a loop started later reads them back and does not repeat
    them. For measuring the seed spread before committing to a whole loop."""
    import modal

    import modal_app
    from rwm.experiments import plan
    from rwm.research import loop

    commit = modal_app._local_commit()
    s = loop.load_spec(spec)
    base = yaml.safe_load((modal_app.LOCAL_REPO / s["base"]).read_text())
    subprocess.run(["modal", "deploy", str(HERE / "modal_app.py")], check=True)
    modal.Function.from_name("retail-world-models", "fetch").remote(modal_app._dataset_job(base), commit)
    for seed in s["seeds"]:
        c = loop.config_for(s, loop.base_params(s), seed, f"start_s{seed}")
        gb = plan.memory_gb(c)
        if plan.machine(c) == "gpu":
            kind = "try_gpu_xlarge" if gb > 32 else "try_gpu_large" if gb > 16 else "try_gpu"
        else:
            kind = "try_cpu_large" if gb > 16 else "try_cpu"
        call = modal.Function.from_name("retail-world-models", kind).spawn(c, commit, f"research/{s['name']}")
        print(f"{c['name']:<64} {kind:<15} {call.object_id}")


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
    if "--start-only" in sys.argv:
        start_only([a for a in sys.argv[1:] if a != "--start-only"][0])
    else:
        main(sys.argv[1])
