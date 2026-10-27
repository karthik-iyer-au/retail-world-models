"""Start every run of a job file on the deployed Modal app, so the runs go on
to the end whatever happens to the computer that started them.

  python cloud/start_job.py configs/jobs/<job>.yaml <out folder on the results volume>

Deploys the app, fetches the job's data, and hands each run (each seed
separately) to the machine kind it needs (`try_gpu`, `try_cpu` and their large
forms). Runs already saved for exactly the same config are read back, not
repeated. Results land in the results volume under <out>.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))


def main(job_file: str, out: str) -> None:
    import modal

    import modal_app
    from rwm.experiments import plan

    commit = modal_app._local_commit()
    job = plan.load(job_file)
    runs = plan.expand(job)
    subprocess.run(["modal", "deploy", str(HERE / "modal_app.py")], check=True)
    modal.Function.from_name("retail-world-models", "fetch").remote({"datasets": job.get("datasets", [])}, commit)
    for r in runs:
        c = r["config"]
        kind = ("try_gpu" if plan.machine(c) == "gpu" else "try_cpu") + ("_large" if plan.memory_gb(c) > 16 else "")
        call = modal.Function.from_name("retail-world-models", kind).spawn(r["config"], commit, out)
        print(f"{r['config']['name']:<56} {kind:<15} {call.object_id}")
    print(f"{len(runs)} runs started at commit {commit[:8]}; results in rwm-results/{out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
