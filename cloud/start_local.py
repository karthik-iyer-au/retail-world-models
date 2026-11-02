"""Queue runs for the NUC worker (cloud/local_app.py, cloud/nuc_worker.sh).

  python cloud/start_local.py --config configs/experiments/<x>.yaml --out <folder>
  python cloud/start_local.py status

The commit is the local HEAD, which must be clean and on GitHub, as for Modal.
"""
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(argv: list[str]) -> None:
    import modal

    from local_app import QUEUE, STATE

    queue = modal.Queue.from_name(QUEUE, create_if_missing=True)
    if argv and argv[0] == "status":
        state = modal.Dict.from_name(STATE, create_if_missing=True)
        print(f"waiting in the queue: {queue.len()}")
        for key, s in sorted(state.items(), key=lambda kv: kv[1]["at"]):
            when = time.strftime("%d %b %H:%M UTC", time.gmtime(s["at"]))
            extra = f" ({s['seconds'] / 3600:.2f} h)" if s.get("seconds") else ""
            print(f"{when}  {s['status']:<13} {key}{extra}")
        return
    import modal_app  # only for the commit check; deploys nothing

    commit = modal_app._local_commit()
    config = yaml.safe_load((modal_app.LOCAL_REPO / argv[argv.index("--config") + 1]).read_text())
    out = argv[argv.index("--out") + 1]
    queue.put({"config": config, "commit": commit, "out": out})
    print(f"{config['name']} queued for the NUC at commit {commit[:8]}; results in rwm-results/{out}")


if __name__ == "__main__":
    main(sys.argv[1:])
