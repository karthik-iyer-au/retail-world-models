"""Start work on rented Vast.ai machines through the deployed `rwm-vast` app (cloud/vast_app.py).

  python cloud/start_vast.py configs/research/<spec>.yaml                 # a research loop
  python cloud/start_vast.py --config configs/experiments/<x>.yaml --out vast_test   # one run
  python cloud/start_vast.py --app rwm-vast-cpu configs/research/<spec>.yaml   # LightGBM, on processor machines
  python cloud/start_vast.py --app <app> --machines 6 configs/research/<spec>.yaml   # at most 6 machines for this loop

The commit is the local HEAD, which must be clean and on GitHub, as for Modal.
Deploy the app first (`modal deploy cloud/vast_app.py`); deploying it does not
touch the `retail-world-models` app or its loops. Data comes from the
`rwm-data` volume, so the data set must be on it.
"""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(argv: list[str]) -> None:
    import modal

    import modal_app  # only for the commit check; deploys nothing

    commit = modal_app._local_commit()
    app = argv[argv.index("--app") + 1] if "--app" in argv else "rwm-vast"
    limit = int(argv[argv.index("--machines") + 1]) if "--machines" in argv else 6
    argv = [a for i, a in enumerate(argv) if a not in ("--app", "--machines") and (i == 0 or argv[i - 1] not in ("--app", "--machines"))]
    if "--config" in argv:
        cfg_file = argv[argv.index("--config") + 1]
        out = argv[argv.index("--out") + 1]
        config = yaml.safe_load((modal_app.LOCAL_REPO / cfg_file).read_text())
        call = modal.Function.from_name(app, "vast_try").spawn(config, commit, out)
        print(f"{config['name']} started at commit {commit[:8]} ({call.object_id}); results in rwm-results/{out}")
        return
    spec = [a for a in argv if not a.startswith("--")][0]
    call = modal.Function.from_name(app, "vast_loop").spawn(spec, commit, 0, limit)
    name = yaml.safe_load((modal_app.LOCAL_REPO / spec).read_text())["name"]
    print(f"research loop on Vast started at commit {commit[:8]} ({call.object_id}); log at rwm-results research/{name}/log.jsonl")


if __name__ == "__main__":
    main(sys.argv[1:])
