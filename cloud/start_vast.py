"""Start work on rented Vast.ai machines through the deployed `rwm-vast` app (cloud/vast_app.py).

  python cloud/start_vast.py configs/research/<spec>.yaml                 # a research loop
  python cloud/start_vast.py --config configs/experiments/<x>.yaml --out vast_test   # one run
  python cloud/start_vast.py --app <app> --machines 40 --split --out final --configs a.yaml b.yaml   # several runs
  python cloud/start_vast.py --app rwm-vast-cpu configs/research/<spec>.yaml   # LightGBM, on processor machines
  python cloud/start_vast.py --app <app> --machines 6 configs/research/<spec>.yaml   # at most 6 machines for this loop
  python cloud/start_vast.py --app <app> --rounds 1 configs/research/lossT_<name>.yaml  # one round only (study T)
  python cloud/start_vast.py --app <app> --machines 40 --split configs/research/<spec>.yaml  # each run split into one
      # piece per (window, copy) on its own machine, then a last run that scores (vast_app._manage)

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
    if "--commit" in argv:  # an earlier commit, so a split run restarts on the code its saves were made with
        import subprocess

        commit = subprocess.run(["git", "-C", str(modal_app.LOCAL_REPO), "rev-parse", argv[argv.index("--commit") + 1]],
                                capture_output=True, text=True, check=True).stdout.strip()
    app = argv[argv.index("--app") + 1] if "--app" in argv else "rwm-vast"
    limit = int(argv[argv.index("--machines") + 1]) if "--machines" in argv else 6
    rounds = int(argv[argv.index("--rounds") + 1]) if "--rounds" in argv else 0  # 0: no limit
    split = "--split" in argv
    opts = ("--app", "--machines", "--rounds", "--commit")
    argv = [a for i, a in enumerate(argv) if a != "--split" and a not in opts and (i == 0 or argv[i - 1] not in opts)]
    if "--configs" in argv:  # several runs at once: --configs a.yaml b.yaml ... --out <folder>
        files = [a for a in argv[argv.index("--configs") + 1:] if a.endswith(".yaml")]
        out = argv[argv.index("--out") + 1]
        configs = [yaml.safe_load((modal_app.LOCAL_REPO / f).read_text()) for f in files]
        call = modal.Function.from_name(app, "vast_batch").spawn(configs, commit, out, limit, split)
        print(f"{len(configs)} runs started at commit {commit[:8]} ({call.object_id}); results in rwm-results/{out}")
        return
    if "--config" in argv:
        cfg_file = argv[argv.index("--config") + 1]
        out = argv[argv.index("--out") + 1]
        config = yaml.safe_load((modal_app.LOCAL_REPO / cfg_file).read_text())
        call = modal.Function.from_name(app, "vast_try").spawn(config, commit, out)
        print(f"{config['name']} started at commit {commit[:8]} ({call.object_id}); results in rwm-results/{out}")
        return
    spec = [a for a in argv if not a.startswith("--")][0]
    call = modal.Function.from_name(app, "vast_loop").spawn(spec, commit, rounds, limit, split)
    name = yaml.safe_load((modal_app.LOCAL_REPO / spec).read_text())["name"]
    print(f"research loop on Vast started at commit {commit[:8]} ({call.object_id}); log at rwm-results research/{name}/log.jsonl")


if __name__ == "__main__":
    main(sys.argv[1:])
