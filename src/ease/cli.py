"""Command line.

    ease info      --model release/edge
    ease demo      --model release/edge --aggregator release/aggregator
    ease serve     --model release/edge --aggregator release/aggregator --data ~/.ease          # one person, no keys
    ease serve     --model release/edge --aggregator release/aggregator --data /srv/ease --multi # workspaces with keys
    ease workspace create --data /srv/ease --id acme --name "Acme agency"                        # prints the key once
    ease workspace list   --data /srv/ease
    ease workspace rotate-key --data /srv/ease --id acme
    ease backup    --data /srv/ease --out /srv/ease-backups
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _settings(model_dir: str) -> dict:
    p = Path(model_dir).parent / "settings.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _runtime(a, data_dir):
    from ease.aggregate import load_aggregator
    from ease.runtime import Runtime
    from ease.scorer import ModelScorer

    scorer = ModelScorer(a.model, canonical=not a.fast)
    agg = load_aggregator(a.aggregator) if a.aggregator else None
    suite = Path(a.regression_suite) if a.regression_suite else Path(a.model) / "regression_suite.npz"
    settings = _settings(a.model)
    alpha = a.alpha if a.alpha is not None else settings.get("alpha", 0.05)
    tau = a.tau if a.tau is not None else settings.get("tau_initial", 0.9)
    eta = settings.get("eta", 0.05)
    return Runtime(data_dir, scorer, agg, alpha=alpha, tau=tau, eta=eta,
                   regression_suite=suite if suite.exists() else None)


def _service(a):
    from ease.aggregate import load_aggregator
    from ease.scorer import ModelScorer
    from ease.service.workspaces import Service

    scorer = ModelScorer(a.model, canonical=not a.fast)
    agg = load_aggregator(a.aggregator) if a.aggregator else None
    suite = Path(a.regression_suite) if a.regression_suite else Path(a.model) / "regression_suite.npz"
    settings = _settings(a.model)
    if a.alpha is not None:
        settings["alpha"] = a.alpha
    if a.tau is not None:
        settings["tau_initial"] = a.tau
    admin_key = os.environ.get("EASE_ADMIN_KEY") or None
    return Service(a.data, scorer, agg, settings=settings, regression_suite=suite if suite.exists() else None,
                   local=not a.multi, admin_key=admin_key)


def _offline_service(data_dir: str):
    """Administration without loading a model: workspaces and backups only need the files."""
    from ease.service.workspaces import Service

    class NoScorer:
        version = "none"

    return Service(data_dir, NoScorer(), None, local=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ease", description="EASE-Delta: decisions that stay current as evidence changes.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", required=True, help="directory of the trained edge model")
        p.add_argument("--aggregator", default=None, help="directory of a trained aggregator; rules if omitted")
        p.add_argument("--regression-suite", default=None)
        p.add_argument("--alpha", type=float, default=None,
                       help="target error rate among endorsed actions (default: release settings, else 0.05)")
        p.add_argument("--tau", type=float, default=None,
                       help="initial endorsement threshold (default: release settings, else 0.9)")
        p.add_argument("--fast", action="store_true",
                       help="dynamic batching: faster, values vary by about 1e-5 with batch composition")

    p = sub.add_parser("serve", help="run the HTTP API")
    common(p)
    p.add_argument("--data", default=str(Path.home() / ".ease"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8791)
    p.add_argument("--multi", action="store_true",
                   help="many workspaces, each behind its own key (create them with `ease workspace create`)")

    p = sub.add_parser("demo", help="run the hand-off packet scenario in the terminal")
    common(p)
    p.add_argument("--data", default=None, help="keep the demo's data here; a temporary directory otherwise")
    p.add_argument("--json", default=None, help="also write the final state to this file")

    p = sub.add_parser("info", help="describe a trained model")
    p.add_argument("--model", required=True)

    p = sub.add_parser("workspace", help="create and manage workspaces of a --multi service")
    p.add_argument("action", choices=["create", "list", "rotate-key"])
    p.add_argument("--data", required=True)
    p.add_argument("--id", default=None, help="3-32 characters: lower-case letters, digits, '-'")
    p.add_argument("--name", default="")

    p = sub.add_parser("backup", help="copy every database and setting of a service to a dated directory")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)

    a = ap.parse_args(argv)

    if a.cmd == "info":
        d = Path(a.model)
        out = {"model": str(d)}
        for name in ("edge_config.json", "training_manifest.json"):
            if (d / name).exists():
                out[name] = json.loads((d / name).read_text())
        print(json.dumps(out, indent=2))
        return 0

    if a.cmd == "workspace":
        svc = _offline_service(a.data)
        if a.action == "create":
            if not a.id:
                print("--id is required", file=sys.stderr)
                return 2
            ws, key = svc.create_workspace(a.id, a.name)
            print(json.dumps({"id": ws.id, "name": ws.name, "key": key,
                              "note": "the key is shown once; only its hash is stored"}, indent=2))
        elif a.action == "list":
            print(json.dumps({"workspaces": svc.list_workspaces()}, indent=2))
        else:
            if not a.id:
                print("--id is required", file=sys.stderr)
                return 2
            print(json.dumps({"id": a.id, "key": svc.rotate_key(a.id)}, indent=2))
        return 0

    if a.cmd == "backup":
        svc = _offline_service(a.data)
        print(json.dumps(svc.backup(a.out), indent=2))
        return 0

    if a.cmd == "demo":
        from ease import demo

        tmp = None
        data = a.data
        if data is None:
            tmp = tempfile.TemporaryDirectory(prefix="ease-demo-")
            data = tmp.name
        rt = _runtime(a, data)
        try:
            res = demo.run(rt)
            if a.json:
                Path(a.json).write_text(json.dumps(res, indent=2, default=str))
        finally:
            rt.close()
            if tmp is not None:
                tmp.cleanup()
        return 0

    if a.cmd == "serve":
        import uvicorn

        from ease.service.api import create_app

        if a.host not in ("127.0.0.1", "localhost", "::1") and not a.multi:
            print(f"warning: binding to {a.host} without --multi exposes a keyless API beyond this machine",
                  file=sys.stderr)
        svc = _service(a)
        try:
            uvicorn.run(create_app(svc), host=a.host, port=a.port, log_level="info")
        finally:
            svc.close()
        return 0
    return 2


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
