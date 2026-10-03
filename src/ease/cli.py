"""Command line.

    ease info   --model runs/stage_a/final
    ease demo   --model runs/stage_a/final --aggregator runs/stage_b/refined_seed1
    ease serve  --model runs/stage_a/final --aggregator runs/stage_b/refined_seed1 --data ~/.ease
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _runtime(a, data_dir):
    from ease.aggregate import load_aggregator
    from ease.runtime import Runtime
    from ease.scorer import ModelScorer

    scorer = ModelScorer(a.model, canonical=not a.fast)
    agg = load_aggregator(a.aggregator) if a.aggregator else None
    suite = Path(a.regression_suite) if a.regression_suite else Path(a.model) / "regression_suite.npz"
    return Runtime(data_dir, scorer, agg, alpha=a.alpha, tau=a.tau,
                   regression_suite=suite if suite.exists() else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ease", description="EASE-Delta: decisions that stay current as evidence changes.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", required=True, help="directory of the trained edge model")
        p.add_argument("--aggregator", default=None, help="directory of a trained aggregator; rules if omitted")
        p.add_argument("--regression-suite", default=None)
        p.add_argument("--alpha", type=float, default=0.05, help="target error rate among endorsed actions")
        p.add_argument("--tau", type=float, default=0.9, help="initial endorsement threshold")
        p.add_argument("--fast", action="store_true",
                       help="dynamic batching: faster, values vary by about 1e-5 with batch composition")

    p = sub.add_parser("serve", help="run the local HTTP API")
    common(p)
    p.add_argument("--data", default=str(Path.home() / ".ease"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8791)

    p = sub.add_parser("demo", help="run the hand-off packet scenario in the terminal")
    common(p)
    p.add_argument("--data", default=None, help="keep the demo's data here; a temporary directory otherwise")
    p.add_argument("--json", default=None, help="also write the final state to this file")

    p = sub.add_parser("info", help="describe a trained model")
    p.add_argument("--model", required=True)

    a = ap.parse_args(argv)

    if a.cmd == "info":
        d = Path(a.model)
        out = {"model": str(d)}
        for name in ("edge_config.json", "training_manifest.json"):
            if (d / name).exists():
                out[name] = json.loads((d / name).read_text())
        print(json.dumps(out, indent=2))
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

        if a.host not in ("127.0.0.1", "localhost", "::1"):
            print(f"warning: binding to {a.host} exposes the API beyond this machine; it has no authentication",
                  file=sys.stderr)
        rt = _runtime(a, a.data)
        try:
            uvicorn.run(create_app(rt), host=a.host, port=a.port, log_level="info")
        finally:
            rt.close()
        return 0
    return 2


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
