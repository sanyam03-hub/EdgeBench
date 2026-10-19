"""Stretch S1: per-layer INT8 error analysis and mixed-precision recovery (see s1_study.py).

Needs the dataset and the FP32 export from a normal run first, e.g.:

    edgebench run --config configs/mobilenet_v3_small.yaml --target configs/targets/cpu_4t.yaml
    python scripts/s1_layer_error.py --config configs/mobilenet_v3_small.yaml \
        --target configs/targets/cpu_4t.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from edgebench.config import load_config
from edgebench.data import locate_imagenette
from edgebench.errors import ConfigError, DatasetError, EdgeBenchError
from edgebench.s1_study import StudySettings, run_study, write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default: results/s1/<model>")
    parser.add_argument("--ks", default="0,1,2,4,8,16,32", help="comma-separated k values")
    parser.add_argument("--random-seeds", type=int, default=3)
    parser.add_argument("--analysis-images", type=int, default=32)
    parser.add_argument("--max-drop", type=float, default=1.0, help="pts below FP32 (search half)")
    parser.add_argument("--no-sweep", action="store_true", help="rankings only")
    parser.add_argument(
        "--no-sensitivity", action="store_true", help="skip the per-node sensitivity ranking"
    )
    args = parser.parse_args(argv)

    def say(message: str) -> None:
        print(message, flush=True)

    try:
        run = load_config(args.config, args.target)
        if run.model.dataset != "imagenette2-160":
            raise ConfigError("S1 needs the Imagenette dataset (dataset: imagenette2-160)")
        root = locate_imagenette(Path(os.environ.get("EDGEBENCH_DATA_DIR", "data")))
        artifacts = Path(os.environ.get("EDGEBENCH_ARTIFACTS_DIR", "artifacts")) / run.model.model
        fp32 = artifacts / "onnx_fp32.onnx"
        if not fp32.is_file():
            raise DatasetError(f"{fp32} not found; run `edgebench run` for this model first")
        settings = StudySettings(
            ks=tuple(int(k) for k in args.ks.split(",")),
            random_seeds=args.random_seeds,
            analysis_images=args.analysis_images,
            max_drop_pts=args.max_drop,
            sensitivity=not args.no_sensitivity,
            sweep=not args.no_sweep,
        )
        say(f"S1 layer-error study: {run.model.model} on {run.target.name}")
        start = time.perf_counter()
        result = run_study(run, root, fp32, artifacts / "s1", settings, progress=say)
        out = args.out or Path("results") / "s1" / run.model.model
        for path in write_outputs(result, out):
            say(f"wrote {path}")
        say(f"done in {time.perf_counter() - start:.0f} s")
    except (ConfigError, DatasetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except EdgeBenchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
