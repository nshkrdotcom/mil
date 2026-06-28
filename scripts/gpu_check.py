#!/usr/bin/env python3
"""Verify that PyTorch can run real CUDA kernels on this GPU."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mil.gpu import MATMUL_SLOW_THRESHOLD_MS, run_cuda_matmul_check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--slow-threshold-ms", type=float, default=MATMUL_SLOW_THRESHOLD_MS)
    args = parser.parse_args(argv)

    result = run_cuda_matmul_check(
        size=args.size,
        dtype_name=args.dtype,
        slow_threshold_ms=args.slow_threshold_ms,
    )
    print("\n".join(result.messages))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
