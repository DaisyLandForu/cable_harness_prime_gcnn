#!/usr/bin/env python3
"""Verify CUDA visibility and write a non-training S05 GPU preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import torch


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402
from steiner_branching.learning.teacher_data import EXPECTED_STACK_ID  # noqa: E402


DEFAULT_OUTPUT = REPO / "results/steiner/raw/s05/gpu_preflight.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if os.environ.get("STEINER_SOLVER_STACK_ID") != EXPECTED_STACK_ID:
        raise SystemExit("run through scripts/steiner/run_with_scip804.sh --python")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is not available to the frozen S05 Python stack")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
        )
    left = torch.arange(256 * 256, device="cuda:0", dtype=torch.float32).reshape(256, 256)
    checksum = float((left @ left.T).sum().cpu())
    if not torch.isfinite(torch.tensor(checksum)):
        raise RuntimeError("CUDA smoke computation produced a non-finite checksum")
    value = {
        "schema_version": 1,
        "stage": "S05",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "solver_stack_id": EXPECTED_STACK_ID,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": len(devices),
        "devices": devices,
        "smoke_checksum": checksum,
        "training_performed": False,
    }
    atomic_write_json(Path(args.output), value)
    print(Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
