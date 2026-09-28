#!/usr/bin/env python3
"""
Run UAV-UGV training-component ablation studies.

Default experiments:

    full
        BC + PPO + SIL

    no_bc
        PPO + SIL

    no_sil
        BC + PPO

    pure_rl
        PPO only

Example
-------

Run all experiments:

    python scripts/run_ablation.py

Only selected experiments:

    python scripts/run_ablation.py \
        --configs no_bc no_sil pure_rl

Only Medium scenarios:

    python scripts/run_ablation.py \
        --scale Medium

Results are stored in:

    results/ablation_results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ============================================================
# Make src/ importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from uav_ugv_rl.ablation import (  # noqa: E402
    ABLATION_CONFIGS,
    run_all_ablations,
)
from uav_ugv_rl.utils import (  # noqa: E402
    device_summary,
    load_scenarios,
)

# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ablation experiments for " "UAV-UGV cooperative routing."
        )
    )

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=(PROJECT_ROOT / "data" / "evaluation_scenarios.json"),
        help="Scenario JSON file.",
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(ABLATION_CONFIGS.keys()),
        default=[
            "full",
            "no_bc",
            "no_sil",
            "pure_rl",
        ],
        help="Ablation configurations to run.",
    )

    parser.add_argument(
        "--scenario-id",
        nargs="+",
        default=None,
        help=("Run only selected scenario IDs."),
    )

    parser.add_argument(
        "--scale",
        choices=[
            "Small",
            "Medium",
            "Large",
            "small",
            "medium",
            "large",
        ],
        default=None,
        help="Run only one scenario scale.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=(PROJECT_ROOT / "checkpoints" / "ablation"),
        help="Ablation checkpoint directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT_ROOT / "results" / "ablation_results.json"),
        help="Output JSON file.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )

    return parser.parse_args()


# ============================================================
# Scenario filtering
# ============================================================


def filter_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    scenario_ids: list[str] | None,
    scale: str | None,
) -> list[dict[str, Any]]:
    """
    Filter scenarios for the ablation study.
    """

    selected = list(scenarios)

    if scenario_ids is not None:

        requested = set(scenario_ids)

        selected = [
            scenario for scenario in selected if scenario["id"] in requested
        ]

    if scale is not None:

        selected = [
            scenario
            for scenario in selected
            if str(
                scenario.get(
                    "scale_name",
                    "",
                )
            ).lower()
            == scale.lower()
        ]

    return selected


# ============================================================
# Main
# ============================================================


def main() -> int:
    args = parse_args()

    scenarios = load_scenarios(args.scenarios)

    scenarios = filter_scenarios(
        scenarios,
        scenario_ids=(args.scenario_id),
        scale=(args.scale),
    )

    if not scenarios:
        raise RuntimeError("No scenarios matched the requested filters.")

    print("\n" + "=" * 72)

    print("UAV-UGV ABLATION STUDY")

    print("=" * 72)

    print(f"Device       : {device_summary()}")

    print(f"Scenarios    : {len(scenarios)}")

    print("Configurations: " + ", ".join(args.configs))

    print(f"Output       : {args.output}")

    print("=" * 72)

    run_all_ablations(
        scenarios=scenarios,
        configs=args.configs,
        checkpoint_dir=(args.checkpoint_dir),
        save_to=(args.output),
        verbose=(not args.quiet),
    )

    print(f"\nAblation study completed." f"\nResults: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
