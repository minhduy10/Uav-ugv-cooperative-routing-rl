#!/usr/bin/env python3
"""
Train UAV-UGV cooperative routing agents.

Examples
--------
Train all scenarios:

    python scripts/train.py

Train using Small configuration:

    python scripts/train.py \
        --config configs/small.yaml

Train only Small scenarios:

    python scripts/train.py \
        --config configs/small.yaml \
        --scale Small

Train one scenario:

    python scripts/train.py \
        --scenario-id small_01

Results:
    results/training_results.json

Checkpoints:
    checkpoints/<scenario_id>.pth
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

# ============================================================
# Make src/ importable when running directly
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from uav_ugv_rl.training import (  # noqa: E402
    TrainingConfig,
    train_scenarios,
)
from uav_ugv_rl.utils import (  # noqa: E402
    device_summary,
    load_scenarios,
    load_yaml,
    save_json,
)

# ============================================================
# Argument parsing
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Train UAV-UGV routing agents " "using BC + PPO + SIL.")
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=("Optional YAML configuration, e.g. " "configs/small.yaml."),
    )

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=(PROJECT_ROOT / "data" / "evaluation_scenarios.json"),
        help="Evaluation scenarios JSON file.",
    )

    parser.add_argument(
        "--scenario-id",
        nargs="+",
        default=None,
        help=(
            "Train only selected scenario IDs. "
            "Example: --scenario-id small_01 small_02"
        ),
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
        help="Train only scenarios of a selected scale.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=(PROJECT_ROOT / "checkpoints"),
        help="Directory used to save checkpoints.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT_ROOT / "results" / "training_results.json"),
        help="JSON file used to store training results.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )

    return parser.parse_args()


# ============================================================
# Configuration helpers
# ============================================================


def _merge_section(
    destination: dict[str, Any],
    source: Any,
) -> None:
    """
    Merge one mapping section into destination.
    """

    if isinstance(
        source,
        Mapping,
    ):
        destination.update(source)


def build_training_config(
    config_data: Mapping[str, Any],
) -> TrainingConfig:
    """
    Convert YAML configuration into TrainingConfig.

    The script accepts both flat configuration:

        bc_epochs: 40
        rl_episodes: 160
        learning_rate: 0.0002

    and nested configuration:

        training:
            bc_epochs: 40
            rl_episodes: 160

        agent:
            learning_rate: 0.0002
            bc_weight: 1.0

        expert:
            expert_solver: heuristic
    """

    combined: dict[
        str,
        Any,
    ] = {}

    # Flat values.
    combined.update(config_data)

    # Common nested layouts.
    _merge_section(
        combined,
        config_data.get("training"),
    )

    _merge_section(
        combined,
        config_data.get("agent"),
    )

    _merge_section(
        combined,
        config_data.get("expert"),
    )

    allowed_fields = {field.name for field in fields(TrainingConfig)}

    filtered = {
        key: value for key, value in combined.items() if key in allowed_fields
    }

    return TrainingConfig(**filtered)


def get_config_scale(
    config_data: Mapping[str, Any],
) -> str | None:
    """
    Find an optional scale name inside config.
    """

    scale = config_data.get("scale_name")

    if scale is None:
        scale = config_data.get("scale")

    environment = config_data.get("environment")

    if scale is None and isinstance(
        environment,
        Mapping,
    ):
        scale = environment.get(
            "scale_name",
            environment.get("scale"),
        )

    if scale is None:
        return None

    return str(scale)


def get_config_num_tasks(
    config_data: Mapping[str, Any],
) -> int | None:
    """
    Find optional num_tasks inside config.
    """

    value = config_data.get("num_tasks")

    environment = config_data.get("environment")

    if value is None and isinstance(
        environment,
        Mapping,
    ):
        value = environment.get("num_tasks")

    if value is None:
        return None

    return int(value)


# ============================================================
# Scenario filtering
# ============================================================


def filter_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    scenario_ids: list[str] | None,
    scale: str | None,
    num_tasks: int | None,
) -> list[dict[str, Any]]:
    """
    Select scenarios requested by the user/configuration.
    """

    selected = list(scenarios)

    if scenario_ids is not None:

        requested = set(scenario_ids)

        selected = [
            scenario for scenario in selected if scenario["id"] in requested
        ]

    if scale is not None:

        scale_lower = scale.lower()

        selected = [
            scenario
            for scenario in selected
            if str(
                scenario.get(
                    "scale_name",
                    "",
                )
            ).lower()
            == scale_lower
        ]

    elif num_tasks is not None:

        selected = [
            scenario
            for scenario in selected
            if int(scenario["num_tasks"]) == num_tasks
        ]

    return selected


# ============================================================
# Main
# ============================================================


def main() -> int:
    args = parse_args()

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config_data: dict[
        str,
        Any,
    ] = {}

    if args.config is not None:

        config_data = load_yaml(args.config)

        print(f"Config: {args.config}")

    training_config = build_training_config(config_data)

    # --------------------------------------------------------
    # Scenarios
    # --------------------------------------------------------

    scenarios = load_scenarios(args.scenarios)

    scale = args.scale or get_config_scale(config_data)

    num_tasks = get_config_num_tasks(config_data)

    scenarios = filter_scenarios(
        scenarios,
        scenario_ids=(args.scenario_id),
        scale=scale,
        num_tasks=num_tasks,
    )

    if not scenarios:
        raise RuntimeError("No scenarios matched the requested filters.")

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print("\n" + "=" * 72)

    print("UAV-UGV TRAINING")

    print("=" * 72)

    print(f"Device      : {device_summary()}")

    print(f"Scenarios   : {len(scenarios)}")

    print(f"BC epochs   : {training_config.bc_epochs}")

    print(f"PPO episodes: {training_config.rl_episodes}")

    print(f"SIL episodes: {training_config.sil_episodes}")

    print(f"Checkpoints : {args.checkpoint_dir}")

    print("=" * 72)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    results = train_scenarios(
        scenarios=scenarios,
        config=training_config,
        checkpoint_dir=(args.checkpoint_dir),
        verbose=(not args.quiet),
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = {
        "config": config_data,
        "num_scenarios": len(scenarios),
        "results": [result.to_dict() for result in results],
    }

    save_json(
        output,
        args.output,
    )

    print(f"\nTraining results saved to:" f"\n  {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
