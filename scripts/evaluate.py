#!/usr/bin/env python3
"""
Evaluate trained UAV-UGV policies.

Examples
--------
Evaluate one scenario:

    python scripts/evaluate.py \
        --scenario-id small_01

The default checkpoint is:

    checkpoints/small_01.pth

Use a custom checkpoint:

    python scripts/evaluate.py \
        --scenario-id small_01 \
        --checkpoint checkpoints/my_model.pth

Evaluate all available scenario checkpoints:

    python scripts/evaluate.py \
        --checkpoint-dir checkpoints

Results:

    results/evaluation_results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

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


from uav_ugv_rl.agent import SuperiorityAgent  # noqa: E402
from uav_ugv_rl.environment import UAVUGVEnv  # noqa: E402
from uav_ugv_rl.heuristics import generate_expert_plan  # noqa: E402
from uav_ugv_rl.training import evaluate_agent  # noqa: E402
from uav_ugv_rl.utils import (  # noqa: E402
    device_summary,
    get_device,
    is_large_scale,
    load_scenarios,
    percentage_improvement,
    save_json,
)

# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate trained UAV-UGV routing policies.")
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
        help="Evaluate selected scenario IDs.",
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
        help="Evaluate one scenario scale.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint for a single scenario. "
            "Cannot be used for multiple scenarios."
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=(PROJECT_ROOT / "checkpoints"),
        help=("Directory containing " "<scenario_id>.pth checkpoints."),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT_ROOT / "results" / "evaluation_results.json"),
        help="Output JSON file.",
    )

    parser.add_argument(
        "--device",
        choices=[
            "cuda",
            "mps",
            "cpu",
        ],
        default=None,
        help="Evaluation device.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=("Action temperature when " "using stochastic evaluation."),
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help=(
            "Sample from the policy instead of "
            "using deterministic argmax actions."
        ),
    )

    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=("Do not calculate the heuristic expert baseline."),
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
# Checkpoint helpers
# ============================================================


def load_checkpoint_metadata(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Read agent configuration stored in a checkpoint.

    Legacy raw state_dict checkpoints simply return {}.
    """

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        return {}

    config = checkpoint.get("agent_config")

    if not isinstance(
        config,
        dict,
    ):
        return {}

    return config


def create_agent_from_checkpoint(
    *,
    checkpoint_path: Path,
    scenario: dict[str, Any],
    device: torch.device,
) -> SuperiorityAgent:
    """
    Reconstruct an agent compatible with a saved checkpoint.
    """

    config = load_checkpoint_metadata(
        checkpoint_path,
        device,
    )

    num_tasks = int(
        config.get(
            "num_tasks",
            scenario["num_tasks"],
        )
    )

    is_large = bool(
        config.get(
            "is_large",
            is_large_scale(
                scenario["num_tasks"],
                scenario["map_size"],
            ),
        )
    )

    agent = SuperiorityAgent(
        num_tasks=num_tasks,
        is_large=is_large,
        lr=float(
            config.get(
                "lr",
                2e-4,
            )
        ),
        bc_weight=float(
            config.get(
                "bc_weight",
                1.0,
            )
        ),
        sil_weight=float(
            config.get(
                "sil_weight",
                0.5,
            )
        ),
        gamma=float(
            config.get(
                "gamma",
                (0.995 if is_large else 0.99),
            )
        ),
        gae_lambda=float(
            config.get(
                "gae_lambda",
                0.95,
            )
        ),
        clip_eps=float(
            config.get(
                "clip_eps",
                0.2,
            )
        ),
        ppo_epochs=int(
            config.get(
                "ppo_epochs",
                (4 if is_large else 3),
            )
        ),
        entropy_coef=float(
            config.get(
                "entropy_coef",
                (0.08 if is_large else 0.05),
            )
        ),
        entropy_min=float(
            config.get(
                "entropy_min",
                0.01,
            )
        ),
        dropout=float(
            config.get(
                "dropout",
                0.1,
            )
        ),
        weight_decay=float(
            config.get(
                "weight_decay",
                1e-4,
            )
        ),
        max_best_buffer=int(
            config.get(
                "max_best_buffer",
                (100 if is_large else 50),
            )
        ),
        device=device,
        use_amp=(device.type == "cuda"),
        compile_model=False,
    )

    agent.load(
        checkpoint_path,
        load_optimizer=False,
        load_sil_buffer=False,
    )

    agent.eval()

    return agent


# ============================================================
# Evaluate one scenario
# ============================================================


def evaluate_scenario(
    *,
    scenario: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    deterministic: bool,
    temperature: float,
    calculate_baseline: bool,
) -> dict[str, Any]:
    """
    Evaluate one trained model.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    nodes = scenario["nodes_loc"]

    num_tasks = int(scenario["num_tasks"])

    map_size = float(scenario["map_size"])

    agent = create_agent_from_checkpoint(
        checkpoint_path=checkpoint_path,
        scenario=scenario,
        device=device,
    )

    env = UAVUGVEnv(
        num_tasks=num_tasks,
        map_size=map_size,
    )

    metrics = evaluate_agent(
        env=env,
        agent=agent,
        nodes=nodes,
        map_size=map_size,
        deterministic=deterministic,
        temperature=temperature,
    )

    result: dict[
        str,
        Any,
    ] = {
        "scenario_id": (scenario["id"]),
        "scale_name": (scenario.get("scale_name")),
        "num_tasks": (num_tasks),
        "map_size": (map_size),
        "checkpoint": str(checkpoint_path),
        **metrics,
    }

    # --------------------------------------------------------
    # Heuristic baseline
    # --------------------------------------------------------

    if calculate_baseline:

        expert_plan, _ = generate_expert_plan(
            nodes_loc=nodes,
            map_size_km=map_size,
            uav_solver="heuristic",
        )

        baseline = float(expert_plan.makespan_s)

        result["baseline_makespan_s"] = baseline

        result["vs_baseline_percent"] = percentage_improvement(
            baseline,
            metrics["makespan_s"],
        )

    return result


# ============================================================
# Printing
# ============================================================


def print_result(
    result: dict[str, Any],
) -> None:
    """
    Print one evaluation result.
    """

    print("\n" + "-" * 72)

    print(f"Scenario : " f"{result['scenario_id']}")

    print(f"Scale    : " f"{result.get('scale_name', '?')}")

    print(f"Success  : " f"{result['success']}")

    print(f"Makespan : " f"{result['makespan_s']:.2f} s")

    print(f"Energy   : " f"{result['energy_kj']:.2f} kJ")

    print(f"Fly ratio: " f"{result['flying_ratio']:.1%}")

    if "baseline_makespan_s" in result:

        print(f"Baseline : " f"{result['baseline_makespan_s']:.2f} s")

        print(f"vs base  : " f"{result['vs_baseline_percent']:+.2f}%")


# ============================================================
# Main
# ============================================================


def main() -> int:
    args = parse_args()

    if args.temperature <= 0:
        raise ValueError("temperature must be greater than 0.")

    device = get_device(args.device)

    scenarios = load_scenarios(args.scenarios)

    scenarios = filter_scenarios(
        scenarios,
        scenario_ids=(args.scenario_id),
        scale=(args.scale),
    )

    if not scenarios:
        raise RuntimeError("No scenarios matched the requested filters.")

    if args.checkpoint is not None and len(scenarios) != 1:
        raise ValueError(
            "--checkpoint can only be used "
            "when exactly one scenario is selected."
        )

    print("\n" + "=" * 72)

    print("UAV-UGV POLICY EVALUATION")

    print("=" * 72)

    print(f"Device    : " f"{device_summary(device)}")

    print(f"Scenarios : " f"{len(scenarios)}")

    print(
        f"Mode      : "
        f"{'stochastic' if args.stochastic else 'deterministic'}"
    )

    print("=" * 72)

    results: list[dict[str, Any]] = []

    for scenario in scenarios:

        if args.checkpoint is not None:

            checkpoint_path = args.checkpoint

        else:

            checkpoint_path = args.checkpoint_dir / (scenario["id"] + ".pth")

        result = evaluate_scenario(
            scenario=scenario,
            checkpoint_path=(checkpoint_path),
            device=device,
            deterministic=(not args.stochastic),
            temperature=(args.temperature),
            calculate_baseline=(not args.skip_baseline),
        )

        results.append(result)

        print_result(result)

    save_json(
        {
            "num_scenarios": len(results),
            "results": results,
        },
        args.output,
    )

    print("\n" + "=" * 72)

    print(f"Evaluation results saved to:" f"\n  {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
