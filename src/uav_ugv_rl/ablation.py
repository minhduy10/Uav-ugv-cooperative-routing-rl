"""
Ablation study utilities for UAV-UGV cooperative routing.

This module evaluates the contribution of individual training
components by enabling or disabling:

    - Behavioral Cloning (BC)
    - PPO reinforcement learning
    - Self-Imitation Learning (SIL)

Default ablation configurations
-------------------------------

1. full
    BC + PPO + SIL

2. no_bc
    PPO + SIL

3. no_sil
    BC + PPO

4. pure_rl
    PPO only

The implementation reuses the common training pipeline from
``training.py`` so that all configurations share:

    - the same environment,
    - the same Transformer policy,
    - the same reward function,
    - the same physics,
    - the same evaluation procedure.

This makes comparisons between ablation variants more consistent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .training import (
    TrainingConfig,
    TrainingResult,
    train_scenario,
)

# ============================================================
# Ablation configuration
# ============================================================


@dataclass(frozen=True)
class AblationConfig:
    """
    Configuration for one ablation experiment.

    Parameters
    ----------
    key:
        Short machine-readable experiment name.

    name:
        Human-readable experiment name.

    description:
        Description of the removed / enabled components.

    enable_bc:
        Enable Behavioral Cloning.

    enable_ppo:
        Enable PPO reinforcement learning.

    enable_sil:
        Enable Self-Imitation Learning.

    bc_epochs:
        Number of BC epochs when BC is enabled.

    rl_episodes:
        Number of PPO episodes when PPO is enabled.

    sil_episodes:
        Number of SIL episodes when SIL is enabled.

    learning_rate:
        Base optimizer learning rate.

    bc_weight:
        Behavioral Cloning loss weight.

    sil_weight:
        Self-Imitation loss weight.

    seed:
        Base random seed.
    """

    key: str

    name: str

    description: str

    enable_bc: bool = True
    enable_ppo: bool = True
    enable_sil: bool = True

    bc_epochs: int = 50
    rl_episodes: int = 200
    sil_episodes: int = 200

    learning_rate: float = 2e-4

    bc_weight: float = 1.0
    sil_weight: float = 0.5

    rl_temperature_start: float = 1.5
    rl_temperature_end: float = 0.7

    sil_temperature: float = 0.8

    max_steps_multiplier: int = 5

    expert_solver: str = "heuristic"
    cpsat_limit_s: float = 10.0

    seed: int = 42

    def __post_init__(self) -> None:

        if not self.key:
            raise ValueError("Ablation key cannot be empty.")

        if self.bc_epochs < 0:
            raise ValueError("bc_epochs must be >= 0.")

        if self.rl_episodes < 0:
            raise ValueError("rl_episodes must be >= 0.")

        if self.sil_episodes < 0:
            raise ValueError("sil_episodes must be >= 0.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")

        if self.expert_solver not in {
            "heuristic",
            "cpsat",
        }:
            raise ValueError(
                "expert_solver must be " "'heuristic' or 'cpsat'."
            )

    # ========================================================
    # Convert to normal training configuration
    # ========================================================

    def to_training_config(
        self,
        *,
        checkpoint_path: str | None = None,
        seed: int | None = None,
    ) -> TrainingConfig:
        """
        Convert an ablation configuration into TrainingConfig.

        Disabled phases automatically receive zero epochs.
        """

        return TrainingConfig(
            bc_epochs=(self.bc_epochs if self.enable_bc else 0),
            rl_episodes=(self.rl_episodes if self.enable_ppo else 0),
            sil_episodes=(self.sil_episodes if self.enable_sil else 0),
            learning_rate=(self.learning_rate),
            bc_weight=(self.bc_weight),
            sil_weight=(self.sil_weight),
            rl_temperature_start=(self.rl_temperature_start),
            rl_temperature_end=(self.rl_temperature_end),
            sil_temperature=(self.sil_temperature),
            max_steps_multiplier=(self.max_steps_multiplier),
            expert_solver=(self.expert_solver),
            cpsat_limit_s=(self.cpsat_limit_s),
            save_best=(checkpoint_path is not None),
            checkpoint_path=(checkpoint_path),
            seed=(self.seed if seed is None else seed),
        )


# ============================================================
# Standard ablation configurations
# ============================================================

ABLATION_CONFIGS: dict[
    str,
    AblationConfig,
] = {
    # --------------------------------------------------------
    # Full model
    # --------------------------------------------------------
    "full": AblationConfig(
        key="full",
        name="Full (BC + PPO + SIL)",
        description=(
            "Complete three-phase training pipeline using "
            "Behavioral Cloning, PPO fine-tuning, and "
            "Self-Imitation Learning."
        ),
        enable_bc=True,
        enable_ppo=True,
        enable_sil=True,
        bc_epochs=50,
        rl_episodes=200,
        sil_episodes=200,
    ),
    # --------------------------------------------------------
    # Remove Behavioral Cloning
    # --------------------------------------------------------
    "no_bc": AblationConfig(
        key="no_bc",
        name="No BC (PPO + SIL)",
        description=(
            "Behavioral Cloning is removed. "
            "The policy starts from random initialization "
            "and learns using PPO followed by SIL."
        ),
        enable_bc=False,
        enable_ppo=True,
        enable_sil=True,
        bc_epochs=0,
        rl_episodes=250,
        sil_episodes=200,
    ),
    # --------------------------------------------------------
    # Remove Self-Imitation Learning
    # --------------------------------------------------------
    "no_sil": AblationConfig(
        key="no_sil",
        name="No SIL (BC + PPO)",
        description=(
            "Self-Imitation Learning is removed. "
            "Training uses expert Behavioral Cloning "
            "followed by PPO fine-tuning."
        ),
        enable_bc=True,
        enable_ppo=True,
        enable_sil=False,
        bc_epochs=100,
        rl_episodes=350,
        sil_episodes=0,
    ),
    # --------------------------------------------------------
    # Pure reinforcement learning
    # --------------------------------------------------------
    "pure_rl": AblationConfig(
        key="pure_rl",
        name="Pure RL (PPO Only)",
        description=(
            "No expert-guided Behavioral Cloning and no "
            "Self-Imitation Learning. The Transformer policy "
            "is trained only with PPO."
        ),
        enable_bc=False,
        enable_ppo=True,
        enable_sil=False,
        bc_epochs=0,
        rl_episodes=450,
        sil_episodes=0,
    ),
}


# ============================================================
# Scenario helpers
# ============================================================


def _get_scenario_nodes(
    scenario: dict[str, Any],
) -> np.ndarray:
    """
    Extract node coordinates from a scenario.

    Supported keys:

        nodes_loc
        nodes
    """

    if "nodes_loc" in scenario:

        nodes = scenario["nodes_loc"]

    elif "nodes" in scenario:

        nodes = scenario["nodes"]

    else:

        raise KeyError("Scenario must contain " "'nodes_loc' or 'nodes'.")

    nodes_array = np.asarray(
        nodes,
        dtype=np.float32,
    )

    if nodes_array.ndim != 2 or nodes_array.shape[1] != 2:
        raise ValueError("Scenario nodes must have " "shape (num_nodes, 2).")

    if len(nodes_array) < 2:
        raise ValueError(
            "Scenario must contain a depot " "and at least one task."
        )

    return nodes_array


def _get_scenario_id(
    scenario: dict[str, Any],
    index: int,
) -> str:
    """
    Return a stable scenario identifier.
    """

    return str(
        scenario.get(
            "id",
            f"scenario_{index + 1}",
        )
    )


# ============================================================
# Best-history helper
# ============================================================


def _find_best_history_record(
    result: TrainingResult,
) -> dict[str, Any] | None:
    """
    Find the successful history record with minimum makespan.
    """

    successful_records = [
        record
        for record in result.history
        if record.get(
            "success",
            False,
        )
    ]

    if not successful_records:
        return None

    return min(
        successful_records,
        key=lambda record: record["makespan_s"],
    )


# ============================================================
# Scenario-result conversion
# ============================================================


def _build_scenario_result(
    *,
    scenario_id: str,
    result: TrainingResult,
) -> dict[str, Any]:
    """
    Convert TrainingResult into ablation-friendly metrics.
    """

    best_history = _find_best_history_record(result)

    flying_ratio: float | None = None

    flying_steps = 0
    riding_steps = 0

    if best_history is not None:

        flying_steps = int(
            best_history.get(
                "flying_steps",
                0,
            )
        )

        riding_steps = int(
            best_history.get(
                "riding_steps",
                0,
            )
        )

        total_motion_steps = flying_steps + riding_steps

        flying_ratio = flying_steps / max(
            1,
            total_motion_steps,
        )

    return {
        "scenario_id": scenario_id,
        "num_tasks": (result.num_tasks),
        "map_size": (result.map_size),
        "baseline_makespan_s": (result.baseline_makespan_s),
        "best_makespan_s": (result.best_makespan_s),
        "best_energy_kj": (result.best_energy_kj),
        "improvement_percent": (result.best_improvement_percent),
        "success": (result.success),
        "flying_steps": (flying_steps),
        "riding_steps": (riding_steps),
        "flying_ratio": (flying_ratio),
        "checkpoint_path": (result.checkpoint_path),
        "history": (result.history),
    }


# ============================================================
# Summary calculation
# ============================================================


def summarize_ablation_results(
    scenario_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute aggregated metrics for one ablation configuration.
    """

    total_scenarios = len(scenario_results)

    successful = [
        result
        for result in scenario_results
        if result.get(
            "success",
            False,
        )
    ]

    success_count = len(successful)

    success_rate = (
        success_count / total_scenarios if total_scenarios > 0 else 0.0
    )

    makespans = [
        float(result["best_makespan_s"])
        for result in successful
        if result.get("best_makespan_s") is not None
    ]

    energies = [
        float(result["best_energy_kj"])
        for result in successful
        if result.get("best_energy_kj") is not None
    ]

    improvements = [
        float(result["improvement_percent"])
        for result in successful
        if result.get("improvement_percent") is not None
    ]

    flying_ratios = [
        float(result["flying_ratio"])
        for result in successful
        if result.get("flying_ratio") is not None
    ]

    beat_baseline_count = sum(
        1
        for result in successful
        if (
            result.get("improvement_percent") is not None
            and result["improvement_percent"] > 0
        )
    )

    return {
        "num_scenarios": (total_scenarios),
        "successful_scenarios": (success_count),
        "success_rate": (success_rate),
        "beat_baseline_count": (beat_baseline_count),
        "avg_best_makespan_s": (
            float(np.mean(makespans)) if makespans else None
        ),
        "avg_energy_kj": (float(np.mean(energies)) if energies else None),
        "avg_flying_ratio": (
            float(np.mean(flying_ratios)) if flying_ratios else None
        ),
        "avg_improvement_percent": (
            float(np.mean(improvements)) if improvements else None
        ),
    }


# ============================================================
# Run one ablation configuration
# ============================================================


def run_ablation_training(
    config: AblationConfig,
    scenarios: Sequence[dict[str, Any]],
    *,
    checkpoint_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run one ablation configuration on multiple scenarios.

    Each scenario receives a newly initialized agent so the
    experiments remain independent.

    Parameters
    ----------
    config:
        Ablation configuration.

    scenarios:
        Evaluation scenarios.

    checkpoint_dir:
        Optional directory for model checkpoints.

    verbose:
        Print experiment progress.

    Returns
    -------
    dict
        Ablation results and aggregate statistics.
    """

    if len(scenarios) == 0:
        raise ValueError("At least one scenario is required.")

    if verbose:

        print("\n" + "=" * 78)

        print(f"ABLATION: {config.name}")

        print(config.description)

        print("-" * 78)

        print(
            f"BC  : "
            f"{'ON' if config.enable_bc else 'OFF'} "
            f"({config.bc_epochs if config.enable_bc else 0} epochs)"
        )

        print(
            f"PPO : "
            f"{'ON' if config.enable_ppo else 'OFF'} "
            f"({config.rl_episodes if config.enable_ppo else 0} episodes)"
        )

        print(
            f"SIL : "
            f"{'ON' if config.enable_sil else 'OFF'} "
            f"({config.sil_episodes if config.enable_sil else 0} episodes)"
        )

        print("=" * 78)

    scenario_results: list[dict[str, Any]] = []

    for index, scenario in enumerate(scenarios):

        scenario_id = _get_scenario_id(
            scenario,
            index,
        )

        nodes = _get_scenario_nodes(scenario)

        map_size = float(
            scenario.get(
                "map_size",
                10.0,
            )
        )

        scale_name = str(
            scenario.get(
                "scale_name",
                "",
            )
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        checkpoint_path: str | None = None

        if checkpoint_dir is not None:

            checkpoint_path = str(
                Path(checkpoint_dir) / config.key / f"{scenario_id}.pth"
            )

        # ----------------------------------------------------
        # Same scenario receives same seed across
        # all ablation configurations.
        # ----------------------------------------------------

        scenario_seed = config.seed + index

        training_config = config.to_training_config(
            checkpoint_path=(checkpoint_path),
            seed=(scenario_seed),
        )

        if verbose:

            print(
                f"\n" f"[{index + 1}/" f"{len(scenarios)}] " f"{scenario_id}"
            )

            print(
                f"Tasks: "
                f"{len(nodes) - 1} | "
                f"Map: "
                f"{map_size:.1f} km"
                + (f" | Scale: " f"{scale_name}" if scale_name else "")
            )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        _, training_result = train_scenario(
            nodes=nodes,
            map_size=map_size,
            config=training_config,
            verbose=verbose,
        )

        scenario_result = _build_scenario_result(
            scenario_id=(scenario_id),
            result=(training_result),
        )

        scenario_result["scale_name"] = scale_name

        scenario_results.append(scenario_result)

    summary = summarize_ablation_results(scenario_results)

    return {
        "key": config.key,
        "name": config.name,
        "description": (config.description),
        "config": asdict(config),
        "bc_enabled": (config.enable_bc),
        "ppo_enabled": (config.enable_ppo),
        "sil_enabled": (config.enable_sil),
        "scenario_results": (scenario_results),
        "summary": summary,
    }


# ============================================================
# Run all ablation configurations
# ============================================================


def run_all_ablations(
    scenarios: Sequence[dict[str, Any]],
    *,
    configs: Sequence[str] | None = None,
    checkpoint_dir: str | Path | None = None,
    save_to: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Run multiple ablation configurations.

    Parameters
    ----------
    scenarios:
        Evaluation scenarios.

    configs:
        Configuration names.

        Default:

            full
            no_bc
            no_sil
            pure_rl

    checkpoint_dir:
        Optional checkpoint directory.

    save_to:
        Optional JSON output path.

    verbose:
        Print training progress.

    Examples
    --------
    Run all configurations:

        results = run_all_ablations(
            scenarios
        )

    Run selected configurations:

        results = run_all_ablations(
            scenarios,
            configs=[
                "full",
                "no_bc",
            ],
        )
    """

    if configs is None:

        configs = [
            "full",
            "no_bc",
            "no_sil",
            "pure_rl",
        ]

    if len(configs) == 0:
        raise ValueError("At least one ablation " "configuration is required.")

    # --------------------------------------------------------
    # Validate config names before expensive training starts.
    # --------------------------------------------------------

    unknown_configs = [
        config_name
        for config_name in configs
        if config_name not in ABLATION_CONFIGS
    ]

    if unknown_configs:

        raise KeyError(
            "Unknown ablation "
            "configuration(s): " + ", ".join(unknown_configs)
        )

    if verbose:

        print("\n" + "#" * 78)

        print("# UAV-UGV ABLATION STUDY")

        print("# Configurations: " + ", ".join(configs))

        print(f"# Scenarios: " f"{len(scenarios)}")

        print("#" * 78)

    all_results: dict[str, dict[str, Any]] = {}

    for config_name in configs:

        config = ABLATION_CONFIGS[config_name]

        result = run_ablation_training(
            config=config,
            scenarios=scenarios,
            checkpoint_dir=(checkpoint_dir),
            verbose=verbose,
        )

        all_results[config_name] = result

    if verbose:

        print_ablation_summary(all_results)

    if save_to is not None:

        save_ablation_results(
            all_results,
            save_to,
        )

        if verbose:
            print(f"\nResults saved to: " f"{save_to}")

    return all_results


# ============================================================
# Print summary
# ============================================================


def print_ablation_summary(
    all_results: dict[str, dict[str, Any]],
) -> None:
    """
    Print an aggregate ablation comparison table.
    """

    print("\n" + "=" * 118)

    print("ABLATION STUDY SUMMARY")

    print("=" * 118)

    header = (
        f"{'Configuration':<24} | "
        f"{'BC':^5} | "
        f"{'PPO':^5} | "
        f"{'SIL':^5} | "
        f"{'Success':>9} | "
        f"{'Makespan':>12} | "
        f"{'Energy':>11} | "
        f"{'Fly Ratio':>10} | "
        f"{'vs Expert':>10}"
    )

    print(header)

    print("-" * 118)

    for key, result in all_results.items():

        summary = result["summary"]

        success_rate = float(summary["success_rate"])

        makespan = summary["avg_best_makespan_s"]

        energy = summary["avg_energy_kj"]

        fly_ratio = summary["avg_flying_ratio"]

        improvement = summary["avg_improvement_percent"]

        makespan_text = f"{makespan:.1f}s" if makespan is not None else "N/A"

        energy_text = f"{energy:.2f}kJ" if energy is not None else "N/A"

        fly_text = f"{fly_ratio:.1%}" if fly_ratio is not None else "N/A"

        improvement_text = (
            f"{improvement:+.2f}%" if improvement is not None else "N/A"
        )

        print(
            f"{key:<24} | "
            f"{'Y' if result['bc_enabled'] else 'N':^5} | "
            f"{'Y' if result['ppo_enabled'] else 'N':^5} | "
            f"{'Y' if result['sil_enabled'] else 'N':^5} | "
            f"{success_rate:>8.1%} | "
            f"{makespan_text:>12} | "
            f"{energy_text:>11} | "
            f"{fly_text:>10} | "
            f"{improvement_text:>10}"
        )

    print("=" * 118)


# ============================================================
# JSON serialization
# ============================================================


def _to_json_serializable(
    value: Any,
) -> Any:
    """
    Recursively convert NumPy objects into JSON-compatible values.
    """

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        dict,
    ):
        return {
            key: _to_json_serializable(item) for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [_to_json_serializable(item) for item in value]

    return value


def save_ablation_results(
    results: dict[str, dict[str, Any]],
    path: str | Path,
) -> None:
    """
    Save ablation results to a JSON file.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable = _to_json_serializable(results)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serializable,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Sample scenarios
# ============================================================


def generate_sample_scenarios(
    *,
    num_scenarios: int = 3,
    num_tasks: int = 10,
    map_size: float = 10.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate small scenarios for testing the ablation pipeline.

    These scenarios are intended for code verification rather than
    final research evaluation.
    """

    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be > 0.")

    if num_tasks <= 0:
        raise ValueError("num_tasks must be > 0.")

    if map_size <= 0:
        raise ValueError("map_size must be > 0.")

    rng = np.random.default_rng(seed)

    scenarios: list[dict[str, Any]] = []

    for index in range(num_scenarios):

        nodes = rng.uniform(
            low=0.0,
            high=map_size,
            size=(
                num_tasks + 1,
                2,
            ),
        )

        # Same depot convention as environment.py.
        nodes[0] = [
            map_size * 0.1,
            map_size * 0.1,
        ]

        scenarios.append(
            {
                "id": (f"sample_{index + 1}"),
                "scale_name": ("Sample"),
                "num_tasks": (num_tasks),
                "map_size": (map_size),
                "nodes_loc": (nodes.tolist()),
            }
        )

    return scenarios


# ============================================================
# Public API
# ============================================================

__all__ = [
    "AblationConfig",
    "ABLATION_CONFIGS",
    "summarize_ablation_results",
    "run_ablation_training",
    "run_all_ablations",
    "print_ablation_summary",
    "save_ablation_results",
    "generate_sample_scenarios",
]
