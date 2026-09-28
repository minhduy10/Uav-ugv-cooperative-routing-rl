"""
UAV-UGV Reinforcement Learning Package.
"""

from .environment import (
    UAVUGVEnv,
    UAVUGVCoopEnv,
    UAVUGVCoopEnv_Benchmark,
    UAVUGVCoopEnv_Flex,
)

from .model import (
    TransformerPolicy,
    ScalableTransformerPolicy,
    build_policy,
)

from .agent import (
    SuperiorityAgent,
    UAVUGVAgent,
)

from .training import (
    TrainingConfig,
    TrainingResult,
    train_scenario,
    train_scenarios,
    evaluate_agent,
)

from .ablation import (
    AblationConfig,
    ABLATION_CONFIGS,
    run_ablation_training,
    run_all_ablations,
    print_ablation_summary,
)

__version__ = "0.1.0"

__all__ = [
    # Environment
    "UAVUGVEnv",
    "UAVUGVCoopEnv",
    "UAVUGVCoopEnv_Benchmark",
    "UAVUGVCoopEnv_Flex",
    # Model
    "TransformerPolicy",
    "ScalableTransformerPolicy",
    "build_policy",
    # Agent
    "SuperiorityAgent",
    "UAVUGVAgent",
    # Training
    "TrainingConfig",
    "TrainingResult",
    "train_scenario",
    "train_scenarios",
    "evaluate_agent",
    # Ablation
    "AblationConfig",
    "ABLATION_CONFIGS",
    "run_ablation_training",
    "run_all_ablations",
    "print_ablation_summary",
]
