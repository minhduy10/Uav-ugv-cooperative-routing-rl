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

__version__ = "0.1.0"

__all__ = [
    "UAVUGVEnv",
    "UAVUGVCoopEnv",
    "UAVUGVCoopEnv_Benchmark",
    "UAVUGVCoopEnv_Flex",
    "TransformerPolicy",
    "ScalableTransformerPolicy",
    "build_policy",
    "SuperiorityAgent",
    "UAVUGVAgent",
    "TrainingConfig",
    "TrainingResult",
    "train_scenario",
    "train_scenarios",
    "evaluate_agent",
]
