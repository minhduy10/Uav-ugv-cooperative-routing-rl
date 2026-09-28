"""
Tests for UAV-UGV Gymnasium environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ============================================================
# Allow tests to import from src/ without installing package
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from uav_ugv_rl.environment import (  # noqa: E402
    MODE_MOVE,
    MODE_WAIT,
    UAVUGVEnv,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_nodes() -> np.ndarray:
    """
    Small deterministic scenario:

        depot = (0, 0)
        task1 = (1, 0)
        task2 = (0, 1)
    """

    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def env() -> UAVUGVEnv:
    """
    Small UAV-UGV environment.
    """

    environment = UAVUGVEnv(
        num_tasks=2,
        map_size=2.0,
    )

    yield environment

    environment.close()


# ============================================================
# Initialization
# ============================================================


def test_environment_initialization(
    env: UAVUGVEnv,
) -> None:
    """
    Environment should expose the expected number of nodes
    and action dimensions.
    """

    assert env.num_tasks == 2
    assert env.num_nodes == 3

    assert env.action_space.nvec.tolist() == [
        3,
        3,
        2,
    ]


# ============================================================
# Reset
# ============================================================


def test_reset_with_fixed_nodes(
    env: UAVUGVEnv,
    simple_nodes: np.ndarray,
) -> None:
    """
    reset() should accept deterministic fixed nodes.
    """

    observation, info = env.reset(
        options={
            "fixed_nodes": simple_nodes,
            "map_size": 2.0,
        }
    )

    assert env.num_nodes == 3
    assert env.num_tasks == 2

    np.testing.assert_allclose(
        env.nodes_loc,
        simple_nodes,
    )

    assert isinstance(
        info,
        dict,
    )

    assert observation["uav_state"].shape == (3,)

    assert observation["ugv_state"].shape == (2,)

    assert observation["nodes_features"].shape == (
        3,
        7,
    )


def test_reset_starts_at_depot(
    env: UAVUGVEnv,
    simple_nodes: np.ndarray,
) -> None:
    """
    UAV and UGV should both start at node 0.
    """

    env.reset(
        options={
            "fixed_nodes": simple_nodes,
            "map_size": 2.0,
        }
    )

    np.testing.assert_allclose(
        env.uav_pos,
        simple_nodes[0],
    )

    np.testing.assert_allclose(
        env.ugv_pos,
        simple_nodes[0],
    )

    assert env.uav_curr_idx == 0
    assert env.ugv_curr_idx == 0

    assert env.visited[0] == 1

    assert np.sum(env.visited) == 1


# ============================================================
# Observation
# ============================================================


def test_observation_node_features(
    env: UAVUGVEnv,
    simple_nodes: np.ndarray,
) -> None:
    """
    Node features should contain seven values:

        x, y,
        visited,
        relative UAV x/y,
        relative UGV x/y
    """

    observation, _ = env.reset(
        options={
            "fixed_nodes": simple_nodes,
            "map_size": 2.0,
        }
    )

    features = observation["nodes_features"]

    assert features.shape == (
        3,
        7,
    )

    # Column index 2 contains visited flag.
    assert features[0, 2] == pytest.approx(1.0)

    assert features[1, 2] == pytest.approx(0.0)

    assert features[2, 2] == pytest.approx(0.0)


# ============================================================
# Energy-aware action masking
# ============================================================


def test_energy_feasible_target(
    env: UAVUGVEnv,
) -> None:
    """
    A nearby target should be feasible with a full battery.
    """

    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    env.reset(
        options={
            "fixed_nodes": nodes,
            "map_size": 2.0,
        }
    )

    assert env.is_energy_feasible(1)


def test_energy_infeasible_target_is_masked(
    env: UAVUGVEnv,
) -> None:
    """
    A very distant task should be removed by the UAV energy mask.

    A nearby task is included to ensure the environment does not
    enter its no-safe-target fallback mode.
    """

    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [20.0, 0.0],
        ],
        dtype=np.float32,
    )

    env.reset(
        options={
            "fixed_nodes": nodes,
            "map_size": 25.0,
        }
    )

    assert env.is_energy_feasible(1)

    assert not env.is_energy_feasible(2)

    (
        uav_mask,
        _,
        _,
    ) = env.get_action_masks()

    assert bool(uav_mask[1])

    assert not bool(uav_mask[2])


# ============================================================
# Step
# ============================================================


def test_environment_step(
    env: UAVUGVEnv,
    simple_nodes: np.ndarray,
) -> None:
    """
    A valid move should advance simulation time.
    """

    env.reset(
        options={
            "fixed_nodes": simple_nodes,
            "map_size": 2.0,
        }
    )

    action = [
        1,
        1,
        MODE_MOVE,
    ]

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    assert observation["nodes_features"].shape == (
        3,
        7,
    )

    assert isinstance(
        reward,
        float,
    )

    assert isinstance(
        terminated,
        bool,
    )

    assert isinstance(
        truncated,
        bool,
    )

    assert isinstance(
        info,
        dict,
    )

    assert env.total_time_s > 0.0

    # Task 1 should have been visited.
    assert env.visited[1] == 1


# ============================================================
# Modes
# ============================================================


def test_operation_modes() -> None:
    """
    Environment operation-mode convention must remain stable.
    """

    assert MODE_MOVE == 0
    assert MODE_WAIT == 1


# ============================================================
# Expert mode
# ============================================================


def test_set_expert_mode(
    env: UAVUGVEnv,
) -> None:
    """
    Expert mode should be explicitly switchable.
    """

    env.set_expert_mode(True)

    assert env.expert_mode is True

    env.set_expert_mode(False)

    assert env.expert_mode is False
