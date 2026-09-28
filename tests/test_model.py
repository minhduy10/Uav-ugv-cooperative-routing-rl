"""
Tests for the Transformer UAV-UGV policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# ============================================================
# Allow imports from src/
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from uav_ugv_rl.model import (  # noqa: E402
    MASKED_LOGIT_VALUE,
    NODE_FEATURE_DIM,
    NUM_OPERATION_MODES,
    ScalableTransformerPolicy,
    TransformerPolicy,
    build_policy,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def small_model() -> TransformerPolicy:
    """
    Smaller model used to keep unit tests fast.
    """

    model = TransformerPolicy(
        in_nodes=7,
        embed_dim=48,
        nhead=6,
        num_layers=1,
        dropout=0.0,
    )

    model.eval()

    return model


@pytest.fixture
def single_observation() -> dict[str, torch.Tensor]:
    """
    One observation with five nodes.
    """

    return {
        "nodes_features": torch.rand(
            5,
            7,
        ),
        "uav_state": torch.rand(3),
        "ugv_state": torch.rand(2),
    }


# ============================================================
# Constants
# ============================================================


def test_model_constants() -> None:
    """
    Model dimensions must match environment.py.
    """

    assert NODE_FEATURE_DIM == 7
    assert NUM_OPERATION_MODES == 2


# ============================================================
# Construction
# ============================================================


def test_transformer_policy_creation() -> None:
    """
    Policy should initialize successfully with valid dimensions.
    """

    model = TransformerPolicy(
        embed_dim=48,
        nhead=6,
        num_layers=1,
    )

    assert isinstance(
        model,
        TransformerPolicy,
    )

    assert model.embed_dim == 48
    assert model.nhead == 6


def test_invalid_attention_dimensions() -> None:
    """
    embed_dim must be divisible by nhead.
    """

    with pytest.raises(ValueError):
        TransformerPolicy(
            embed_dim=50,
            nhead=6,
        )


# ============================================================
# Single-observation forward
# ============================================================


def test_forward_shapes(
    small_model: TransformerPolicy,
    single_observation: dict[
        str,
        torch.Tensor,
    ],
) -> None:
    """
    forward() should create three categorical distributions
    and one value estimate.
    """

    num_nodes = 5

    uav_mask = torch.ones(
        num_nodes,
        dtype=torch.bool,
    )

    ugv_mask = torch.ones(
        num_nodes,
        dtype=torch.bool,
    )

    mode_mask = torch.ones(
        2,
        dtype=torch.bool,
    )

    (
        uav_distribution,
        ugv_distribution,
        mode_distribution,
        value,
    ) = small_model(
        single_observation,
        uav_mask,
        ugv_mask,
        mode_mask,
        num_tasks=4,
    )

    assert uav_distribution.logits.shape == (1, 5)

    assert ugv_distribution.logits.shape == (1, 5)

    assert mode_distribution.logits.shape == (1, 2)

    assert value.shape == (
        1,
        1,
    )


# ============================================================
# Batched forward
# ============================================================


def test_batched_forward_shapes(
    small_model: TransformerPolicy,
) -> None:
    """
    PPO/BC forward should support batched trajectories.
    """

    batch_size = 4
    num_nodes = 6

    observations = {
        "nodes_features": torch.rand(
            batch_size,
            num_nodes,
            7,
        ),
        "uav_state": torch.rand(
            batch_size,
            3,
        ),
        "ugv_state": torch.rand(
            batch_size,
            2,
        ),
    }

    uav_masks = torch.ones(
        batch_size,
        num_nodes,
        dtype=torch.bool,
    )

    ugv_masks = torch.ones(
        batch_size,
        num_nodes,
        dtype=torch.bool,
    )

    mode_masks = torch.ones(
        batch_size,
        2,
        dtype=torch.bool,
    )

    (
        uav_logits,
        ugv_logits,
        mode_logits,
        values,
    ) = small_model.batched_forward(
        observations,
        uav_masks,
        ugv_masks,
        mode_masks,
        num_tasks=num_nodes - 1,
    )

    assert uav_logits.shape == (
        batch_size,
        num_nodes,
    )

    assert ugv_logits.shape == (
        batch_size,
        num_nodes,
    )

    assert mode_logits.shape == (
        batch_size,
        2,
    )

    assert values.shape == (batch_size,)


# ============================================================
# Action masking
# ============================================================


def test_uav_action_mask(
    small_model: TransformerPolicy,
) -> None:
    """
    Invalid UAV actions should receive the masked logit value.
    """

    batch_size = 1
    num_nodes = 4

    observations = {
        "nodes_features": torch.rand(
            batch_size,
            num_nodes,
            7,
        ),
        "uav_state": torch.rand(
            batch_size,
            3,
        ),
        "ugv_state": torch.rand(
            batch_size,
            2,
        ),
    }

    uav_mask = torch.tensor(
        [
            [
                True,
                False,
                True,
                False,
            ]
        ],
        dtype=torch.bool,
    )

    ugv_mask = torch.ones(
        batch_size,
        num_nodes,
        dtype=torch.bool,
    )

    mode_mask = torch.ones(
        batch_size,
        2,
        dtype=torch.bool,
    )

    (
        uav_logits,
        _,
        _,
        _,
    ) = small_model.batched_forward(
        observations,
        uav_mask,
        ugv_mask,
        mode_mask,
        num_tasks=3,
    )

    assert uav_logits[
        0,
        1,
    ].item() == pytest.approx(MASKED_LOGIT_VALUE)

    assert uav_logits[
        0,
        3,
    ].item() == pytest.approx(MASKED_LOGIT_VALUE)


def test_mode_action_mask(
    small_model: TransformerPolicy,
) -> None:
    """
    Invalid operation mode should be masked.
    """

    observations = {
        "nodes_features": torch.rand(
            1,
            4,
            7,
        ),
        "uav_state": torch.rand(
            1,
            3,
        ),
        "ugv_state": torch.rand(
            1,
            2,
        ),
    }

    node_mask = torch.ones(
        1,
        4,
        dtype=torch.bool,
    )

    mode_mask = torch.tensor(
        [
            [
                True,
                False,
            ]
        ],
        dtype=torch.bool,
    )

    (
        _,
        _,
        mode_logits,
        _,
    ) = small_model.batched_forward(
        observations,
        node_mask,
        node_mask,
        mode_mask,
        num_tasks=3,
    )

    assert mode_logits[
        0,
        1,
    ].item() == pytest.approx(MASKED_LOGIT_VALUE)


# ============================================================
# Sampling
# ============================================================


def test_policy_can_sample_action(
    small_model: TransformerPolicy,
    single_observation: dict[
        str,
        torch.Tensor,
    ],
) -> None:
    """
    All three policy distributions should support sampling.
    """

    node_mask = torch.ones(
        5,
        dtype=torch.bool,
    )

    mode_mask = torch.ones(
        2,
        dtype=torch.bool,
    )

    (
        uav_distribution,
        ugv_distribution,
        mode_distribution,
        _,
    ) = small_model(
        single_observation,
        node_mask,
        node_mask,
        mode_mask,
        num_tasks=4,
    )

    uav_action = uav_distribution.sample()

    ugv_action = ugv_distribution.sample()

    mode_action = mode_distribution.sample()

    assert 0 <= int(uav_action.item()) < 5

    assert 0 <= int(ugv_action.item()) < 5

    assert 0 <= int(mode_action.item()) < 2


# ============================================================
# Critic
# ============================================================


def test_value_function(
    small_model: TransformerPolicy,
    single_observation: dict[
        str,
        torch.Tensor,
    ],
) -> None:
    """
    value() should return one value per observation.
    """

    value = small_model.value(
        single_observation,
        num_tasks=4,
    )

    assert value.shape == (
        1,
        1,
    )

    assert torch.isfinite(value).all()


# ============================================================
# Factory
# ============================================================


def test_build_small_policy() -> None:
    """
    Small / Medium factory configuration.
    """

    model = build_policy(is_large=False)

    assert isinstance(
        model,
        ScalableTransformerPolicy,
    )

    assert model.embed_dim == 192
    assert model.nhead == 6
    assert model.num_layers == 3


def test_build_large_policy() -> None:
    """
    Large factory configuration.
    """

    model = build_policy(is_large=True)

    assert isinstance(
        model,
        ScalableTransformerPolicy,
    )

    assert model.embed_dim == 256
    assert model.nhead == 8
    assert model.num_layers == 4


# ============================================================
# Parameter count
# ============================================================


def test_parameter_count(
    small_model: TransformerPolicy,
) -> None:
    """
    Policy should contain trainable parameters.
    """

    assert small_model.num_parameters > 0


# ============================================================
# Invalid node-feature shape
# ============================================================


def test_invalid_node_feature_dimension(
    small_model: TransformerPolicy,
) -> None:
    """
    Model must reject observations that do not use seven
    node features.
    """

    observations = {
        "nodes_features": torch.rand(
            1,
            5,
            5,
        ),
        "uav_state": torch.rand(
            1,
            3,
        ),
        "ugv_state": torch.rand(
            1,
            2,
        ),
    }

    node_mask = torch.ones(
        1,
        5,
        dtype=torch.bool,
    )

    mode_mask = torch.ones(
        1,
        2,
        dtype=torch.bool,
    )

    with pytest.raises(ValueError):
        small_model.batched_forward(
            observations,
            node_mask,
            node_mask,
            mode_mask,
            num_tasks=4,
        )
