"""
Tests for UAV and UGV physical / energy models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


from uav_ugv_rl.physics import (  # noqa: E402
    HOVER_POWER_W,
    MAX_FLIGHT_DIST_KM,
    MAX_FLIGHT_DIST_M,
    MAX_FUEL_J,
    RECHARGE_POWER_W,
    UAV,
    UAV_ENERGY_PER_METER,
    UAV_SPEED_MPS,
    UGV,
    UGV_SPEED_MPS,
    can_complete_flight,
    fuel_ratio,
    hovering_energy_j,
    recharge_time_s,
    required_flight_energy_j,
    travel_time_s,
    uav_energy_per_meter,
    uav_flight_energy_j,
    uav_power_w,
    ugv_energy_per_meter,
    ugv_power_w,
)

# ============================================================
# UAV power model
# ============================================================


def test_uav_power_is_positive() -> None:
    """
    UAV power consumption at operating speed must be positive.
    """

    power = uav_power_w(UAV_SPEED_MPS)

    assert power > 0.0


def test_uav_energy_per_meter_consistency() -> None:
    """
    Energy per metre should equal power divided by velocity.
    """

    expected = uav_power_w(UAV_SPEED_MPS) / UAV_SPEED_MPS

    assert UAV_ENERGY_PER_METER == pytest.approx(expected)

    assert uav_energy_per_meter(UAV_SPEED_MPS) == pytest.approx(expected)


# ============================================================
# UAV flight energy
# ============================================================


def test_uav_flight_energy() -> None:
    """
    Flight energy must scale linearly with distance.
    """

    distance_m = 1000.0

    expected = distance_m * UAV_ENERGY_PER_METER

    energy = uav_flight_energy_j(distance_m)

    assert energy == pytest.approx(expected)


def test_required_flight_energy() -> None:
    """
    Required energy should include outbound + return distances.
    """

    outbound_m = 1000.0
    return_m = 500.0

    expected = (outbound_m + return_m) * UAV_ENERGY_PER_METER

    result = required_flight_energy_j(
        outbound_distance_m=(outbound_m),
        return_distance_m=(return_m),
    )

    assert result == pytest.approx(expected)


def test_required_energy_safety_factor() -> None:
    """
    Safety factor should increase required energy.
    """

    without_margin = required_flight_energy_j(
        outbound_distance_m=1000.0,
        safety_factor=1.0,
    )

    with_margin = required_flight_energy_j(
        outbound_distance_m=1000.0,
        safety_factor=1.05,
    )

    assert with_margin == pytest.approx(without_margin * 1.05)


# ============================================================
# Hover energy
# ============================================================


def test_hovering_energy() -> None:
    """
    Hover energy = hover power * hover time.
    """

    hover_time = 10.0

    expected = HOVER_POWER_W * hover_time

    assert hovering_energy_j(hover_time) == pytest.approx(expected)


# ============================================================
# Battery range
# ============================================================


def test_max_flight_distance_consistency() -> None:
    """
    Maximum distance should equal battery energy divided by
    energy consumption per metre.
    """

    expected_m = MAX_FUEL_J / UAV_ENERGY_PER_METER

    assert MAX_FLIGHT_DIST_M == pytest.approx(expected_m)

    assert MAX_FLIGHT_DIST_KM == pytest.approx(expected_m / 1000.0)


def test_default_uav_range() -> None:
    """
    UAV dataclass should use the same physical constants.
    """

    uav = UAV()

    assert uav.speed_mps == pytest.approx(UAV_SPEED_MPS)

    assert uav.max_energy_j == pytest.approx(MAX_FUEL_J)

    assert uav.max_distance_m == pytest.approx(MAX_FLIGHT_DIST_M)


# ============================================================
# Flight feasibility
# ============================================================


def test_can_complete_short_flight() -> None:
    """
    Full UAV battery should easily support a short sortie.
    """

    feasible = can_complete_flight(
        available_energy_j=MAX_FUEL_J,
        outbound_distance_m=500.0,
        return_distance_m=500.0,
    )

    assert feasible is True


def test_cannot_complete_oversized_flight() -> None:
    """
    Flight beyond battery range should be rejected.
    """

    feasible = can_complete_flight(
        available_energy_j=MAX_FUEL_J,
        outbound_distance_m=(MAX_FLIGHT_DIST_M * 2.0),
    )

    assert feasible is False


# ============================================================
# Recharge
# ============================================================


def test_recharge_time_from_empty() -> None:
    """
    Empty battery recharge time:

        battery capacity / recharge power
    """

    expected = MAX_FUEL_J / RECHARGE_POWER_W

    result = recharge_time_s(0.0)

    assert result == pytest.approx(expected)


def test_recharge_time_when_full() -> None:
    """
    Full battery should require no charging time.
    """

    assert recharge_time_s(MAX_FUEL_J) == pytest.approx(0.0)


# ============================================================
# Fuel ratio
# ============================================================


def test_fuel_ratio() -> None:
    """
    Fuel ratio must be normalized to [0, 1].
    """

    assert fuel_ratio(MAX_FUEL_J) == pytest.approx(1.0)

    assert fuel_ratio(MAX_FUEL_J / 2.0) == pytest.approx(0.5)

    assert fuel_ratio(0.0) == pytest.approx(0.0)


def test_fuel_ratio_is_clamped() -> None:
    """
    Invalid energy values should still produce a valid ratio.
    """

    assert fuel_ratio(-100.0) == pytest.approx(0.0)

    assert fuel_ratio(MAX_FUEL_J * 2.0) == pytest.approx(1.0)


# ============================================================
# Travel time
# ============================================================


def test_travel_time() -> None:
    """
    Travel time = distance / velocity.
    """

    assert travel_time_s(
        distance_m=100.0,
        speed_mps=10.0,
    ) == pytest.approx(10.0)


# ============================================================
# UGV model
# ============================================================


def test_ugv_power_model() -> None:
    """
    UGV power model should produce positive power.
    """

    power = ugv_power_w(UGV_SPEED_MPS)

    assert power > 0.0


def test_ugv_energy_per_meter() -> None:
    """
    UGV energy per metre must equal power / velocity.
    """

    expected = ugv_power_w(UGV_SPEED_MPS) / UGV_SPEED_MPS

    assert ugv_energy_per_meter(UGV_SPEED_MPS) == pytest.approx(expected)


def test_default_ugv() -> None:
    """
    Default UGV should use the configured speed.
    """

    ugv = UGV()

    assert ugv.speed_mps == pytest.approx(UGV_SPEED_MPS)

    assert ugv.power_w > 0.0


# ============================================================
# Invalid inputs
# ============================================================


def test_negative_distance_raises_error() -> None:
    """
    Negative flight distance must be rejected.
    """

    with pytest.raises(ValueError):
        uav_flight_energy_j(-1.0)


def test_zero_speed_raises_error() -> None:
    """
    Travel time cannot be calculated with zero speed.
    """

    with pytest.raises(ValueError):
        travel_time_s(
            100.0,
            0.0,
        )
