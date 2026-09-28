"""
Physical models and constants for the UAV-UGV cooperative routing system.

This module centralizes all physical parameters used by the project,
including:

- UAV / UGV speed
- UAV battery capacity
- UAV power consumption
- UGV power consumption
- Hovering power
- Recharge power
- Energy-per-distance calculations
- Travel-time calculations

Units
-----
Distance:
    meters (m), unless explicitly stated otherwise.

Speed:
    meters per second (m/s).

Power:
    watts (W).

Energy:
    joules (J).

Time:
    seconds (s).
"""

from __future__ import annotations

from dataclasses import dataclass

# ============================================================
# Core physical constants
# ============================================================

# UAV cruising speed.
UAV_SPEED_MPS: float = 10.0

# UGV travelling speed.
UGV_SPEED_MPS: float = 4.5


# ============================================================
# UAV battery
# ============================================================

# UAV maximum battery capacity.
MAX_FUEL_KJ: float = 287.7

MAX_FUEL_J: float = MAX_FUEL_KJ * 1000.0


# ============================================================
# Recharge
# ============================================================

# Charging power when UAV is docked on the UGV.
RECHARGE_POWER_W: float = 5000.0


# ============================================================
# Hovering
# ============================================================

# Approximate UAV hovering power.
#
# The original experimental implementation uses 200 W when
# the UAV reaches a rendezvous point before the UGV and must
# wait in the air.
HOVER_POWER_W: float = 200.0


# ============================================================
# Reference map size
# ============================================================

# Maximum map size used by the original experiments.
#
# Small:  16 km
# Medium: 25 km
# Large:  40 km
MAX_MAP_SIZE: float = 40.0


# ============================================================
# UAV physical model
# ============================================================


def uav_power_w(speed_mps: float) -> float:
    """
    Calculate UAV power consumption at a given speed.

    The power model used by the original implementation is:

        P(v) =
            0.0461 * v^3
            - 0.5834 * v^2
            - 1.8761 * v
            + 229.6

    Parameters
    ----------
    speed_mps:
        UAV speed in meters per second.

    Returns
    -------
    float
        UAV power consumption in watts.

    Raises
    ------
    ValueError
        If speed is negative.
    """

    if speed_mps < 0:
        raise ValueError("speed_mps must be greater than or equal to 0.")

    v = float(speed_mps)

    return 0.0461 * v**3 - 0.5834 * v**2 - 1.8761 * v + 229.6


def uav_energy_per_meter(
    speed_mps: float = UAV_SPEED_MPS,
) -> float:
    """
    Calculate UAV energy consumption per travelled meter.

    Energy per meter is:

        E_per_meter = Power / Speed

    because:

        W / (m/s)
        = J/s * s/m
        = J/m

    Parameters
    ----------
    speed_mps:
        UAV speed in meters per second.

    Returns
    -------
    float
        Energy consumption in joules per meter.
    """

    if speed_mps <= 0:
        raise ValueError("speed_mps must be greater than 0.")

    return uav_power_w(speed_mps) / speed_mps


# Energy consumption at the default UAV cruising speed.
UAV_ENERGY_PER_METER: float = uav_energy_per_meter(UAV_SPEED_MPS)


# Maximum theoretical UAV flight distance with a full battery.
MAX_FLIGHT_DIST_M: float = MAX_FUEL_J / UAV_ENERGY_PER_METER

MAX_FLIGHT_DIST_KM: float = MAX_FLIGHT_DIST_M / 1000.0


# ============================================================
# UGV physical model
# ============================================================


def ugv_power_w(
    speed_mps: float = UGV_SPEED_MPS,
) -> float:
    """
    Calculate UGV power consumption at a given speed.

    The model used by the original implementation is:

        P(v) = 464.8 * v + 356.3

    Parameters
    ----------
    speed_mps:
        UGV speed in meters per second.

    Returns
    -------
    float
        UGV power consumption in watts.
    """

    if speed_mps < 0:
        raise ValueError("speed_mps must be greater than or equal to 0.")

    return 464.8 * float(speed_mps) + 356.3


def ugv_energy_per_meter(
    speed_mps: float = UGV_SPEED_MPS,
) -> float:
    """
    Calculate UGV energy consumption per travelled meter.

    Parameters
    ----------
    speed_mps:
        UGV speed in meters per second.

    Returns
    -------
    float
        Energy consumption in joules per meter.
    """

    if speed_mps <= 0:
        raise ValueError("speed_mps must be greater than 0.")

    return ugv_power_w(speed_mps) / speed_mps


UGV_ENERGY_PER_METER: float = ugv_energy_per_meter(UGV_SPEED_MPS)


# ============================================================
# Distance / time helpers
# ============================================================


def travel_time_s(
    distance_m: float,
    speed_mps: float,
) -> float:
    """
    Calculate travelling time.

    Parameters
    ----------
    distance_m:
        Travel distance in meters.

    speed_mps:
        Vehicle speed in meters per second.

    Returns
    -------
    float
        Travel time in seconds.
    """

    if distance_m < 0:
        raise ValueError("distance_m must be greater than or equal to 0.")

    if speed_mps <= 0:
        raise ValueError("speed_mps must be greater than 0.")

    return float(distance_m) / float(speed_mps)


def uav_travel_time_s(
    distance_m: float,
) -> float:
    """
    Calculate UAV travel time using the default cruising speed.
    """

    return travel_time_s(
        distance_m,
        UAV_SPEED_MPS,
    )


def ugv_travel_time_s(
    distance_m: float,
) -> float:
    """
    Calculate UGV travel time using the default travelling speed.
    """

    return travel_time_s(
        distance_m,
        UGV_SPEED_MPS,
    )


# ============================================================
# UAV energy helpers
# ============================================================


def uav_flight_energy_j(
    distance_m: float,
    speed_mps: float = UAV_SPEED_MPS,
) -> float:
    """
    Calculate energy consumed by the UAV for a flight.

    Parameters
    ----------
    distance_m:
        Flight distance in meters.

    speed_mps:
        UAV speed.

    Returns
    -------
    float
        Required energy in joules.
    """

    if distance_m < 0:
        raise ValueError("distance_m must be greater than or equal to 0.")

    return float(distance_m) * uav_energy_per_meter(speed_mps)


def hovering_energy_j(
    hover_time_s: float,
) -> float:
    """
    Calculate UAV hovering energy.

    Parameters
    ----------
    hover_time_s:
        Hovering duration in seconds.

    Returns
    -------
    float
        Hovering energy in joules.
    """

    if hover_time_s < 0:
        raise ValueError("hover_time_s must be greater than or equal to 0.")

    return float(hover_time_s) * HOVER_POWER_W


def required_flight_energy_j(
    outbound_distance_m: float,
    return_distance_m: float = 0.0,
    hover_time_s: float = 0.0,
    safety_factor: float = 1.0,
) -> float:
    """
    Calculate total UAV energy required for a mission.

    The calculation includes:

        outbound flight
        + return flight
        + optional hovering
        + optional safety margin

    Parameters
    ----------
    outbound_distance_m:
        Distance from the UAV to the destination.

    return_distance_m:
        Required return distance.

    hover_time_s:
        Optional hovering time.

    safety_factor:
        Energy safety multiplier.

        Example:

            1.05 = 5% safety margin.

    Returns
    -------
    float
        Total required energy in joules.
    """

    if outbound_distance_m < 0:
        raise ValueError("outbound_distance_m must be >= 0.")

    if return_distance_m < 0:
        raise ValueError("return_distance_m must be >= 0.")

    if hover_time_s < 0:
        raise ValueError("hover_time_s must be >= 0.")

    if safety_factor < 1.0:
        raise ValueError("safety_factor must be >= 1.0.")

    flight_distance_m = outbound_distance_m + return_distance_m

    flight_energy = uav_flight_energy_j(flight_distance_m)

    hover_energy = hovering_energy_j(hover_time_s)

    return (flight_energy + hover_energy) * safety_factor


# ============================================================
# Recharge helpers
# ============================================================


def recharge_time_s(
    current_energy_j: float,
    max_energy_j: float = MAX_FUEL_J,
    recharge_power_w: float = RECHARGE_POWER_W,
) -> float:
    """
    Calculate the time required to recharge a UAV.

    Parameters
    ----------
    current_energy_j:
        Current UAV battery energy.

    max_energy_j:
        Maximum battery capacity.

    recharge_power_w:
        Charging power.

    Returns
    -------
    float
        Required charging time in seconds.
    """

    if max_energy_j <= 0:
        raise ValueError("max_energy_j must be greater than 0.")

    if recharge_power_w <= 0:
        raise ValueError("recharge_power_w must be greater than 0.")

    current_energy_j = max(
        0.0,
        min(
            float(current_energy_j),
            max_energy_j,
        ),
    )

    energy_deficit_j = max_energy_j - current_energy_j

    return energy_deficit_j / recharge_power_w


# ============================================================
# Battery helpers
# ============================================================


def fuel_ratio(
    energy_j: float,
    max_energy_j: float = MAX_FUEL_J,
) -> float:
    """
    Convert battery energy to a normalized [0, 1] ratio.
    """

    if max_energy_j <= 0:
        raise ValueError("max_energy_j must be greater than 0.")

    return max(
        0.0,
        min(
            1.0,
            float(energy_j) / max_energy_j,
        ),
    )


def can_complete_flight(
    available_energy_j: float,
    outbound_distance_m: float,
    return_distance_m: float = 0.0,
    hover_time_s: float = 0.0,
    safety_factor: float = 1.05,
) -> bool:
    """
    Check whether the UAV has enough energy to safely
    complete a flight.

    This helper can be used by dynamic energy-aware masking.
    """

    required_energy = required_flight_energy_j(
        outbound_distance_m=outbound_distance_m,
        return_distance_m=return_distance_m,
        hover_time_s=hover_time_s,
        safety_factor=safety_factor,
    )

    return bool(available_energy_j >= required_energy)


# ============================================================
# Vehicle specifications
# ============================================================


@dataclass(frozen=True)
class UAV:
    """
    UAV physical specification.
    """

    speed_mps: float = UAV_SPEED_MPS
    max_energy_j: float = MAX_FUEL_J

    @property
    def power_w(self) -> float:
        """Power consumption at cruising speed."""

        return uav_power_w(self.speed_mps)

    @property
    def energy_per_meter_j(self) -> float:
        """Energy consumption per meter."""

        return uav_energy_per_meter(self.speed_mps)

    @property
    def max_distance_m(self) -> float:
        """Maximum theoretical flight distance."""

        return self.max_energy_j / self.energy_per_meter_j

    @property
    def max_distance_km(self) -> float:
        """Maximum theoretical flight distance in km."""

        return self.max_distance_m / 1000.0


@dataclass(frozen=True)
class UGV:
    """
    UGV physical specification.
    """

    speed_mps: float = UGV_SPEED_MPS

    @property
    def power_w(self) -> float:
        """UGV power consumption."""

        return ugv_power_w(self.speed_mps)

    @property
    def energy_per_meter_j(self) -> float:
        """UGV energy consumption per meter."""

        return ugv_energy_per_meter(self.speed_mps)


# ============================================================
# Default vehicle instances
# ============================================================

DEFAULT_UAV = UAV()

DEFAULT_UGV = UGV()


# ============================================================
# Public API
# ============================================================

__all__ = [
    # Constants
    "UAV_SPEED_MPS",
    "UGV_SPEED_MPS",
    "MAX_FUEL_KJ",
    "MAX_FUEL_J",
    "RECHARGE_POWER_W",
    "HOVER_POWER_W",
    "MAX_MAP_SIZE",
    "UAV_ENERGY_PER_METER",
    "UGV_ENERGY_PER_METER",
    "MAX_FLIGHT_DIST_M",
    "MAX_FLIGHT_DIST_KM",
    # UAV functions
    "uav_power_w",
    "uav_energy_per_meter",
    "uav_flight_energy_j",
    "hovering_energy_j",
    "required_flight_energy_j",
    "can_complete_flight",
    # UGV functions
    "ugv_power_w",
    "ugv_energy_per_meter",
    # Time / battery helpers
    "travel_time_s",
    "uav_travel_time_s",
    "ugv_travel_time_s",
    "recharge_time_s",
    "fuel_ratio",
    # Vehicle classes
    "UAV",
    "UGV",
    "DEFAULT_UAV",
    "DEFAULT_UGV",
]
