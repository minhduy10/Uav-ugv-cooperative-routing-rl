"""
Heuristic and OR-Tools-based planning utilities for UAV-UGV routing.

This module provides the expert-planning side of the project.

Main responsibilities
---------------------
1. Build UGV rendezvous stops using greedy set cover.
2. Order UGV stops using nearest-neighbour TSP + 2-opt.
3. Allocate UAV tasks to UGV route segments.
4. Optimize each UAV sortie using:
       - Cheapest insertion + 2-opt, or
       - OR-Tools CP-SAT when available.
5. Build a cooperative UAV-UGV plan.
6. Convert the expert plan into RL actions for Behavioral Cloning.

Coordinate units
----------------
Node coordinates are expressed in kilometres (km).

Vehicle speeds:
    meters per second (m/s)

Energy:
    joules (J) / kilojoules (kJ)

Time:
    seconds (s)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .physics import (
    MAX_FUEL_J,
    RECHARGE_POWER_W,
    UAV,
    UGV,
    UAV_SPEED_MPS,
    UGV_SPEED_MPS,
)

# ============================================================
# Type aliases
# ============================================================

Point = tuple[float, float]
Action = list[int]


# ============================================================
# Planning results
# ============================================================


@dataclass
class SegmentResult:
    """
    Result of one cooperative UAV-UGV route segment.

    Attributes
    ----------
    p:
        Segment index.

    origin:
        Starting UGV rendezvous point.

    dest:
        Ending UGV rendezvous point.

    task_indices:
        Indices of tasks assigned to this segment.

        Indices refer to the task list passed to cooperative_plan(),
        not to the full node array containing the depot.

    uav_path:
        Ordered UAV path:

            origin -> tasks -> destination

    uav_energy_kJ:
        UAV energy consumed during the segment.

        A value of 0 indicates that the UGV carries the UAV
        through the route because the sortie is not feasible
        within the UAV battery capacity.
    """

    p: int
    origin: Point
    dest: Point
    task_indices: list[int]
    uav_path: list[Point]
    uav_energy_kJ: float


@dataclass
class PlanResult:
    """
    Complete cooperative UAV-UGV plan.
    """

    segments: list[SegmentResult]
    makespan_s: float


# ============================================================
# Geometry helpers
# ============================================================


def _as_point(point: Sequence[float]) -> Point:
    """
    Convert a 2-D coordinate to a Point tuple.
    """

    if len(point) != 2:
        raise ValueError("A point must contain exactly two coordinates.")

    return (
        float(point[0]),
        float(point[1]),
    )


def dist_km(
    a: Point,
    b: Point,
) -> float:
    """
    Euclidean distance between two points in kilometres.
    """

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def path_length_km(
    path: Sequence[Point],
) -> float:
    """
    Calculate total length of an open path.
    """

    if len(path) < 2:
        return 0.0

    return sum(
        dist_km(
            path[i],
            path[i + 1],
        )
        for i in range(len(path) - 1)
    )


# ============================================================
# TSP helpers
# ============================================================


def tsp_nearest_neighbor(
    points: Sequence[Point],
    start_idx: int = 0,
) -> list[int]:
    """
    Build a TSP ordering using the nearest-neighbour heuristic.

    The returned route does not repeat the starting node at the end.
    """

    n = len(points)

    if n == 0:
        return []

    if not 0 <= start_idx < n:
        raise ValueError(f"start_idx must be between 0 and {n - 1}.")

    unvisited = set(range(n))
    unvisited.remove(start_idx)

    order = [start_idx]
    current = start_idx

    while unvisited:

        next_idx = min(
            unvisited,
            key=lambda idx: dist_km(
                points[current],
                points[idx],
            ),
        )

        unvisited.remove(next_idx)
        order.append(next_idx)

        current = next_idx

    return order


def cycle_length_km(
    points: Sequence[Point],
    order: Sequence[int],
) -> float:
    """
    Calculate length of a closed TSP cycle.
    """

    if len(order) < 2:
        return 0.0

    total = sum(
        dist_km(
            points[order[i]],
            points[order[i + 1]],
        )
        for i in range(len(order) - 1)
    )

    # Close the cycle.
    total += dist_km(
        points[order[-1]],
        points[order[0]],
    )

    return total


def two_opt_cycle(
    points: Sequence[Point],
    order: Sequence[int],
    max_iter: int = 2000,
) -> list[int]:
    """
    Improve a TSP cycle using 2-opt local search.

    The first node stays fixed so that the depot remains the
    beginning of the route.
    """

    best = list(order)

    if len(best) <= 3:
        return best

    best_length = cycle_length_km(
        points,
        best,
    )

    n = len(best)

    improved = True
    iteration = 0

    while improved and iteration < max_iter:

        improved = False
        iteration += 1

        for i in range(1, n - 2):

            for k in range(i + 1, n - 1):

                candidate = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]

                candidate_length = cycle_length_km(
                    points,
                    candidate,
                )

                if candidate_length + 1e-9 < best_length:

                    best = candidate
                    best_length = candidate_length

                    improved = True
                    break

            if improved:
                break

    return best


def tsp_cycle(
    points: Sequence[Point],
    start_idx: int = 0,
    do_2opt: bool = True,
) -> list[int]:
    """
    Build a TSP route with nearest neighbour and optional 2-opt.
    """

    order = tsp_nearest_neighbor(
        points,
        start_idx=start_idx,
    )

    if do_2opt:
        order = two_opt_cycle(
            points,
            order,
        )

    return order


# ============================================================
# UGV rendezvous selection
# ============================================================


def greedy_min_set_cover(
    tasks: Sequence[Point],
    depot: Point,
    cover_radius_km: float,
) -> list[Point]:
    """
    Select UGV rendezvous stops using greedy minimum set cover.

    Each selected UGV stop covers all tasks located within
    ``cover_radius_km``.

    The depot is always included as the first stop.
    """

    if cover_radius_km <= 0:
        raise ValueError("cover_radius_km must be greater than 0.")

    if len(tasks) == 0:
        return [depot]

    uncovered = set(range(len(tasks)))

    stops: list[Point] = [depot]

    # Tasks already reachable from the depot.
    depot_covered = {
        idx
        for idx in uncovered
        if dist_km(
            depot,
            tasks[idx],
        )
        <= cover_radius_km
    }

    uncovered -= depot_covered

    candidates = list(tasks)

    while uncovered:

        best_idx: int | None = None
        best_gain = -1

        for candidate_idx, candidate in enumerate(candidates):

            gain = sum(
                1
                for task_idx in uncovered
                if dist_km(
                    candidate,
                    tasks[task_idx],
                )
                <= cover_radius_km
            )

            if gain > best_gain:
                best_gain = gain
                best_idx = candidate_idx

        if best_idx is None or best_gain <= 0:
            raise RuntimeError("Unable to construct UGV cover stops.")

        chosen = candidates[best_idx]

        stops.append(chosen)

        covered = {
            task_idx
            for task_idx in uncovered
            if dist_km(
                chosen,
                tasks[task_idx],
            )
            <= cover_radius_km
        }

        uncovered -= covered

    return stops


# ============================================================
# Task allocation
# ============================================================


def allocate_tasks_to_segments(
    tasks: Sequence[Point],
    depot: Point,
    ordered_stops: Sequence[Point],
    cover_radius_km: float,
) -> dict[int, list[int]]:
    """
    Assign every task to a cooperative UGV segment.

    Segment indices begin at 1:

        segment 1:
            ordered_stops[0] -> ordered_stops[1]

        segment 2:
            ordered_stops[1] -> ordered_stops[2]

        ...

    Tasks are assigned to the first destination stop that can
    cover them within ``cover_radius_km``.
    """

    del depot  # Kept in the API for compatibility with notebook code.

    if len(ordered_stops) < 2:
        return {}

    segments: dict[int, list[int]] = {
        p: []
        for p in range(
            1,
            len(ordered_stops),
        )
    }

    for task_idx, task in enumerate(tasks):

        assigned_segment: int | None = None

        for p in range(
            1,
            len(ordered_stops),
        ):

            destination = ordered_stops[p]

            if (
                dist_km(
                    destination,
                    task,
                )
                <= cover_radius_km
            ):

                assigned_segment = p
                break

        # Fallback used by the original implementation.
        if assigned_segment is None:
            assigned_segment = 1

        segments[assigned_segment].append(task_idx)

    return segments


# ============================================================
# UAV path heuristic
# ============================================================


def cheapest_insertion_path(
    origin: Point,
    dest: Point,
    task_pts: Sequence[Point],
) -> list[Point]:
    """
    Build an open UAV route using cheapest insertion.

    The origin and destination remain fixed.

    Route:

        origin -> tasks -> dest
    """

    path: list[Point] = [
        origin,
        dest,
    ]

    for task in task_pts:

        best_position: int | None = None
        best_delta = float("inf")

        for k in range(len(path) - 1):

            delta = (
                dist_km(
                    path[k],
                    task,
                )
                + dist_km(
                    task,
                    path[k + 1],
                )
                - dist_km(
                    path[k],
                    path[k + 1],
                )
            )

            if delta < best_delta:

                best_delta = delta
                best_position = k + 1

        if best_position is None:
            best_position = len(path) - 1

        path.insert(
            best_position,
            task,
        )

    return path


def two_opt_path(
    path: Sequence[Point],
    max_iter: int = 2000,
) -> list[Point]:
    """
    Improve an open path using 2-opt.

    Unlike TSP cycle optimization, both the start and end point
    remain fixed.
    """

    best = list(path)

    if len(best) <= 4:
        return best

    best_length = path_length_km(best)

    n = len(best)

    improved = True
    iteration = 0

    while improved and iteration < max_iter:

        improved = False
        iteration += 1

        for i in range(1, n - 2):

            for k in range(
                i + 1,
                n - 1,
            ):

                candidate = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]

                candidate_length = path_length_km(candidate)

                if candidate_length + 1e-9 < best_length:

                    best = candidate
                    best_length = candidate_length

                    improved = True
                    break

            if improved:
                break

    return best


def solve_uav_segment_heuristic(
    origin: Point,
    dest: Point,
    tasks: Sequence[Point],
) -> list[Point]:
    """
    Solve one UAV sortie using:

        Cheapest Insertion
        -> 2-opt

    This solver has no OR-Tools dependency.
    """

    initial_path = cheapest_insertion_path(
        origin,
        dest,
        tasks,
    )

    return two_opt_path(initial_path)


# ============================================================
# Optional OR-Tools CP-SAT solver
# ============================================================


def solve_uav_segment_cpsat_optional(
    origin: Point,
    dest: Point,
    tasks: Sequence[Point],
    uav: UAV,
    time_limit_s: float = 10.0,
) -> list[Point] | None:
    """
    Solve a UAV segment using OR-Tools CP-SAT.

    The function intentionally imports OR-Tools locally.

    If OR-Tools is not installed or CP-SAT cannot find a feasible
    route, ``None`` is returned and the caller can fall back to the
    heuristic solver.

    Parameters
    ----------
    origin:
        UAV starting point.

    dest:
        UAV rendezvous destination.

    tasks:
        Tasks that must be visited during the sortie.

    uav:
        UAV physical specification.

    time_limit_s:
        CP-SAT time limit.
    """

    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None

    if time_limit_s <= 0:
        raise ValueError("time_limit_s must be greater than 0.")

    nodes = [
        origin,
        *tasks,
        dest,
    ]

    n = len(nodes)

    # No intermediate tasks.
    if n == 2:
        return [
            origin,
            dest,
        ]

    # --------------------------------------------------------
    # Distance / time / energy matrices
    # --------------------------------------------------------

    distance_m = [[0] * n for _ in range(n)]

    time_s = [[0] * n for _ in range(n)]

    for i in range(n):

        for j in range(n):

            distance_m[i][j] = int(
                round(
                    dist_km(
                        nodes[i],
                        nodes[j],
                    )
                    * 1000.0
                )
            )

            time_s[i][j] = int(round(distance_m[i][j] / uav.speed_mps))

    energy_per_meter = uav.energy_per_meter_j

    energy_j = [
        [int(round(energy_per_meter * distance_m[i][j])) for j in range(n)]
        for i in range(n)
    ]

    battery_capacity_j = int(round(uav.max_energy_j))

    # --------------------------------------------------------
    # CP-SAT model
    # --------------------------------------------------------

    model = cp_model.CpModel()

    arc_variables: dict[
        tuple[int, int],
        object,
    ] = {}

    circuit_arcs = []

    for i in range(n):

        for j in range(n):

            # No self loop.
            if i == j:
                continue

            # No normal arc entering origin.
            if j == 0:
                continue

            # No normal arc leaving destination.
            if i == n - 1:
                continue

            variable = model.NewBoolVar(f"x_{i}_{j}")

            arc_variables[(i, j)] = variable

            circuit_arcs.append(
                (
                    i,
                    j,
                    variable,
                )
            )

    # Add fixed dummy edge:
    #
    # destination -> origin
    #
    # This turns the open origin->destination path
    # into a CP-SAT circuit.
    dummy = model.NewBoolVar("destination_to_origin")

    model.Add(dummy == 1)

    circuit_arcs.append(
        (
            n - 1,
            0,
            dummy,
        )
    )

    model.AddCircuit(circuit_arcs)

    # --------------------------------------------------------
    # Remaining UAV energy
    # --------------------------------------------------------

    remaining_energy = [
        model.NewIntVar(
            0,
            battery_capacity_j,
            f"fuel_{i}",
        )
        for i in range(n)
    ]

    model.Add(remaining_energy[0] == battery_capacity_j)

    for (
        i,
        j,
    ), variable in arc_variables.items():

        model.Add(
            remaining_energy[j] == remaining_energy[i] - energy_j[i][j]
        ).OnlyEnforceIf(variable)

    # --------------------------------------------------------
    # Objective: minimize flight time
    # --------------------------------------------------------

    model.Minimize(
        sum(
            time_s[i][j] * variable
            for (
                i,
                j,
            ), variable in arc_variables.items()
        )
    )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = float(time_limit_s)

    status = solver.Solve(model)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        return None

    # --------------------------------------------------------
    # Reconstruct route
    # --------------------------------------------------------

    route_indices = [0]
    current = 0

    visited_indices = {0}

    while current != n - 1:

        next_node: int | None = None

        for j in range(n):

            variable = arc_variables.get(
                (
                    current,
                    j,
                )
            )

            if variable is not None and solver.Value(variable) == 1:
                next_node = j
                break

        if next_node is None:
            return None

        if next_node in visited_indices and next_node != n - 1:
            return None

        route_indices.append(next_node)

        visited_indices.add(next_node)

        current = next_node

        # Defensive guard against malformed reconstruction.
        if len(route_indices) > n:
            return None

    return [nodes[idx] for idx in route_indices]


# ============================================================
# Cooperative UAV-UGV planner
# ============================================================


def cooperative_plan(
    tasks: Sequence[Point],
    depot: Point,
    uav: UAV,
    ugv: UGV,
    cover_radius_factor: float = 0.5,
    uav_solver: str = "heuristic",
    cpsat_limit_s: float = 10.0,
    battery_usage_limit: float = 0.99,
) -> PlanResult:
    """
    Create a cooperative UAV-UGV expert plan.

    Pipeline
    --------
    Tasks
        ↓
    Greedy UGV set cover
        ↓
    TSP + 2-opt for UGV rendezvous ordering
        ↓
    Allocate tasks to segments
        ↓
    Optimize UAV route for each segment
        ↓
    Check UAV energy feasibility
        ↓
    UAV flies OR UGV carries UAV
        ↓
    Compute total makespan

    Parameters
    ----------
    tasks:
        Task coordinates, excluding the depot.

    depot:
        Initial UAV/UGV location.

    uav:
        UAV physical parameters.

    ugv:
        UGV physical parameters.

    cover_radius_factor:
        Fraction of UAV maximum theoretical range used
        as the UGV coverage radius.

    uav_solver:
        Either:

            "heuristic"
            "cpsat"

        CP-SAT automatically falls back to the heuristic
        solver if OR-Tools is unavailable or no feasible
        route is found.

    cpsat_limit_s:
        Maximum CP-SAT solve time per segment.

    battery_usage_limit:
        Maximum fraction of UAV battery allowed for one sortie.

        The original implementation uses 0.99.
    """

    if cover_radius_factor <= 0:
        raise ValueError("cover_radius_factor must be greater than 0.")

    if not 0 < battery_usage_limit <= 1.0:
        raise ValueError("battery_usage_limit must be in (0, 1].")

    if uav_solver not in {
        "heuristic",
        "cpsat",
    }:
        raise ValueError("uav_solver must be 'heuristic' or 'cpsat'.")

    task_points = [_as_point(task) for task in tasks]

    depot = _as_point(depot)

    if len(task_points) == 0:
        return PlanResult(
            segments=[],
            makespan_s=0.0,
        )

    # --------------------------------------------------------
    # Coverage radius
    # --------------------------------------------------------

    cover_radius_km = cover_radius_factor * uav.max_distance_km

    # --------------------------------------------------------
    # Select UGV rendezvous stops
    # --------------------------------------------------------

    stops = greedy_min_set_cover(
        tasks=task_points,
        depot=depot,
        cover_radius_km=cover_radius_km,
    )

    # --------------------------------------------------------
    # Optimize UGV stop ordering
    # --------------------------------------------------------

    stop_order = tsp_cycle(
        stops,
        start_idx=0,
        do_2opt=True,
    )

    ordered_stops = [stops[idx] for idx in stop_order]

    # Return to depot.
    ordered_stops.append(depot)

    # --------------------------------------------------------
    # Allocate UAV tasks
    # --------------------------------------------------------

    segment_tasks = allocate_tasks_to_segments(
        tasks=task_points,
        depot=depot,
        ordered_stops=ordered_stops,
        cover_radius_km=cover_radius_km,
    )

    segments: list[SegmentResult] = []

    current_time_s = 0.0

    # --------------------------------------------------------
    # Solve every segment
    # --------------------------------------------------------

    for p in range(
        1,
        len(ordered_stops),
    ):

        origin = ordered_stops[p - 1]

        destination = ordered_stops[p]

        task_indices = segment_tasks.get(
            p,
            [],
        )

        segment_task_points = [task_points[idx] for idx in task_indices]

        # Direct UGV route.
        ugv_direct_distance_km = dist_km(
            origin,
            destination,
        )

        # ----------------------------------------------------
        # UAV route solver
        # ----------------------------------------------------

        uav_path: list[Point] | None = None

        if uav_solver == "cpsat" and len(segment_task_points) > 0:

            uav_path = solve_uav_segment_cpsat_optional(
                origin=origin,
                dest=destination,
                tasks=segment_task_points,
                uav=uav,
                time_limit_s=cpsat_limit_s,
            )

        # Fall back to lightweight heuristic.
        if uav_path is None:

            uav_path = solve_uav_segment_heuristic(
                origin=origin,
                dest=destination,
                tasks=segment_task_points,
            )

        # ----------------------------------------------------
        # UAV energy requirement
        # ----------------------------------------------------

        uav_path_distance_m = path_length_km(uav_path) * 1000.0

        required_uav_energy_j = uav.energy_per_meter_j * uav_path_distance_m

        allowed_energy_j = uav.max_energy_j * battery_usage_limit

        # ----------------------------------------------------
        # Zero-fail carrier fallback
        # ----------------------------------------------------

        if required_uav_energy_j > allowed_energy_j:

            # UAV does not fly independently.
            #
            # UGV carries the UAV along the generated route.
            uav_time_s = 0.0
            uav_energy_kJ = 0.0

            ugv_time_s = uav_path_distance_m / ugv.speed_mps

            recharge_time = 0.0

        else:

            # UAV performs the sortie.
            uav_time_s = uav_path_distance_m / uav.speed_mps

            uav_energy_kJ = required_uav_energy_j / 1000.0

            ugv_time_s = ugv_direct_distance_km * 1000.0 / ugv.speed_mps

            # No recharge required after final return to depot.
            if p == len(ordered_stops) - 1:
                recharge_time = 0.0
            else:
                recharge_time = required_uav_energy_j / RECHARGE_POWER_W

        # UAV and UGV operate concurrently.
        segment_motion_time_s = max(
            uav_time_s,
            ugv_time_s,
        )

        arrival_time_s = current_time_s + segment_motion_time_s

        departure_time_s = arrival_time_s + recharge_time

        segments.append(
            SegmentResult(
                p=p,
                origin=origin,
                dest=destination,
                task_indices=list(task_indices),
                uav_path=list(uav_path),
                uav_energy_kJ=float(uav_energy_kJ),
            )
        )

        current_time_s = departure_time_s

    return PlanResult(
        segments=segments,
        makespan_s=float(current_time_s),
    )


# ============================================================
# Adaptive cover radius
# ============================================================


def adaptive_cover_radius_factor(
    map_size_km: float,
) -> float:
    """
    Return the cover-radius factor used by the experiments.

    Small:
        <= 16 km -> 0.25

    Medium:
        <= 25 km -> 0.18

    Large:
        > 25 km -> 0.12
    """

    if map_size_km <= 0:
        raise ValueError("map_size_km must be greater than 0.")

    if map_size_km <= 16.0:
        return 0.25

    if map_size_km <= 25.0:
        return 0.18

    return 0.12


# ============================================================
# RL action translation
# ============================================================


def _nearest_node_index(
    nodes_loc: np.ndarray,
    point: Point,
) -> int:
    """
    Find index of the node closest to a route point.
    """

    distances = np.linalg.norm(
        nodes_loc
        - np.asarray(
            point,
            dtype=np.float32,
        ),
        axis=1,
    )

    return int(np.argmin(distances))


def translate_plan_to_actions(
    plan: PlanResult,
    nodes_loc: np.ndarray,
) -> list[Action]:
    """
    Convert an expert cooperative plan into RL actions.

    RL action format:

        [
            uav_target_index,
            ugv_target_index,
            mode,
        ]

    Modes:

        0 -> UGV moves
        1 -> UGV waits / UAV rendezvous-recharge

    The generated action sequence can be used as expert
    demonstrations for Behavioral Cloning.
    """

    nodes = np.asarray(
        nodes_loc,
        dtype=np.float32,
    )

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes_loc must have shape (num_nodes, 2).")

    actions_queue: list[Action] = []

    for segment in plan.segments:

        destination_idx = _nearest_node_index(
            nodes,
            segment.dest,
        )

        # ----------------------------------------------------
        # Zero-fail carrier
        # ----------------------------------------------------

        if segment.uav_energy_kJ == 0.0 and len(segment.task_indices) > 0:

            # UAV rides on UGV through all points.
            for point in segment.uav_path[1:]:

                point_idx = _nearest_node_index(
                    nodes,
                    point,
                )

                actions_queue.append(
                    [
                        point_idx,
                        point_idx,
                        0,
                    ]
                )

        # ----------------------------------------------------
        # Normal cooperative operation
        # ----------------------------------------------------

        else:

            # UAV follows its optimized route while
            # UGV heads toward the segment destination.
            for point in segment.uav_path[1:]:

                point_idx = _nearest_node_index(
                    nodes,
                    point,
                )

                actions_queue.append(
                    [
                        point_idx,
                        destination_idx,
                        0,
                    ]
                )

        # ----------------------------------------------------
        # Rendezvous / recharge
        # ----------------------------------------------------

        actions_queue.append(
            [
                destination_idx,
                destination_idx,
                1,
            ]
        )

    return actions_queue


def translate_ortools_to_actions(
    coop_plan: PlanResult,
    nodes_loc: np.ndarray,
) -> list[Action]:
    """
    Backward-compatible alias used by the original notebook.

    New code may use ``translate_plan_to_actions`` instead.
    """

    return translate_plan_to_actions(
        plan=coop_plan,
        nodes_loc=nodes_loc,
    )


# ============================================================
# Expert-plan convenience function
# ============================================================


def generate_expert_plan(
    nodes_loc: np.ndarray,
    map_size_km: float,
    *,
    uav_solver: str = "heuristic",
    cpsat_limit_s: float = 10.0,
) -> tuple[
    PlanResult,
    list[Action],
]:
    """
    Generate an expert cooperative plan directly from a node array.

    Parameters
    ----------
    nodes_loc:
        Node coordinates.

        nodes_loc[0] = depot
        nodes_loc[1:] = tasks

    map_size_km:
        Map size used to select the adaptive coverage radius.

    uav_solver:
        "heuristic" or "cpsat".

    Returns
    -------
    tuple
        (
            PlanResult,
            expert_action_queue,
        )
    """

    nodes = np.asarray(
        nodes_loc,
        dtype=np.float32,
    )

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes_loc must have shape (num_nodes, 2).")

    if len(nodes) < 2:
        raise ValueError(
            "nodes_loc must contain a depot and at least one task."
        )

    depot = _as_point(nodes[0])

    tasks = [_as_point(point) for point in nodes[1:]]

    uav = UAV(
        speed_mps=UAV_SPEED_MPS,
        max_energy_j=MAX_FUEL_J,
    )

    ugv = UGV(
        speed_mps=UGV_SPEED_MPS,
    )

    cover_factor = adaptive_cover_radius_factor(map_size_km)

    plan = cooperative_plan(
        tasks=tasks,
        depot=depot,
        uav=uav,
        ugv=ugv,
        cover_radius_factor=cover_factor,
        uav_solver=uav_solver,
        cpsat_limit_s=cpsat_limit_s,
    )

    actions = translate_plan_to_actions(
        plan=plan,
        nodes_loc=nodes,
    )

    return (
        plan,
        actions,
    )


# ============================================================
# Public API
# ============================================================

__all__ = [
    # Types / results
    "Point",
    "Action",
    "SegmentResult",
    "PlanResult",
    # Geometry
    "dist_km",
    "path_length_km",
    # TSP
    "tsp_nearest_neighbor",
    "cycle_length_km",
    "two_opt_cycle",
    "tsp_cycle",
    # UGV planning
    "greedy_min_set_cover",
    "allocate_tasks_to_segments",
    # UAV planning
    "cheapest_insertion_path",
    "two_opt_path",
    "solve_uav_segment_heuristic",
    "solve_uav_segment_cpsat_optional",
    # Cooperative planner
    "cooperative_plan",
    "adaptive_cover_radius_factor",
    # RL expert actions
    "translate_plan_to_actions",
    "translate_ortools_to_actions",
    "generate_expert_plan",
]
