"""
UAV-UGV cooperative routing environment.

This module defines the Gymnasium environment used by the RL agent.

Action
------
[action_uav, action_ugv, mode]

action_uav:
    Target node selected for the UAV.

action_ugv:
    Target node selected for the UGV.

mode:
    0 -> UGV moves toward action_ugv.
    1 -> UGV stays at its current location.

Observation
-----------
uav_state:
    [x, y, remaining_energy]

ugv_state:
    [x, y]

nodes_features:
    For every node:
    [
        x,
        y,
        visited,
        relative_x_to_uav,
        relative_y_to_uav,
        relative_x_to_ugv,
        relative_y_to_ugv,
    ]

All positions are normalized by map size.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .physics import (
    HOVER_POWER_W,
    MAX_FUEL_J,
    RECHARGE_POWER_W,
    UAV_ENERGY_PER_METER,
    UAV_SPEED_MPS,
    UGV_SPEED_MPS,
)


# ============================================================
# Action modes
# ============================================================

MODE_MOVE = 0
MODE_WAIT = 1


class UAVUGVEnv(gym.Env):
    """
    Gymnasium environment for cooperative UAV-UGV routing.

    Parameters
    ----------
    num_tasks:
        Number of task nodes, excluding the depot.

    map_size:
        Width/height of the square map in kilometres.

    safety_factor:
        Energy safety margin used by dynamic energy-aware masking.

        Example:
            1.05 -> require 5% more energy than theoretically needed.

    dock_tolerance_km:
        Maximum distance between UAV and UGV to consider them docked.

    min_flying_ratio:
        Minimum desired proportion of UAV flying steps.

    enforce_min_flying_ratio:
        Whether to discourage excessive UAV riding through action masking.

    max_steps_multiplier:
        Maximum episode length is:

            max_steps = num_nodes * max_steps_multiplier

    stuck_limit:
        Number of consecutive steps without visiting new nodes before
        stuck recovery is triggered.

    enable_stuck_recovery:
        Enable the recovery mechanism from the original experimental code.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(
        self,
        num_tasks: int = 30,
        map_size: float = 16.0,
        safety_factor: float = 1.05,
        dock_tolerance_km: float = 0.1,
        min_flying_ratio: float = 0.10,
        enforce_min_flying_ratio: bool = True,
        max_steps_multiplier: int = 5,
        stuck_limit: int = 5,
        enable_stuck_recovery: bool = True,
    ) -> None:
        super().__init__()

        if num_tasks <= 0:
            raise ValueError("num_tasks must be greater than 0.")

        if map_size <= 0:
            raise ValueError("map_size must be greater than 0.")

        if safety_factor < 1.0:
            raise ValueError("safety_factor must be >= 1.0.")

        self.num_tasks = int(num_tasks)
        self.num_nodes = self.num_tasks + 1

        self.area_size = float(map_size)

        self.safety_factor = float(safety_factor)
        self.dock_tolerance_km = float(dock_tolerance_km)

        self.min_flying_ratio = float(min_flying_ratio)
        self.enforce_min_flying_ratio = enforce_min_flying_ratio

        self.max_steps_multiplier = int(max_steps_multiplier)
        self.stuck_limit = int(stuck_limit)

        self.enable_stuck_recovery = enable_stuck_recovery

        # Used while replaying OR-Tools expert demonstrations.
        # Fuel feasibility checks can be bypassed in expert mode.
        self.expert_mode = False

        self._configure_spaces()

        # State variables are initialized by reset().
        self.nodes_loc: np.ndarray
        self.uav_pos: np.ndarray
        self.ugv_pos: np.ndarray

        self.uav_fuel_J = MAX_FUEL_J

        self.uav_curr_idx = 0
        self.ugv_curr_idx = 0

        self.visited: np.ndarray

        self.total_time_s = 0.0
        self.total_energy_J = 0.0

        self.flying_steps = 0
        self.riding_steps = 0

        self.consecutive_fly_tasks = 0

        self.stuck_counter = 0
        self.step_count = 0

        self.last_min_dist = 0.0

        self.success = False

    # ========================================================
    # Gymnasium spaces
    # ========================================================

    def _configure_spaces(self) -> None:
        """
        Configure action and observation spaces.

        This method is called again if fixed benchmark nodes with a
        different number of tasks are supplied to reset().
        """

        # [UAV target, UGV target, mode]
        self.action_space = spaces.MultiDiscrete(
            [
                self.num_nodes,
                self.num_nodes,
                2,
            ]
        )

        self.observation_space = spaces.Dict(
            {
                "uav_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "ugv_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),

                # Features:
                #
                # x
                # y
                # visited
                # relative x to UAV
                # relative y to UAV
                # relative x to UGV
                # relative y to UGV
                #
                # Relative positions can be negative.
                "nodes_features": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.num_nodes, 7),
                    dtype=np.float32,
                ),
            }
        )

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Reset the environment.

        options may contain:

        fixed_nodes:
            Predefined benchmark node coordinates.

        map_size:
            Map size associated with fixed_nodes.

        expert_mode:
            Whether the environment should run in expert replay mode.
        """

        super().reset(seed=seed)

        options = options or {}

        # ----------------------------------------------------
        # Load benchmark instance or create random map
        # ----------------------------------------------------

        if "fixed_nodes" in options:
            nodes = np.asarray(
                options["fixed_nodes"],
                dtype=np.float32,
            )

            if nodes.ndim != 2 or nodes.shape[1] != 2:
                raise ValueError(
                    "fixed_nodes must have shape (num_nodes, 2)."
                )

            if len(nodes) < 2:
                raise ValueError(
                    "fixed_nodes must contain a depot and at least one task."
                )

            self.nodes_loc = nodes.copy()

            self.num_nodes = len(nodes)
            self.num_tasks = self.num_nodes - 1

            self.area_size = float(
                options.get(
                    "map_size",
                    self.area_size,
                )
            )

            # Fixed benchmark may contain a different number of nodes.
            self._configure_spaces()

        else:
            self.nodes_loc = self.np_random.uniform(
                low=0.0,
                high=self.area_size,
                size=(self.num_nodes, 2),
            ).astype(np.float32)

            # Depot location used in the original implementation.
            self.nodes_loc[0] = np.array(
                [
                    self.area_size * 0.1,
                    self.area_size * 0.1,
                ],
                dtype=np.float32,
            )

        if "expert_mode" in options:
            self.expert_mode = bool(options["expert_mode"])

        # ----------------------------------------------------
        # Initial UAV / UGV state
        # ----------------------------------------------------

        self.uav_pos = self.nodes_loc[0].copy()
        self.ugv_pos = self.nodes_loc[0].copy()

        self.uav_curr_idx = 0
        self.ugv_curr_idx = 0

        self.uav_fuel_J = MAX_FUEL_J

        # Depot is already visited.
        self.visited = np.zeros(
            self.num_nodes,
            dtype=np.int8,
        )

        self.visited[0] = 1

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.total_time_s = 0.0
        self.total_energy_J = 0.0

        self.flying_steps = 0
        self.riding_steps = 0

        self.consecutive_fly_tasks = 0

        self.stuck_counter = 0
        self.step_count = 0

        self.success = False

        self.last_min_dist = self._get_min_unvisited_dist()

        return self._get_obs(), self._get_info()

    # ========================================================
    # Observation
    # ========================================================

    def _get_obs(self) -> dict[str, np.ndarray]:
        """
        Build normalized neural-network observation.
        """

        # Absolute normalized node positions.
        norm_nodes = self.nodes_loc / self.area_size

        # Relative node positions to UAV.
        uav_relative = (
            self.nodes_loc - self.uav_pos
        ) / self.area_size

        # Relative node positions to UGV.
        ugv_relative = (
            self.nodes_loc - self.ugv_pos
        ) / self.area_size

        visited_column = self.visited.reshape(-1, 1).astype(
            np.float32
        )

        node_features = np.hstack(
            [
                norm_nodes,
                visited_column,
                uav_relative,
                ugv_relative,
            ]
        ).astype(np.float32)

        fuel_ratio = np.clip(
            self.uav_fuel_J / MAX_FUEL_J,
            0.0,
            1.0,
        )

        uav_state = np.array(
            [
                self.uav_pos[0] / self.area_size,
                self.uav_pos[1] / self.area_size,
                fuel_ratio,
            ],
            dtype=np.float32,
        )

        ugv_state = np.array(
            [
                self.ugv_pos[0] / self.area_size,
                self.ugv_pos[1] / self.area_size,
            ],
            dtype=np.float32,
        )

        return {
            "uav_state": uav_state,
            "ugv_state": ugv_state,
            "nodes_features": node_features,
        }

    # ========================================================
    # Information
    # ========================================================

    def _get_info(
        self,
        *,
        safe_override: bool = False,
        new_visits: int = 0,
    ) -> dict[str, Any]:

        total_motion_steps = (
            self.flying_steps
            + self.riding_steps
        )

        flying_ratio = (
            self.flying_steps
            / max(1, total_motion_steps)
        )

        return {
            "success": self.success,
            "total_time_s": self.total_time_s,
            "total_energy_J": self.total_energy_J,
            "fuel_J": self.uav_fuel_J,
            "fuel_ratio": self.uav_fuel_J / MAX_FUEL_J,
            "visited_tasks": int(
                np.sum(self.visited[1:])
            ),
            "total_tasks": self.num_tasks,
            "flying_steps": self.flying_steps,
            "riding_steps": self.riding_steps,
            "flying_ratio": flying_ratio,
            "safe_override": safe_override,
            "new_visits": new_visits,
            "step_count": self.step_count,
        }

    # ========================================================
    # Utility functions
    # ========================================================

    def _nearest_node_index(
        self,
        position: np.ndarray,
    ) -> int:
        """
        Return the node nearest to the supplied position.
        """

        distances = np.linalg.norm(
            self.nodes_loc - position,
            axis=1,
        )

        return int(np.argmin(distances))

    def _get_min_unvisited_dist(self) -> float:
        """
        Distance from UAV to the nearest unvisited node in kilometres.
        """

        unvisited = np.where(
            self.visited == 0
        )[0]

        if len(unvisited) == 0:
            return 0.0

        distances = np.linalg.norm(
            self.nodes_loc[unvisited] - self.uav_pos,
            axis=1,
        )

        return float(np.min(distances))

    def _is_docked(self) -> bool:
        """
        Check whether UAV and UGV are at approximately
        the same position.
        """

        return bool(
            np.linalg.norm(
                self.uav_pos - self.ugv_pos
            )
            < self.dock_tolerance_km
        )

    # ========================================================
    # Energy calculations
    # ========================================================

    def required_energy_to_visit_and_return(
        self,
        target_idx: int,
    ) -> float:
        """
        Calculate UAV energy required to:

            UAV -> target -> current UGV position

        Returns
        -------
        float
            Required energy in joules.
        """

        target = self.nodes_loc[target_idx]

        dist_to_target_m = (
            np.linalg.norm(
                target - self.uav_pos
            )
            * 1000.0
        )

        dist_return_m = (
            np.linalg.norm(
                self.ugv_pos - target
            )
            * 1000.0
        )

        total_distance_m = (
            dist_to_target_m
            + dist_return_m
        )

        return (
            total_distance_m
            * UAV_ENERGY_PER_METER
        )

    def is_energy_feasible(
        self,
        target_idx: int,
    ) -> bool:
        """
        Check whether the UAV can safely visit a node
        and still return to the UGV.
        """

        required_energy = (
            self.required_energy_to_visit_and_return(
                target_idx
            )
        )

        return bool(
            self.uav_fuel_J
            >= required_energy
            * self.safety_factor
        )

    # ========================================================
    # Dynamic energy-aware action masking
    # ========================================================

    def get_action_masks(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate valid-action masks.

        Returns
        -------
        uav_mask:
            Valid UAV target nodes.

        ugv_mask:
            Valid UGV target nodes.

        mode_mask:
            Valid operating modes.

            index 0 -> MOVE
            index 1 -> WAIT

        Notes
        -----
        UAV nodes are dynamically masked according to:

            current UAV energy
            + flight distance
            + required return distance
            + safety margin

        This is the project's dynamic energy-aware masking.
        """

        uav_mask = np.zeros(
            self.num_nodes,
            dtype=bool,
        )

        ugv_mask = np.zeros(
            self.num_nodes,
            dtype=bool,
        )

        mode_mask = np.zeros(
            2,
            dtype=bool,
        )

        unvisited = np.where(
            self.visited == 0
        )[0]

        ugv_current_node = self._nearest_node_index(
            self.ugv_pos
        )

        is_docked = self._is_docked()

        # Episode is complete.
        if len(unvisited) == 0:
            uav_mask[ugv_current_node] = True
            ugv_mask[ugv_current_node] = True
            mode_mask[MODE_WAIT] = True

            return (
                uav_mask,
                ugv_mask,
                mode_mask,
            )

        # ----------------------------------------------------
        # Current flying ratio
        # ----------------------------------------------------

        total_steps = (
            self.flying_steps
            + self.riding_steps
        )

        flying_ratio = (
            self.flying_steps
            / max(1, total_steps)
        )

        must_fly = (
            self.enforce_min_flying_ratio
            and total_steps >= 3
            and flying_ratio < self.min_flying_ratio
        )

        # ----------------------------------------------------
        # Dynamic energy-aware mask
        # ----------------------------------------------------

        safe_flying_targets: list[int] = []

        for node_idx in unvisited:

            if self.is_energy_feasible(
                int(node_idx)
            ):
                uav_mask[node_idx] = True

                safe_flying_targets.append(
                    int(node_idx)
                )

        has_safe_target = (
            len(safe_flying_targets) > 0
        )

        # ----------------------------------------------------
        # No safe flight is currently possible
        # ----------------------------------------------------

        if not has_safe_target:

            if is_docked:
                # UAV rides with UGV toward a new task.
                ugv_mask[unvisited] = True
                uav_mask[unvisited] = True

                mode_mask[MODE_MOVE] = True

            else:
                # UAV must return to UGV.
                uav_mask[ugv_current_node] = True
                ugv_mask[ugv_current_node] = True

                mode_mask[MODE_WAIT] = True

        # ----------------------------------------------------
        # At least one safe UAV flight exists
        # ----------------------------------------------------

        else:

            # Allow UAV to rendezvous with UGV
            # when its energy becomes low.
            if (
                self.uav_fuel_J
                < MAX_FUEL_J * 0.90
            ):
                uav_mask[ugv_current_node] = True

            # UGV may move to any unvisited task.
            ugv_mask[unvisited] = True

            # UGV may also remain at its current node.
            ugv_mask[ugv_current_node] = True

            mode_mask[MODE_MOVE] = True
            mode_mask[MODE_WAIT] = True

            # Prevent pointless recharge while already
            # docked with almost-full battery.
            if (
                is_docked
                and self.uav_fuel_J
                >= MAX_FUEL_J * 0.95
            ):
                uav_mask[ugv_current_node] = False

            # ------------------------------------------------
            # Minimum UAV flying-ratio enforcement
            # ------------------------------------------------

            if must_fly:

                non_ugv_targets = [
                    idx
                    for idx in safe_flying_targets
                    if idx != ugv_current_node
                ]

                if non_ugv_targets:
                    uav_mask[ugv_current_node] = False

                    # Prevent immediate recharge/wait behavior.
                    mode_mask[MODE_WAIT] = False

        # ----------------------------------------------------
        # Safety fallbacks
        # ----------------------------------------------------

        if not np.any(uav_mask):
            uav_mask[ugv_current_node] = True

        if not np.any(ugv_mask):
            ugv_mask[ugv_current_node] = True

        if not np.any(mode_mask):
            mode_mask[MODE_WAIT] = True

        return (
            uav_mask,
            ugv_mask,
            mode_mask,
        )

    # ========================================================
    # Environment transition
    # ========================================================

    def step(self, action):
        """
        Execute one UAV-UGV decision.
        """

        if len(action) != 3:
            raise ValueError(
                "Action must be [uav_target, ugv_target, mode]."
            )

        selected_uav_idx = int(action[0])
        selected_ugv_idx = int(action[1])
        mode = int(action[2])

        if not 0 <= selected_uav_idx < self.num_nodes:
            raise ValueError(
                f"Invalid UAV node index: {selected_uav_idx}"
            )

        if not 0 <= selected_ugv_idx < self.num_nodes:
            raise ValueError(
                f"Invalid UGV node index: {selected_ugv_idx}"
            )

        if mode not in (MODE_MOVE, MODE_WAIT):
            raise ValueError(
                f"Invalid mode: {mode}"
            )

        self.step_count += 1

        # ----------------------------------------------------
        # Determine actual UGV destination
        # ----------------------------------------------------

        ugv_start_node = self._nearest_node_index(
            self.ugv_pos
        )

        if mode == MODE_MOVE:
            actual_ugv_idx = selected_ugv_idx
        else:
            # UGV waits at current position.
            actual_ugv_idx = ugv_start_node

        uav_idx = selected_uav_idx
        ugv_idx = actual_ugv_idx

        uav_destination = self.nodes_loc[
            uav_idx
        ].copy()

        ugv_destination = self.nodes_loc[
            ugv_idx
        ].copy()

        # ----------------------------------------------------
        # Travel distances
        # ----------------------------------------------------

        uav_distance_m = (
            np.linalg.norm(
                uav_destination - self.uav_pos
            )
            * 1000.0
        )

        ugv_distance_m = (
            np.linalg.norm(
                ugv_destination - self.ugv_pos
            )
            * 1000.0
        )

        uav_time_s = (
            uav_distance_m
            / UAV_SPEED_MPS
        )

        ugv_time_s = (
            ugv_distance_m
            / UGV_SPEED_MPS
        )

        start_docked = self._is_docked()

        same_destination = (
            uav_idx == ugv_idx
        )

        # UAV can physically ride the UGV only when
        # it starts the step docked and both vehicles
        # have the same destination.
        is_riding = (
            start_docked
            and same_destination
        )

        safe_override = False
        fuel_failure = False

        # ====================================================
        # Safety validation
        # ====================================================

        if (
            not self.expert_mode
            and not is_riding
        ):

            forward_energy = (
                uav_distance_m
                * UAV_ENERGY_PER_METER
            )

            # UAV may need to hover while waiting for UGV.
            if (
                np.linalg.norm(
                    uav_destination
                    - ugv_destination
                )
                < self.dock_tolerance_km
                and uav_time_s < ugv_time_s
            ):
                forward_energy += (
                    ugv_time_s
                    - uav_time_s
                ) * HOVER_POWER_W

            return_distance_m = (
                np.linalg.norm(
                    ugv_destination
                    - uav_destination
                )
                * 1000.0
            )

            return_energy = (
                return_distance_m
                * UAV_ENERGY_PER_METER
            )

            required_energy = (
                forward_energy
                + return_energy
            )

            # ------------------------------------------------
            # Invalid action -> force UAV to return
            # to the current UGV location.
            # ------------------------------------------------

            if (
                self.uav_fuel_J
                < required_energy
                * self.safety_factor
            ):

                safe_override = True

                ugv_idx = ugv_start_node
                uav_idx = ugv_start_node

                ugv_destination = (
                    self.ugv_pos.copy()
                )

                uav_destination = (
                    self.ugv_pos.copy()
                )

                ugv_distance_m = 0.0
                ugv_time_s = 0.0

                uav_distance_m = (
                    np.linalg.norm(
                        uav_destination
                        - self.uav_pos
                    )
                    * 1000.0
                )

                uav_time_s = (
                    uav_distance_m
                    / UAV_SPEED_MPS
                )

                mode = MODE_WAIT

                is_riding = start_docked

        # ====================================================
        # Execute motion
        # ====================================================

        sortie_bonus = 0.0
        uav_fuel_cost = 0.0

        rendezvous = (
            np.linalg.norm(
                uav_destination
                - ugv_destination
            )
            < self.dock_tolerance_km
        )

        # ----------------------------------------------------
        # UAV riding on UGV
        # ----------------------------------------------------

        if is_riding:

            recharge_time_s = max(
                0.0,
                MAX_FUEL_J - self.uav_fuel_J,
            ) / RECHARGE_POWER_W

            # Recharge can happen while travelling on UGV.
            step_time_s = max(
                ugv_time_s,
                recharge_time_s,
            )

            self.uav_fuel_J = MAX_FUEL_J

            self.riding_steps += 1

            if self.consecutive_fly_tasks >= 2:
                sortie_bonus = (
                    5.0
                    * self.consecutive_fly_tasks
                )

            self.consecutive_fly_tasks = 0

        # ----------------------------------------------------
        # UAV flying
        # ----------------------------------------------------

        else:

            uav_fuel_cost = (
                uav_distance_m
                * UAV_ENERGY_PER_METER
            )

            # UAV reaches rendezvous before UGV and must hover.
            if (
                rendezvous
                and uav_time_s < ugv_time_s
            ):

                hover_time_s = (
                    ugv_time_s
                    - uav_time_s
                )

                uav_fuel_cost += (
                    hover_time_s
                    * HOVER_POWER_W
                )

            step_time_s = max(
                uav_time_s,
                ugv_time_s,
            )

            self.uav_fuel_J -= uav_fuel_cost

            if self.expert_mode:
                # Expert trajectories originate from OR-Tools
                # and are trusted during BC replay.
                self.uav_fuel_J = max(
                    0.0,
                    self.uav_fuel_J,
                )

            elif self.uav_fuel_J < 0.0:
                fuel_failure = True

            # ------------------------------------------------
            # Recharge after rendezvous
            # ------------------------------------------------

            if (
                rendezvous
                and not fuel_failure
            ):

                recharge_time_s = (
                    max(
                        0.0,
                        MAX_FUEL_J
                        - self.uav_fuel_J,
                    )
                    / RECHARGE_POWER_W
                )

                # UAV has to land first,
                # therefore recharge is sequential.
                step_time_s += (
                    recharge_time_s
                )

                self.uav_fuel_J = MAX_FUEL_J

                if self.consecutive_fly_tasks >= 2:
                    sortie_bonus = (
                        5.0
                        * self.consecutive_fly_tasks
                    )

                self.consecutive_fly_tasks = 0

            else:
                self.consecutive_fly_tasks += 1

            self.flying_steps += 1

        # ====================================================
        # Update state
        # ====================================================

        self.total_energy_J += max(
            0.0,
            uav_fuel_cost,
        )

        self.total_time_s += step_time_s

        self.uav_pos = uav_destination.copy()
        self.ugv_pos = ugv_destination.copy()

        self.uav_curr_idx = uav_idx
        self.ugv_curr_idx = ugv_idx

        # ----------------------------------------------------
        # Count newly completed tasks
        # ----------------------------------------------------

        new_visits = 0

        if self.visited[uav_idx] == 0:
            new_visits += 1

        if (
            ugv_idx != uav_idx
            and self.visited[ugv_idx] == 0
        ):
            new_visits += 1

        self.visited[uav_idx] = 1
        self.visited[ugv_idx] = 1

        if new_visits > 0:
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1

        # ====================================================
        # Reward shaping
        # ====================================================

        time_scale = (
            self.area_size
            * 100.0
        )

        # ----------------------------------------------------
        # 1. Makespan penalty
        # ----------------------------------------------------

        reward = (
            -step_time_s
            / time_scale
        )

        # ----------------------------------------------------
        # 2. Task progress
        # ----------------------------------------------------

        tasks_remaining = int(
            np.sum(
                self.visited == 0
            )
        )

        tasks_total = self.num_tasks

        progress_ratio = (
            1.0
            - tasks_remaining
            / max(1, tasks_total)
        )

        # ----------------------------------------------------
        # 3. Distance improvement
        # ----------------------------------------------------

        current_min_dist = (
            self._get_min_unvisited_dist()
        )

        distance_improvement = (
            self.last_min_dist
            - current_min_dist
        )

        if distance_improvement > 0:
            reward += (
                3.0
                * distance_improvement
                / self.area_size
            )

        self.last_min_dist = (
            current_min_dist
        )

        # ----------------------------------------------------
        # 4. New task completion bonus
        # ----------------------------------------------------

        if new_visits > 0:

            completion_bonus = (
                2.0
                + 3.0
                * progress_ratio
            )

            reward += (
                completion_bonus
                * new_visits
            )

            # UAV and UGV finish different tasks
            # simultaneously.
            if new_visits == 2:
                reward += 100.0

        # ----------------------------------------------------
        # 5. Discourage unnecessary riding
        # ----------------------------------------------------

        if (
            is_riding
            and self.uav_fuel_J
            > MAX_FUEL_J * 0.5
        ):
            reward -= 3.0

        # Flying and UGV travelling to exactly the same task
        # is inefficient compared with simply riding.
        if (
            not is_riding
            and uav_idx == ugv_idx
        ):
            reward -= 50.0

        # ----------------------------------------------------
        # 6. Multi-task sortie bonus
        # ----------------------------------------------------

        reward += sortie_bonus

        # ----------------------------------------------------
        # 7. Fuel preservation bonus
        # ----------------------------------------------------

        fuel_ratio = (
            self.uav_fuel_J
            / MAX_FUEL_J
        )

        if fuel_ratio > 0.30:
            reward += 0.5

        # ----------------------------------------------------
        # 8. Invalid/unsafe action penalty
        # ----------------------------------------------------

        if safe_override:
            reward -= 5.0

        # ====================================================
        # Termination
        # ====================================================

        terminated = False
        truncated = False

        # ----------------------------------------------------
        # Fuel failure
        # ----------------------------------------------------

        if fuel_failure:
            reward -= 100.0
            terminated = True

        # ----------------------------------------------------
        # Stuck recovery
        # ----------------------------------------------------

        if (
            not terminated
            and not self.expert_mode
            and self.stuck_counter
            >= self.stuck_limit
        ):

            reward -= 30.0

            if self.enable_stuck_recovery:

                unvisited = np.where(
                    self.visited == 0
                )[0]

                if len(unvisited) > 0:

                    recovery_idx = int(
                        unvisited[0]
                    )

                    recovery_position = (
                        self.nodes_loc[
                            recovery_idx
                        ].copy()
                    )

                    self.uav_pos = (
                        recovery_position.copy()
                    )

                    self.ugv_pos = (
                        recovery_position.copy()
                    )

                    self.uav_curr_idx = (
                        recovery_idx
                    )

                    self.ugv_curr_idx = (
                        recovery_idx
                    )

                    self.visited[
                        recovery_idx
                    ] = 1

                    self.uav_fuel_J = (
                        MAX_FUEL_J
                    )

                    self.stuck_counter = 0

                    # Equivalent recovery penalty
                    # used by the original experiment.
                    self.total_time_s += 500.0

            else:
                truncated = True

        # ----------------------------------------------------
        # Successful completion
        # ----------------------------------------------------

        if (
            not fuel_failure
            and np.all(
                self.visited == 1
            )
        ):

            self.success = True
            terminated = True
            truncated = False

            base_reward = (
                50.0
                + 20.0
                * (
                    tasks_total
                    / 50.0
                )
            )

            time_bonus = (
                time_scale
                * 10.0
            ) / max(
                1.0,
                self.total_time_s,
            )

            reward += (
                base_reward
                + time_bonus
            )

        # ----------------------------------------------------
        # Maximum episode length
        # ----------------------------------------------------

        max_steps = (
            self.num_nodes
            * self.max_steps_multiplier
        )

        if (
            not terminated
            and self.step_count
            >= max_steps
        ):
            truncated = True

        info = self._get_info(
            safe_override=safe_override,
            new_visits=new_visits,
        )

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ========================================================
    # Expert mode
    # ========================================================

    def set_expert_mode(
        self,
        enabled: bool = True,
    ) -> None:
        """
        Enable/disable expert replay mode.

        Expert mode is used while collecting OR-Tools
        demonstrations for Behavioral Cloning.
        """

        self.expert_mode = bool(enabled)

    # ========================================================
    # Render
    # ========================================================

    def render(self) -> None:
        """
        Simple textual environment rendering.
        """

        visited_tasks = int(
            np.sum(
                self.visited[1:]
            )
        )

        fuel_percent = (
            self.uav_fuel_J
            / MAX_FUEL_J
            * 100.0
        )

        print(
            f"Step: {self.step_count} | "
            f"Tasks: {visited_tasks}/{self.num_tasks} | "
            f"Fuel: {fuel_percent:.1f}% | "
            f"Time: {self.total_time_s:.1f}s | "
            f"Energy: {self.total_energy_J / 1000.0:.2f} kJ"
        )


# ============================================================
# Backward-compatible aliases
# ============================================================

# Main name recommended for the new project.
UAVUGVCoopEnv = UAVUGVEnv

# Names used by versions of the original notebook.
UAVUGVCoopEnv_Benchmark = UAVUGVEnv
UAVUGVCoopEnv_Flex = UAVUGVEnv


__all__ = [
    "MODE_MOVE",
    "MODE_WAIT",
    "UAVUGVEnv",
    "UAVUGVCoopEnv",
    "UAVUGVCoopEnv_Benchmark",
    "UAVUGVCoopEnv_Flex",
]