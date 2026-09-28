"""
Common utility functions for the UAV-UGV routing project.

This module contains lightweight helpers shared across training,
evaluation, ablation studies, scripts, and notebooks.

Main responsibilities
---------------------
- Reproducibility / random seeds
- Device detection
- JSON / YAML loading and saving
- NumPy -> JSON serialization
- Scenario validation and normalization
- Directory / checkpoint path helpers
- Basic experiment statistics
- Formatting utilities

Design note
-----------
This module intentionally does not import other project modules such as:

    environment.py
    model.py
    agent.py
    training.py

This keeps ``utils.py`` dependency-free from the rest of the package
and avoids circular imports.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

# ============================================================
# Project paths
# ============================================================


def get_project_root() -> Path:
    """
    Return the repository root directory.

    Expected structure:

        project/
        ├── configs/
        ├── data/
        ├── scripts/
        └── src/
            └── uav_ugv_rl/
                └── utils.py

    Therefore ``utils.py`` is located three levels below the
    project root.
    """

    return Path(__file__).resolve().parents[2]


def ensure_directory(
    path: str | Path,
) -> Path:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        Resolved Path object.
    """

    directory = Path(path)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


# ============================================================
# Reproducibility
# ============================================================


def set_seed(
    seed: int,
    *,
    deterministic: bool = False,
) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Random seed.

    deterministic:
        If True, request deterministic PyTorch algorithms where
        possible.

        Deterministic execution can reduce performance and some
        operations may not have deterministic implementations.
    """

    if seed < 0:
        raise ValueError("seed must be greater than or equal to 0.")

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:

        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


# Backward-compatible / descriptive alias.
set_random_seed = set_seed


# ============================================================
# Device helpers
# ============================================================


def get_device(
    preferred: str | None = None,
) -> torch.device:
    """
    Select a PyTorch device.

    Parameters
    ----------
    preferred:
        Optional preferred device:

            "cuda"
            "mps"
            "cpu"

        If None, the best available device is selected automatically.

    Returns
    -------
    torch.device
    """

    if preferred is not None:

        preferred = preferred.lower()

        if preferred == "cuda":

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")

            return torch.device("cuda")

        if preferred == "mps":

            if not (
                hasattr(
                    torch.backends,
                    "mps",
                )
                and torch.backends.mps.is_available()
            ):
                raise RuntimeError("MPS was requested but is not available.")

            return torch.device("mps")

        if preferred == "cpu":
            return torch.device("cpu")

        raise ValueError("preferred must be one of: " "'cuda', 'mps', 'cpu'.")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def device_summary(
    device: str | torch.device | None = None,
) -> str:
    """
    Return a human-readable description of the selected device.
    """

    if device is None:
        device_obj = get_device()
    else:
        device_obj = torch.device(device)

    if device_obj.type == "cuda":

        index = (
            device_obj.index
            if device_obj.index is not None
            else torch.cuda.current_device()
        )

        name = torch.cuda.get_device_name(index)

        memory_gb = torch.cuda.get_device_properties(index).total_memory / 1e9

        return f"CUDA: {name} " f"({memory_gb:.1f} GB)"

    if device_obj.type == "mps":
        return "Apple Metal Performance Shaders (MPS)"

    return "CPU"


# ============================================================
# JSON serialization
# ============================================================


def to_serializable(
    value: Any,
) -> Any:
    """
    Convert common scientific Python objects into JSON-compatible
    Python objects.

    Supported conversions include:

        np.ndarray      -> list
        np.float*       -> float
        np.int*         -> int
        np.bool_        -> bool
        torch.Tensor    -> list / scalar
        Path            -> str

    Containers are processed recursively.
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
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        torch.Tensor,
    ):

        tensor = value.detach().cpu()

        if tensor.numel() == 1:
            return tensor.item()

        return tensor.tolist()

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        Mapping,
    ):
        return {str(key): to_serializable(item) for key, item in value.items()}

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [to_serializable(item) for item in value]

    return value


def load_json(
    path: str | Path,
) -> Any:
    """
    Load a JSON file.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_json(
    data: Any,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """
    Save data to JSON.

    NumPy arrays, tensors, and NumPy scalar values are automatically
    converted to JSON-compatible representations.
    """

    file_path = Path(path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_data = to_serializable(data)

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serializable_data,
            file,
            indent=indent,
            ensure_ascii=False,
        )

    return file_path


# ============================================================
# YAML helpers
# ============================================================


def _import_yaml():
    """
    Import PyYAML lazily.

    PyYAML is kept as an optional runtime dependency of this module
    so importing ``uav_ugv_rl.utils`` does not immediately require it.
    """

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML configuration files. "
            "Install it with: pip install pyyaml"
        ) from exc

    return yaml


def load_yaml(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration.
    """

    yaml = _import_yaml()

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "The root of a configuration YAML file "
            "must be a mapping/dictionary."
        )

    return data


def save_yaml(
    data: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """
    Save a mapping to a YAML file.
    """

    yaml = _import_yaml()

    file_path = Path(path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_data = to_serializable(dict(data))

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            serializable_data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    return file_path


# ============================================================
# Configuration helpers
# ============================================================


def deep_update(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values in ``override`` replace values in ``base``.

    Example
    -------
    base:

        {
            "training": {
                "lr": 0.0002,
                "gamma": 0.99,
            }
        }

    override:

        {
            "training": {
                "gamma": 0.995,
            }
        }

    result:

        {
            "training": {
                "lr": 0.0002,
                "gamma": 0.995,
            }
        }
    """

    result = {key: value for key, value in base.items()}

    for key, value in override.items():

        if (
            key in result
            and isinstance(
                result[key],
                Mapping,
            )
            and isinstance(
                value,
                Mapping,
            )
        ):

            result[key] = deep_update(
                result[key],
                value,
            )

        else:

            result[key] = value

    return result


def load_config(
    config_path: str | Path,
    *,
    default_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load a YAML experiment configuration.

    Optionally merge it with a default YAML configuration.

    Parameters
    ----------
    config_path:
        Scale-specific configuration, for example:

            configs/small.yaml

    default_path:
        Optional shared defaults:

            configs/default.yaml

    Returns
    -------
    dict
        Final merged configuration.
    """

    config = load_yaml(config_path)

    if default_path is None:
        return config

    defaults = load_yaml(default_path)

    return deep_update(
        defaults,
        config,
    )


# ============================================================
# Scenario validation
# ============================================================


def validate_nodes(
    nodes: Any,
    *,
    map_size: float | None = None,
) -> np.ndarray:
    """
    Validate and normalize node coordinates.

    Parameters
    ----------
    nodes:
        Node coordinates with shape:

            [num_nodes, 2]

    map_size:
        Optional map size in kilometres.

        When supplied, coordinates are checked to ensure they are
        within approximately [0, map_size].

    Returns
    -------
    np.ndarray
        float32 node coordinates.
    """

    node_array = np.asarray(
        nodes,
        dtype=np.float32,
    )

    if node_array.ndim != 2:
        raise ValueError("nodes must be a 2-D array.")

    if node_array.shape[1] != 2:
        raise ValueError("nodes must have shape (num_nodes, 2).")

    if len(node_array) < 2:
        raise ValueError(
            "nodes must contain at least " "one depot and one task."
        )

    if not np.all(np.isfinite(node_array)):
        raise ValueError("nodes contain NaN or infinite values.")

    if np.any(node_array < 0):
        raise ValueError("node coordinates cannot be negative.")

    if map_size is not None:

        if map_size <= 0:
            raise ValueError("map_size must be greater than 0.")

        tolerance = 1e-5

        if np.any(node_array > float(map_size) + tolerance):
            raise ValueError("node coordinates exceed map_size.")

    return node_array


def normalize_scenario(
    scenario: Mapping[str, Any],
    *,
    index: int = 0,
) -> dict[str, Any]:
    """
    Normalize different scenario dictionary formats.

    Accepted node keys:

        nodes_loc
        nodes

    Returned format:

        {
            "id": str,
            "scale_name": str,
            "num_tasks": int,
            "map_size": float,
            "nodes_loc": list,
        }

    Extra scenario fields are preserved.
    """

    if not isinstance(
        scenario,
        Mapping,
    ):
        raise TypeError("scenario must be a mapping/dictionary.")

    normalized = dict(scenario)

    # --------------------------------------------------------
    # Map size
    # --------------------------------------------------------

    if "map_size" not in normalized:
        raise KeyError("scenario is missing 'map_size'.")

    map_size = float(normalized["map_size"])

    if map_size <= 0:
        raise ValueError("scenario map_size must be greater than 0.")

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    if "nodes_loc" in normalized:

        raw_nodes = normalized["nodes_loc"]

    elif "nodes" in normalized:

        raw_nodes = normalized["nodes"]

    else:

        raise KeyError("scenario must contain " "'nodes_loc' or 'nodes'.")

    nodes = validate_nodes(
        raw_nodes,
        map_size=map_size,
    )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    scenario_id = str(
        normalized.get(
            "id",
            f"scenario_{index + 1}",
        )
    )

    # --------------------------------------------------------
    # Tasks
    # --------------------------------------------------------

    num_tasks = len(nodes) - 1

    declared_num_tasks = normalized.get("num_tasks")

    if declared_num_tasks is not None:

        if int(declared_num_tasks) != num_tasks:

            raise ValueError(
                f"Scenario '{scenario_id}' declares "
                f"num_tasks={declared_num_tasks}, "
                f"but contains {num_tasks} task nodes."
            )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    scale_name = str(
        normalized.get(
            "scale_name",
            infer_scale_name(
                num_tasks=num_tasks,
                map_size=map_size,
            ),
        )
    )

    normalized.update(
        {
            "id": scenario_id,
            "scale_name": scale_name,
            "num_tasks": num_tasks,
            "map_size": map_size,
            "nodes_loc": nodes.tolist(),
        }
    )

    # Keep one canonical nodes key.
    normalized.pop(
        "nodes",
        None,
    )

    return normalized


def normalize_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Normalize a sequence of scenarios.
    """

    return [
        normalize_scenario(
            scenario,
            index=index,
        )
        for index, scenario in enumerate(scenarios)
    ]


def load_scenarios(
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    Load and validate evaluation scenarios from JSON.

    The JSON root must be a list.
    """

    data = load_json(path)

    if not isinstance(
        data,
        list,
    ):
        raise ValueError("Scenario JSON must contain a list of scenarios.")

    return normalize_scenarios(data)


def save_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """
    Validate and save evaluation scenarios.
    """

    normalized = normalize_scenarios(scenarios)

    return save_json(
        normalized,
        path,
    )


# ============================================================
# Scale helpers
# ============================================================


def infer_scale_name(
    num_tasks: int,
    map_size: float,
) -> str:
    """
    Infer the standard experiment scale.

    Current project convention:

        Small:
            30 tasks / ~16 km

        Medium:
            60 tasks / ~25 km

        Large:
            100 tasks / ~40 km

    The thresholds make the helper usable for nearby custom sizes.
    """

    if num_tasks <= 0:
        raise ValueError("num_tasks must be greater than 0.")

    if map_size <= 0:
        raise ValueError("map_size must be greater than 0.")

    if num_tasks >= 100 or map_size > 25.0:
        return "Large"

    if num_tasks >= 60 or map_size > 16.0:
        return "Medium"

    return "Small"


def is_large_scale(
    num_tasks: int,
    map_size: float,
) -> bool:
    """
    Return True when the large Transformer configuration is appropriate.
    """

    return (
        infer_scale_name(
            num_tasks,
            map_size,
        )
        == "Large"
    )


# ============================================================
# Result / metric helpers
# ============================================================


def percentage_improvement(
    baseline: float,
    candidate: float,
) -> float:
    """
    Calculate percentage improvement over a baseline.

    Positive value:
        candidate is better / smaller.

    Negative value:
        candidate is worse / larger.

    Formula:

        (baseline - candidate) / baseline * 100
    """

    if baseline <= 0:
        raise ValueError("baseline must be greater than 0.")

    return (float(baseline) - float(candidate)) / float(baseline) * 100.0


def safe_mean(
    values: Sequence[float | int],
) -> float | None:
    """
    Calculate a mean while safely handling an empty sequence.
    """

    if len(values) == 0:
        return None

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def moving_average(
    values: Sequence[float | int],
    window: int,
) -> np.ndarray:
    """
    Calculate a moving average.

    Parameters
    ----------
    values:
        Input sequence.

    window:
        Moving-average window size.

    Returns
    -------
    np.ndarray
        Moving average.

    Notes
    -----
    The returned array contains only complete windows.

    Therefore:

        len(result)
        =
        len(values) - window + 1
    """

    if window <= 0:
        raise ValueError("window must be greater than 0.")

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(array) < window:
        return np.array(
            [],
            dtype=np.float64,
        )

    kernel = (
        np.ones(
            window,
            dtype=np.float64,
        )
        / window
    )

    return np.convolve(
        array,
        kernel,
        mode="valid",
    )


def flying_ratio(
    flying_steps: int,
    riding_steps: int,
) -> float:
    """
    Calculate UAV flying-step ratio.
    """

    if flying_steps < 0:
        raise ValueError("flying_steps cannot be negative.")

    if riding_steps < 0:
        raise ValueError("riding_steps cannot be negative.")

    return flying_steps / max(
        1,
        flying_steps + riding_steps,
    )


# ============================================================
# Formatting
# ============================================================


def format_seconds(
    seconds: float,
) -> str:
    """
    Format seconds into a compact human-readable duration.

    Examples
    --------
    45.2:
        "45.2s"

    125:
        "2m 5.0s"

    3725:
        "1h 2m 5.0s"
    """

    if seconds < 0:
        raise ValueError("seconds cannot be negative.")

    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, remaining_seconds = divmod(
        seconds,
        60,
    )

    if minutes < 60:

        return f"{int(minutes)}m " f"{remaining_seconds:.1f}s"

    hours, remaining_minutes = divmod(
        int(minutes),
        60,
    )

    return f"{hours}h " f"{remaining_minutes}m " f"{remaining_seconds:.1f}s"


def format_energy(
    energy_j: float,
) -> str:
    """
    Format energy in J or kJ.
    """

    if energy_j < 0:
        raise ValueError("energy_j cannot be negative.")

    if energy_j >= 1000.0:

        return f"{energy_j / 1000.0:.2f} kJ"

    return f"{energy_j:.2f} J"


# ============================================================
# PyTorch model helpers
# ============================================================


def count_parameters(
    model: torch.nn.Module,
    *,
    trainable_only: bool = True,
) -> int:
    """
    Count model parameters.

    Parameters
    ----------
    model:
        PyTorch module.

    trainable_only:
        Count only parameters where ``requires_grad=True``.
    """

    if trainable_only:

        return sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    return sum(parameter.numel() for parameter in model.parameters())


def model_size_mb(
    model: torch.nn.Module,
) -> float:
    """
    Estimate model parameter + buffer memory in megabytes.
    """

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )

    total_bytes = parameter_bytes + buffer_bytes

    return total_bytes / (1024.0**2)


# ============================================================
# Observation helpers
# ============================================================


def copy_observation(
    observation: Mapping[
        str,
        Any,
    ],
) -> dict[str, Any]:
    """
    Safely copy an environment observation.

    NumPy arrays and tensors are cloned rather than referenced.
    """

    copied: dict[
        str,
        Any,
    ] = {}

    for key, value in observation.items():

        if isinstance(
            value,
            np.ndarray,
        ):

            copied[key] = value.copy()

        elif isinstance(
            value,
            torch.Tensor,
        ):

            copied[key] = value.detach().clone()

        else:

            copied[key] = value

    return copied


# ============================================================
# Checkpoint helpers
# ============================================================


def make_checkpoint_path(
    *,
    checkpoint_dir: str | Path,
    experiment_name: str,
    scenario_id: str | None = None,
    suffix: str = ".pth",
) -> Path:
    """
    Construct a checkpoint path.

    Examples
    --------
    Without scenario:

        checkpoints/full.pth

    With scenario:

        checkpoints/full/small_01.pth
    """

    checkpoint_dir = Path(checkpoint_dir)

    if not experiment_name:
        raise ValueError("experiment_name cannot be empty.")

    if not suffix.startswith("."):
        suffix = "." + suffix

    if scenario_id is None:

        output_path = checkpoint_dir / (experiment_name + suffix)

    else:

        output_path = checkpoint_dir / experiment_name / (scenario_id + suffix)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_path


# ============================================================
# Experiment-name helper
# ============================================================


def make_experiment_name(
    scale: str,
    *,
    variant: str | None = None,
    seed: int | None = None,
) -> str:
    """
    Build a stable experiment name.

    Example
    -------
    >>> make_experiment_name(
    ...     "small",
    ...     variant="full",
    ...     seed=42,
    ... )

    "small_full_seed42"
    """

    parts = [
        str(scale)
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
    ]

    if variant:

        parts.append(
            str(variant)
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        )

    if seed is not None:

        parts.append(f"seed{seed}")

    return "_".join(parts)


# ============================================================
# Public API
# ============================================================

__all__ = [
    # Paths
    "get_project_root",
    "ensure_directory",
    # Reproducibility
    "set_seed",
    "set_random_seed",
    # Device
    "get_device",
    "device_summary",
    # JSON / YAML
    "to_serializable",
    "load_json",
    "save_json",
    "load_yaml",
    "save_yaml",
    "deep_update",
    "load_config",
    # Scenarios
    "validate_nodes",
    "normalize_scenario",
    "normalize_scenarios",
    "load_scenarios",
    "save_scenarios",
    "infer_scale_name",
    "is_large_scale",
    # Metrics
    "percentage_improvement",
    "safe_mean",
    "moving_average",
    "flying_ratio",
    # Formatting
    "format_seconds",
    "format_energy",
    # PyTorch
    "count_parameters",
    "model_size_mb",
    # Observations
    "copy_observation",
    # Checkpoints / experiments
    "make_checkpoint_path",
    "make_experiment_name",
]
