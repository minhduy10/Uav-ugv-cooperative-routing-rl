# UAV-UGV Cooperative Routing with Reinforcement Learning

A reinforcement learning framework for cooperative UAV-UGV routing with
energy-aware action masking, expert-guided Behavioral Cloning, PPO
fine-tuning, Self-Imitation Learning, and heuristic / OR-Tools baselines.

The project studies how an Unmanned Aerial Vehicle (UAV) and an Unmanned
Ground Vehicle (UGV) can cooperate to visit a set of task locations while
minimizing mission makespan and respecting UAV battery constraints.

---

## Overview

The training pipeline consists of three main stages:

```text
Heuristic / OR-Tools Expert
            |
            v
   Behavioral Cloning
            |
            v
      PPO Fine-tuning
            |
            v
   Best Trajectory Buffer
            |
            v
 Self-Imitation Learning
```

The policy uses a Transformer encoder to process task-node features and
produces three decisions at every environment step:

```text
UAV target node
UGV target node
UGV operation mode
```

The action format is:

```python
[uav_target, ugv_target, mode]
```

where:

```text
mode = 0  -> UGV moves
mode = 1  -> UGV waits / UAV rendezvous or recharge
```

---

## Main Features

- Cooperative UAV-UGV routing environment built with Gymnasium
- Transformer-based actor-critic policy
- Dynamic energy-aware UAV action masking
- UAV battery and energy-consumption model
- UGV mobility model
- Mobile UAV recharge support
- Heuristic and optional OR-Tools expert planner
- Behavioral Cloning from expert demonstrations
- PPO fine-tuning
- Generalized Advantage Estimation (GAE)
- Self-Imitation Learning
- Small, Medium, and Large benchmark configurations
- Ablation studies
- Deterministic model evaluation
- Unit tests for physics, environment, and policy model

---

## Project Structure

```text
uav-ugv-routing-rl/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── configs/
│   ├── small.yaml
│   ├── medium.yaml
│   └── large.yaml
│
├── data/
│   └── evaluation_scenarios.json
│
├── src/
│   └── uav_ugv_rl/
│       ├── __init__.py
│       ├── physics.py
│       ├── heuristics.py
│       ├── environment.py
│       ├── model.py
│       ├── agent.py
│       ├── training.py
│       ├── ablation.py
│       └── utils.py
│
├── scripts/
│   ├── train.py
│   ├── run_ablation.py
│   └── evaluate.py
│
├── notebooks/
│   └── demo.ipynb
│
├── checkpoints/
│
├── results/
│   └── .gitkeep
│
└── tests/
    ├── test_environment.py
    ├── test_energy.py
    └── test_model.py
```

---

## Environment

Each scenario contains one depot and a set of task nodes.

The environment observation contains:

```python
{
    "uav_state": ...,
    "ugv_state": ...,
    "nodes_features": ...
}
```

### UAV state

```text
[x, y, remaining_energy]
```

Shape:

```text
(3,)
```

### UGV state

```text
[x, y]
```

Shape:

```text
(2,)
```

### Node features

Each node contains seven features:

```text
x
y
visited
relative_x_to_uav
relative_y_to_uav
relative_x_to_ugv
relative_y_to_ugv
```

Shape:

```text
(num_nodes, 7)
```

---

## Energy-Aware Action Masking

Before the policy selects a UAV destination, the environment checks whether
the UAV has enough remaining energy to:

1. fly from its current position to the candidate task, and
2. safely return to / rendezvous with the UGV.

The basic feasibility condition is:

```text
available_energy >= required_energy × safety_factor
```

A default 5% safety margin is used:

```text
safety_factor = 1.05
```

Unsafe UAV destinations are masked before the policy samples an action.

Because UAV position, UGV position, battery state, and visited tasks change
during the episode, the mask is recomputed dynamically at every step.

---

## Physical Model

Default vehicle parameters:

| Parameter | Value |
|---|---:|
| UAV speed | 10.0 m/s |
| UGV speed | 4.5 m/s |
| UAV battery capacity | 287.7 kJ |
| Recharge power | 5000 W |
| UAV hover power | 200 W |

The UAV power model is:

```text
P(v) = 0.0461v³ - 0.5834v² - 1.8761v + 229.6
```

and energy consumption per metre is:

```text
E/m = P(v) / v
```

---

## Transformer Policy

The policy architecture is:

```text
Node Features
     |
     v
Node Embedding
     |
     v
Transformer Encoder
     |
     +----------------------+
     |                      |
     v                      v
Graph Representation    Node Keys
     |
     + UAV State
     + UGV State
     + Scale Embedding
     |
     v
Global Context
     |
     +-------------------+-------------------+----------------+
     |                   |                   |                |
     v                   v                   v                v
UAV Pointer Head   UGV Pointer Head     Mode Head       Value Head
```

### Small / Medium

```text
Embedding dimension : 192
Attention heads     : 6
Transformer layers  : 3
```

### Large

```text
Embedding dimension : 256
Attention heads     : 8
Transformer layers  : 4
```

---

## Benchmark Scales

The standard experiment configurations are:

| Scale | Tasks | Map Size |
|---|---:|---:|
| Small | 30 | 16 km |
| Medium | 60 | 25 km |
| Large | 100 | 40 km |

Configuration files are located in:

```text
configs/
├── small.yaml
├── medium.yaml
└── large.yaml
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/uav-ugv-routing-rl.git
cd uav-ugv-routing-rl
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Training

### Small

```bash
python scripts/train.py --config configs/small.yaml
```

### Medium

```bash
python scripts/train.py --config configs/medium.yaml
```

### Large

```bash
python scripts/train.py --config configs/large.yaml
```

Training consists of:

```text
Phase 1 -> Behavioral Cloning
Phase 2 -> PPO Fine-tuning
Phase 3 -> Self-Imitation Learning
```

Checkpoints are stored in:

```text
checkpoints/
```

Training statistics are stored in:

```text
results/training_results.json
```

---

## Evaluation

Evaluate a trained scenario:

```bash
python scripts/evaluate.py \
    --scenario-id small_01
```

Use a specific checkpoint:

```bash
python scripts/evaluate.py \
    --scenario-id small_01 \
    --checkpoint checkpoints/small_01.pth
```

Evaluate only Medium scenarios:

```bash
python scripts/evaluate.py \
    --scale Medium
```

Evaluation results are written to:

```text
results/evaluation_results.json
```

Main reported metrics include:

```text
Makespan
Energy consumption
Success
Flying ratio
Flying steps
Riding steps
Improvement relative to expert baseline
```

---

## Ablation Study

Four standard configurations are provided:

```text
full
    BC + PPO + SIL

no_bc
    PPO + SIL

no_sil
    BC + PPO

pure_rl
    PPO only
```

Run all ablations:

```bash
python scripts/run_ablation.py
```

Run selected variants:

```bash
python scripts/run_ablation.py \
    --configs no_bc no_sil pure_rl
```

Run ablations only on Large scenarios:

```bash
python scripts/run_ablation.py \
    --scale Large
```

Results are written to:

```text
results/ablation_results.json
```

---

## Evaluation Scenario Format

`data/evaluation_scenarios.json` should contain a JSON list.

Example:

```json
[
  {
    "id": "small_01",
    "scale_name": "Small",
    "num_tasks": 30,
    "map_size": 16.0,
    "nodes_loc": [
      [1.6, 1.6],
      [2.4, 5.1],
      [7.2, 3.8]
    ]
  }
]
```

The first node is always the depot:

```text
nodes_loc[0] = depot
nodes_loc[1:] = tasks
```

The complete scenario must contain:

```text
num_tasks + 1
```

nodes.

---

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run a specific test module:

```bash
pytest tests/test_environment.py -v
```

```bash
pytest tests/test_energy.py -v
```

```bash
pytest tests/test_model.py -v
```

---

## Main Modules

### `physics.py`

Contains:

- UAV power model
- UGV power model
- battery capacity
- flight-energy calculations
- recharge calculations
- flight feasibility checks

### `environment.py`

Contains:

- Gymnasium UAV-UGV environment
- dynamic energy-aware action masks
- UAV / UGV movement
- recharge behavior
- reward calculation
- success and truncation logic

### `heuristics.py`

Contains:

- nearest-neighbour TSP
- 2-opt
- greedy set cover
- UAV sortie heuristic
- optional CP-SAT route optimization
- expert trajectory generation

### `model.py`

Contains:

- Transformer encoder
- UAV pointer policy
- UGV pointer policy
- mode policy
- critic network

### `agent.py`

Contains:

- action selection
- Behavioral Cloning
- PPO
- GAE
- Self-Imitation Learning
- checkpoint management

### `training.py`

Contains:

- expert trajectory collection
- RL trajectory collection
- BC / PPO / SIL training pipeline
- policy evaluation
- multi-scenario training

### `ablation.py`

Contains:

- Full model experiment
- No-BC experiment
- No-SIL experiment
- Pure-RL experiment
- aggregate ablation metrics

---

## Reproducibility

Experiment configuration files use a fixed random seed by default:

```yaml
seed: 42
```

For more reliable research comparisons, run multiple seeds and report
the mean and standard deviation across independent runs.

---

## License

This project is released under the MIT License.

See [LICENSE](LICENSE) for details.