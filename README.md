# Reinforcement_Learning_AI

# U-Boot V2: Q-Learning Agent for Submarine Navigation

## Project Description

This project implements a **Reinforcement Learning agent (Q-Learning)** capable of navigating a U-Boot submarine through a hostile ocean, avoiding sea mines and achieving its objective: intercepting an enemy convoy.

The environment is based on **FrozenLake-v1** from Gymnasium, but has been significantly modified through custom wrappers to simulate a realistic naval mission with fuel constraints, continuous observations, and a more informative reward scheme.

---

## Objectives

- Implement a Q-Learning agent from scratch capable of learning an optimal navigation policy.
- Design a custom environment with advanced mechanics (fuel management, continuous observations, dense rewards).
- Analyze the training process through success metrics and cumulative reward.
- Visualize agent behavior through mission animations.

---

## Project Structure

```
Reinforcement_Learning_AI/
├── agents/
│   └── q_agent.py          # Q-Learning agent with state space discretization
├── env/
│   ├── map_generator.py    # Procedural 8x8 map generator
│   ├── uboat_env.py        # Environment factory with applied wrappers
│   └── wrappers.py         # Custom wrappers (Obs, Fuel, Rew, Monitor)
├── utils/
│   ├── animation.py        # Animated mission visualization
│   └── training.py         # Training loop and metrics plotting
├── results/
│   └── v2_curves.png       # Generated training curves
├── main.py                 # Main execution script
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## Theoretical Foundations

### Q-Learning

The **Q-Learning** algorithm is a *model-free* reinforcement learning method that learns an action-value function \( Q(s, a) \) representing the expected utility of taking action \( a \) in state \( s \) and following an optimal policy thereafter.

The Q-table update follows the Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot [r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a)]$$

Where:
- \( \alpha \): Learning rate
- \( \gamma \): Discount factor
- \( r \): Immediate reward
- \( s' \): Next state

### State Space Discretization

Since the original environment provides continuous observations (4 normalized features), the agent uses **bin-based discretization** to convert the continuous space into a finite set of states representable in the Q-table.

### ε-Greedy Policy

The agent employs an **ε-greedy** exploration-exploitation policy with exponential decay:
- Initially explores randomly (\( \varepsilon = 1.0 \))
- Progressively exploits learned knowledge (\( \varepsilon \rightarrow 0.01 \))

---

## Environment Features

### Procedural Map
- 8×8 grid randomly generated with ~15% mine density
- Always guarantees a traversable path between start (S) and goal (G)
- Cells: **S** (Start), **F** (Free), **H** (Hole/Mine), **G** (Goal)

### Custom Wrappers

| Wrapper | Functionality |
|---------|--------------|
| `ObsWrapper` | Transforms discrete cell index into a continuous 4-feature vector: `[row_norm, col_norm, dist_goal_norm, dist_mine_norm]` |
| `FuelWrapper` | Introduces a 150-unit fuel limit. When fuel ≤ 20%, up/left actions are randomly redirected to down/right |
| `RewWrapper` | Replaces original binary reward with: +10 (success), -5 (mine), -0.01 (ordinary step) |
| `Monitor` | Records episode statistics: successes, mine sinkings, fuel depletion, timeouts |

---

## Agent Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `bins` | 8 | Number of intervals to discretize each feature |
| `actions` | 4 | Available actions: ↑ ↓ ← → |
| `lr` (α) | 0.1 | Learning rate |
| `gamma` (γ) | 0.99 | Discount factor |
| `eps` initial | 1.0 | Initial exploration rate |
| `eps_min` | 0.01 | Minimum exploration |
| `eps_decay` | 0.998 | Exponential decay per episode |

---

## Installation and Usage

### Prerequisites

- Python 3.8+
- pip

### Dependencies Installation

```bash
pip install gymnasium matplotlib numpy
```

### Execution

```bash
python main.py
```

The script will perform:
1. **Training**: 10,000 episodes with logging every 1,000 episodes
2. **Visualization**: Generate training curves in `results/v2_curves.png`
3. **Animation**: Display 6 missions of the trained agent

---

## Expected Results

During training, you will observe:
- **Increasing success rate**: From ~0% to stabilizing around 75-90%
- **Positive average reward**: The agent learns to avoid mines and manage fuel
- **ε decay**: Exploration decreases progressively, favoring exploitation

### Example Statistics

```
=== STATISTICS ===
  Convoy intercepted    : 7500  ( 75.0%)
  Sunk by mine          : 1500  ( 15.0%)
  Out of fuel           :  800  (  8.0%)
  Time out              :  200  (  2.0%)
  Average steps: 45.3
```

---

## Animation

The animation displays:
- Submarine movement (orange square with "U")
- Traveled trajectory (orange line)
- Status bars: fuel and torpedoes
- Final message: **"CONVOY SUNK!"** (success) or **"U-BOOT SUNK!"** (failure)

---

## Authors

- Iago Méndez García 
- Pablo Crespo de la Cruz 
- Carlos Segovia Dominguez 

---

