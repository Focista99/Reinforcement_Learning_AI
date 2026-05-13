# U-Boot V7: Cooperative Q-Learning in the Battle of the Atlantic

## Project Description

This project implements a **multi-agent Reinforcement Learning system** based on **tabular Q-Learning** in which two cooperative U-Boot submarines learn to intercept an Allied convoy crossing the Atlantic Ocean.

The environment is a custom grid-world simulating the Battle of the Atlantic: the convoy departs from the American coast (right) and sails west toward the English coast (left), escorted by destroyers. The two submarines emerge from the southern edge of the map and must coordinate to maximise their combined score — sinking high-value targets while evading destroyer patrols.

---

## Objectives

- Design a multi-agent cooperative environment from scratch, including ship movement, torpedo mechanics, sonar detection and escort AI.
- Implement a shared tabular Q-Learning agent capable of coordinating two submarines through a rich 11-feature observation space.
- Apply **curriculum learning** across three training phases of increasing difficulty.
- Analyse convergence through score and sinking-rate metrics.
- Visualise learned behaviour with a full-featured matplotlib animation.

---

## Project Structure

```
Reinforcement_Learning_AI/
├── agents/
│   ├── __init__.py
│   └── q_agent.py          # UBoatAgent: tabular Q-Learning with shared policy
├── env/
│   ├── __init__.py
│   ├── entities.py         # Ship and Submarine dataclasses
│   └── atlantic_env.py     # AtlanticEnv: step/reset/spawn/fire/observation logic
├── utils/
│   ├── __init__.py
│   ├── training.py         # train() loop and record_episode() helper
│   └── visualization.py    # make_ocean() texture and render_video() animation
├── results/
│   └── (training curves and recorded episodes saved here)
├── main.py                 # Entry point: 3-phase curriculum + rendering
├── requirements.txt        # Project dependencies
└── README.md
```

---

## Scenario

| Parameter | Value |
|-----------|-------|
| Grid | 20 × 36 cells |
| English coast (left) | columns 0–1 |
| American coast (right) | columns 34–35 |
| Open Atlantic | columns 2–33 |
| Convoy direction | East → West |
| U-Boot spawn | Bottom edge (row 19), ascending |
| Max steps per episode | 180 |
| Max ships on map | 12 |

### Ship Types

| Symbol | Type | Reward | Speed (steps/move) |
|--------|------|--------|--------------------|
| C | Cargo ship | +10 | 3 |
| T | Tanker | +25 | 4 |
| P | Troop transport | +40 | 3 |
| D | Destroyer | −60 (if sunk by sub) | 2 |

A destroyer that reaches the same cell as a submarine **sinks it**, ending the episode.

---

## Theoretical Foundations

### Q-Learning

**Q-Learning** is a model-free, off-policy reinforcement learning algorithm that learns an action-value function $Q(s, a)$ representing the expected cumulative reward of taking action $a$ in state $s$ and following an optimal policy thereafter.

The update rule follows the Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

Where:
- $\alpha$ — learning rate
- $\gamma$ — discount factor
- $r$ — immediate reward
- $s'$ — next state

### Observation Space (11 features)

Each submarine receives an independent 11-dimensional continuous observation:

| Index | Feature | Range |
|-------|---------|-------|
| 0 | Submarine row (normalised) | [0, 1] |
| 1 | Submarine column (normalised within sea) | [0, 1] |
| 2 | Remaining torpedoes (normalised) | [0, 1] |
| 3 | Distance to highest-value target in sonar | [0, 1] |
| 4 | Direction to target — row component | [−1, 1] |
| 5 | Direction to target — col component | [−1, 1] |
| 6 | Target type (normalised) | [0, 1] |
| 7 | Distance to nearest destroyer in sonar | [0, 1] |
| 8 | Destroyer direction — row component | [−1, 1] |
| 9 | Partner submarine distance (normalised) | [0, 1] |
| 10 | Partner has better shot at target? | {0, 1} |

### State Space Discretisation

The continuous observation is converted to a 10-element discrete tuple for the Q-table:

| Component | Bins | Description |
|-----------|------|-------------|
| s0 | 4 | Row zone |
| s1 | 4 | Column zone |
| s2 | 3 | Torpedo level (full / medium / low) |
| s3 | 2 | Target within torpedo range? |
| s4 | 9 | Bearing to target (8 compass directions + none) |
| s5 | 4 | Best target type |
| s6 | 2 | Destroyer within 3 cells? |
| s7 | 2 | Destroyer to the north? |
| s8 | 2 | Partner too close? |
| s9 | 2 | Partner has priority on target? |

### ε-Greedy Policy

The agent uses an **ε-greedy** exploration-exploitation policy with exponential decay:

$$\varepsilon_{t+1} = \max(\varepsilon_{\min},\ \varepsilon_t \cdot d)$$

Both submarines share a single Q-table. This keeps the state space compact and allows knowledge transfer between agents while still receiving independent observations.

---

## Reward Design

| Event | Reward |
|-------|--------|
| Step cost | −0.05 |
| Positioning bonus (ascending toward convoy) | up to +0.02 |
| Sinking a cargo ship | +10 |
| Sinking a tanker | +25 |
| Sinking a troop transport | +40 |
| Wasted torpedo (no target in range) | −2 |
| Sub destroyed by destroyer | −80 + episode end |
| Submarines too close together | −1.5 × (MIN\_SEP − dist) each |

---

## Curriculum Training (3 Phases)

Training uses curriculum learning to progressively increase difficulty:

| Phase | Episodes | Destroyers | Purpose |
|-------|----------|------------|---------|
| 1 | 1 500 | No | Learn to intercept and fire at convoy |
| 2 | 2 500 | Yes | Learn to evade escort and select targets |
| 3 | 1 000 | Yes | Fine-tune coordination at low exploration |

### Agent Hyperparameters

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Learning rate α | 0.22 | 0.15 | 0.07 |
| Discount factor γ | 0.97 | 0.97 | 0.97 |
| Initial ε | 1.00 | 0.45 | 0.15 |
| Min ε | 0.05 | 0.05 | 0.04 |
| ε decay | 0.9991 | 0.9993 | 0.9997 |

---

## Cooperative Mechanics

Two behaviours promote coordination without explicit communication:

1. **Zone division**: U-1 is assigned the northern patrol zone, U-2 the southern zone. The positioning bonus guides each submarine toward its zone centre.
2. **Target exclusivity**: When firing, a submarine checks whether its partner has a shorter path to the same target and yields if so, avoiding wasted torpedoes.
3. **Separation penalty**: If both submarines are within `MIN_SUB_SEP = 4` cells (Chebyshev distance) of each other, both receive a proportional penalty, encouraging them to spread across the map.

---

## Environment Mechanics

### Destroyer AI

The escort destroyer uses a three-tier behaviour:

1. **Intercept mode** (sub detected within sonar + 2 cells): moves directly toward the nearest submarine.
2. **Convoy protection** (sub not yet detected): positions itself 3 columns ahead of the convoy centroid, oscillating vertically to cover a wider band.
3. **Advance**: moves west with the convoy when no submarine is in range.

### Sonar & Torpedo Ranges

| System | Range (Chebyshev distance) |
|--------|---------------------------|
| Sonar | 7 cells |
| Torpedo | 4 cells |

---

## Installation and Usage

### Prerequisites

- Python 3.9+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy
matplotlib
```

### Run

```bash
python main.py
```

The script will:

1. **Phase 1** — Train 1 500 episodes without destroyers, logging every 500.
2. **Phase 2** — Train 2 500 episodes with destroyers, reusing the learned Q-table.
3. **Phase 3** — Fine-tune for 1 000 episodes at low learning rate and low ε.
4. **Record** — Run one near-greedy episode and capture all frames.
5. **Render** — Display the animated mission with sonar screens, score timeline and tactical panel.

---

## Animation

The visualisation renders a multi-panel display:

- **Main map**: Atlantic grid with ocean texture, coastlines, ship icons, submarine trails, torpedo fire lines and explosion markers.
- **Sonar screens**: One per submarine — rotating sweep, contact blips colour-coded by ship type, range rings.
- **Tactical panel**: Position, torpedo bar, last action and alive status for each U-Boot; per-type sinking count and total score.
- **Score timeline**: Cumulative episode reward plotted frame by frame.

---

## Expected Results

After completing all three training phases you should observe:

- **Phase 1**: Agent quickly learns to move north and fire at unescorted convoy. Average sinkings ≥ 3 per episode by episode 1 000.
- **Phase 2**: Sinking rate drops initially (destroyer avoidance cost), then recovers as evasion is learned. Q-table grows to several thousand states.
- **Phase 3**: Convergence — low variance in episode score, submarine survival rate increases noticeably.

### Example Console Output

```
[ PHASE 1 ]  Intercept convoy without escort  (1500 eps)
  Ep   500  Reward   18.43  Sunk 2.81  ε=0.6034  Q-states 1247
  Ep  1000  Reward   31.12  Sunk 4.05  ε=0.3638  Q-states 2541
  Ep  1500  Reward   38.67  Sunk 4.89  ε=0.2193  Q-states 3102

[ PHASE 2 ]  Evade destroyers and hunt convoy  (2500 eps)
  ...

[ PHASE 3 ]  Coordinated refinement  (1000 eps)
  ...

[ REC ]  Recording greedy episode...
  Steps 147  |  Sunk 6  |  Score +183.4
    Troop transport : x2  (+80pts)
    Tanker          : x2  (+50pts)
    Cargo ship      : x2  (+20pts)
```

---

## Known Limitations and Future Work

- **Shared Q-table**: both submarines use a single policy. Independent agents with a communication channel could improve specialisation.
- **Tabular Q-Learning**: scales poorly if the observation space is expanded. A DQN or PPO approach would generalise better.
- **Deterministic spawns**: convoy always spawns at the American coast column. Randomising convoy routes would require a richer state representation.
- **No MP4 export**: the current renderer displays the animation interactively; `ani.save()` with an ffmpeg writer should be wired up for offline review.

---

## Authors

- Iago Méndez García
- Pablo Crespo de la Cruz
- Carlos Segovia Domínguez

---
