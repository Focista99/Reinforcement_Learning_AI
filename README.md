# U-Boot V7: Hybrid Q-Learning + Fuzzy Logic in the Battle of the Atlantic

## Project Description

This project implements a **hybrid multi-agent Artificial Intelligence system** that combines **tabular Q-Learning** and **Fuzzy Logic** to control two cooperative U-Boot submarines in a simulation of the Battle of the Atlantic.

The environment is a custom grid-world: an Allied convoy departs from the American coast (left) and crosses the Atlantic eastward toward England (right), escorted by destroyers. Two U-Boots emerge from the southern edge of the map and must coordinate to intercept and sink as much of the convoy as possible.

The central contribution of this version is the integration of a **Fuzzy Logic Advisory Layer** on top of the Q-Learning agent. A `FuzzyAdvisor` runs two independent Mamdani inference systems — one for fire readiness and one for evasion urgency — whose outputs bias the Q-table values before every greedy decision. This hybrid architecture combines the long-term strategic learning of Q-Learning with the interpretable tactical reasoning of Fuzzy Logic.

---

## Objectives

- Design a cooperative multi-agent environment from scratch including convoy movement, torpedo mechanics, sonar detection, and escort AI.
- Implement a tabular Q-Learning agent with a 10-element discrete state space derived from an 11-feature continuous observation.
- Design and integrate a Mamdani Fuzzy Inference System that provides tactical advice (when to fire, when to evade) as a bias over Q-table action selection.
- Apply three-phase **curriculum learning** to progressively increase training difficulty.
- Analyse training convergence through reward, sinking rate, and win-rate metrics.
- Visualise learned behaviour with a full multi-panel matplotlib animation that shows sonar displays, fuzzy advisor scores, and a cumulative score timeline in real time.

---

## Project Structure

```
Reinforcement_Learning_AI/
├── agents/
│   ├── __init__.py
│   ├── q_agent.py          # UBoatAgent: Q-Learning + Fuzzy advisory bias
│   └── fuzzy_advisor.py    # FuzzyAdvisor: two Mamdani FIS (fire + evade)
├── env/
│   ├── __init__.py
│   ├── entities.py         # Ship and Submarine dataclasses
│   └── atlantic_env.py     # AtlanticEnv: step/reset/spawn/fire/observation
├── utils/
│   ├── __init__.py
│   ├── training.py         # train() loop and record_episode() helper
│   └── visualization.py    # make_ocean(), render_video(), sonar display
├── assets/                 # (optional) PNG sprites for ships and submarines
├── main.py                 # Entry point: 3-phase curriculum + rendering
├── requirements.txt
└── README.md
```

---

## Scenario

| Parameter | Value |
|-----------|-------|
| Grid | 20 × 36 cells |
| American coast | columns 0–1 (left, convoy origin) |
| English coast | columns 34–35 (right, convoy destination) |
| Convoy direction | West → East |
| U-Boot spawn | Bottom edge (row 19), ascending north |
| Max episode steps | 180 |
| Max ships on map | 12 |
| U-Boot movement | 2 cells per action (speed advantage) |
| Victory condition | Sink convoy ships worth ≥ 100 points |

### Ship Types

| Symbol | Type | Sinking reward | Steps per cell |
|--------|------|----------------|----------------|
| C | Cargo | +10 pts | 3 |
| T | Tanker | +25 pts | 4 |
| P | Troop transport | +40 pts | 3 |
| D | Destroyer | −60 pts if sunk | 3 |

The episode ends only when **both** submarines are destroyed or when the U-Boots reach the victory threshold — losing a single sub does not end the mission.

---

## Theoretical Foundations

### Q-Learning

**Q-Learning** is a model-free, off-policy reinforcement learning algorithm that learns an action-value function $Q(s, a)$ representing the expected cumulative reward of taking action $a$ in state $s$ and following an optimal policy thereafter.

The update rule is derived from the Bellman optimality equation:

$$Q(s,a) \;\leftarrow\; Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

| Symbol | Meaning |
|--------|---------|
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $r$ | Immediate reward |
| $s'$ | Next state |

An **ε-greedy** policy balances exploration and exploitation:
$$\varepsilon_{t+1} = \max(\varepsilon_{\min},\; \varepsilon_t \cdot d)$$

### Fuzzy Logic

**Fuzzy Logic** extends classical Boolean logic by allowing partial membership in sets. A variable can simultaneously belong to multiple fuzzy sets with degrees of membership in $[0, 1]$, modelling linguistic concepts such as *close*, *fast*, or *urgent*.

A **Mamdani Fuzzy Inference System (FIS)** operates in four stages:

1. **Fuzzification** — map crisp inputs to fuzzy membership degrees.
2. **Rule evaluation** — apply fuzzy IF–THEN rules using min (AND) and max (OR) operators.
3. **Aggregation** — combine the outputs of all fired rules into a single fuzzy set.
4. **Defuzzification** — compute a crisp output value via the **centroid method**:

$$y^* = \frac{\int y \cdot \mu(y)\, dy}{\int \mu(y)\, dy}$$

### Hybrid Q-Fuzzy Architecture

The two paradigms are combined as follows:

```
Observation (11 features)
        │
        ├──► Discretise ──► Q-table lookup ──► Q-values [6 actions]
        │                                             │
        └──► FuzzyAdvisor                             │
               │                                     │
               ├── Fire Readiness FIS ──► fire_score ─► boost Q[FIRE]
               │
               └── Evasion Urgency FIS ──► evade_score ─► boost Q[evade_dir]
                                                     │
                                              argmax ──► action
```

The fuzzy boost scales with the **spread of current Q-values**:

$$\Delta Q[\text{FIRE}] = \frac{\text{fire\_score}}{10} \cdot \text{spread}(Q) \cdot 0.45$$

This guarantees that in well-explored states (large Q-value spread) the fuzzy advice has minimal effect, while in uncertain states (all Q-values near zero, spread ≈ 1) the fuzzy system provides meaningful tactical guidance.

---

## Fuzzy Inference Systems

### System 1 — Fire Readiness

**Purpose:** determine how urgently the submarine should fire a torpedo right now.

**Input variables:**

| Variable | Universe | Linguistic terms |
|----------|----------|-----------------|
| `dist_target` | [0, 1] | `in_range`, `near`, `far` |
| `target_value` | [0, 1] | `low`, `medium`, `high` |
| `torpedoes` | [0, 1] | `scarce`, `ok`, `plenty` |

> `dist_target` = Chebyshev distance / `SONAR_RANGE`.  
> The torpedo-range boundary maps to ≈ 0.71 (5 / 7).  
> `target_value` = ship type / 3 → cargo=0, tanker=0.33, transport=0.67.

**Output variable:**

| Variable | Universe | Linguistic terms |
|----------|----------|-----------------|
| `fire_score` | [0, 10] | `hold`, `prepare`, `shoot` |

**Rule base (9 rules):**

| Condition | Output |
|-----------|--------|
| in_range AND high value AND plenty torps | shoot |
| in_range AND high value AND ok torps | shoot |
| in_range AND medium value AND plenty torps | shoot |
| in_range AND medium value AND ok torps | prepare |
| in_range AND low value AND plenty torps | prepare |
| in_range AND low value AND ok torps | hold |
| torpedoes scarce (any) | hold |
| target near (outside range) | hold |
| target far | hold |

---

### System 2 — Evasion Urgency

**Purpose:** determine how urgently the submarine should evade a nearby destroyer.

**Input variables:**

| Variable | Universe | Linguistic terms |
|----------|----------|-----------------|
| `dest_dist` | [0, 1] | `critical`, `close`, `safe` |
| `dest_dir` | [−1, 1] | `from_north`, `lateral`, `from_south` |

> `dest_dist` = Chebyshev distance / `SONAR_RANGE`. Small values mean the destroyer is close.  
> `dest_dir` = row component of direction: negative means the destroyer is to the north of the sub.

**Output variable:**

| Variable | Universe | Linguistic terms |
|----------|----------|-----------------|
| `evade_urgency` | [0, 10] | `none`, `low`, `high` |

**Rule base (5 rules):**

| Condition | Output |
|-----------|--------|
| critical distance | high |
| close AND from north | high |
| close AND lateral | low |
| close AND from south | low |
| safe distance | none |

When `evade_score > 4`, the agent boosts the appropriate escape direction: south if the destroyer approaches from the north, north otherwise.

---

## Observation Space (11 features)

Each submarine receives an independent observation vector:

| Index | Feature | Range |
|-------|---------|-------|
| 0 | Sub row (normalised) | [0, 1] |
| 1 | Sub col (normalised within open sea) | [0, 1] |
| 2 | Remaining torpedoes (normalised) | [0, 1] |
| 3 | Distance to highest-value target in sonar | [0, 1] |
| 4 | Direction to target — row component | [−1, 1] |
| 5 | Direction to target — col component | [−1, 1] |
| 6 | Target type (normalised) | [0, 1] |
| 7 | Distance to nearest destroyer in sonar | [0, 1] |
| 8 | Direction to destroyer — row component | [−1, 1] |
| 9 | Partner submarine distance (normalised) | [0, 1] |
| 10 | Partner has better shot at target? | {0, 1} |

### State Discretisation (Q-table keys)

The 11 continuous features are mapped to a 10-element tuple for the Q-table:

| Component | Bins | Description |
|-----------|------|-------------|
| s0 | 4 | Row zone |
| s1 | 4 | Column zone |
| s2 | 3 | Torpedo level (full / medium / low) |
| s3 | 2 | Target within torpedo range? |
| s4 | 9 | Bearing to target (8 compass + none) |
| s5 | 4 | Best target type |
| s6 | 2 | Destroyer within 3 cells? |
| s7 | 2 | Destroyer to the north? |
| s8 | 2 | Partner too close? |
| s9 | 2 | Partner has priority on target? |

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
| No torpedoes remaining | −1 |
| Destroyed by destroyer | −80 + episode continues |
| Submarines too close together | −1.5 × (MIN\_SEP − dist) each |
| Victory bonus (both subs, on reaching threshold) | +60 each |

---

## Curriculum Training (3 Phases)

| Phase | Episodes | Destroyers | Purpose |
|-------|----------|------------|---------|
| 1 | 1 500 | No | Learn to intercept and fire at the eastbound convoy |
| 2 | 2 500 | Yes | Learn to evade escort while hunting; fuzzy evade guidance active |
| 3 | 1 000 | Yes | Fine-tune coordination at low learning rate and low ε |

### Agent Hyperparameters

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Learning rate α | 0.22 | 0.15 | 0.07 |
| Discount factor γ | 0.97 | 0.97 | 0.97 |
| Initial ε | 1.00 | 0.45 | 0.15 |
| Min ε | 0.05 | 0.05 | 0.04 |
| ε decay | 0.9991 | 0.9993 | 0.9997 |
| Fuzzy advisory | active | active | active |

---

## Cooperative Mechanics

Three mechanisms promote inter-submarine coordination:

1. **Zone division** — U-1 is seeded in the northern half, U-2 in the southern half. A positioning bonus rewards each sub for staying near its zone centre (ascending toward convoy lanes).
2. **Target exclusivity** — before firing, each sub checks whether its partner has a shorter path to the same target and yields priority if so, preventing torpedo overlap.
3. **Separation penalty** — a proportional penalty is applied to both subs whenever their Chebyshev distance falls below `MIN_SUB_SEP = 4` cells, encouraging map coverage.

---

## Environment Mechanics

### Destroyer AI

Three-tier behaviour:
1. **Intercept** — when a sub is detected within `SONAR_RANGE + 2` cells, the destroyer charges directly toward it.
2. **Protect** — positions itself 3 cells **east** of the convoy centroid (between convoy and England), oscillating vertically for wider coverage.
3. **Advance** — sails east with the convoy when no sub is in range.

### Key Ranges

| System | Range (Chebyshev) |
|--------|------------------|
| Sonar | 7 cells |
| Torpedo | 5 cells |
| Min sub separation | 4 cells |
| Sub movement | 2 cells / action |

---

## PNG Sprite Support

Custom sprites can be used for any entity. Set the paths in `constants.py`:

```python
SPRITE_PATHS = {
    "uboat_0":   "assets/uboat_orange.png",
    "uboat_1":   "assets/uboat_violet.png",
    "cargo":     "assets/cargo.png",
    "tanker":    "assets/tanker.png",
    "transport": "assets/transport.png",
    "destroyer": "assets/destroyer.png",
}
```

Recommended sprite size: **64 × 64 px**, transparent PNG background. If a file is missing the renderer falls back to the default shape-based drawing automatically.

---

## Installation and Usage

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy
matplotlib
scikit-fuzzy
networkx
```

> `networkx` is required internally by `scikit-fuzzy` to build the rule dependency graph.

### Run

```bash
python main.py
```

The script executes:
1. **Phase 1** — 1 500 episodes without destroyers. Logs every 500 episodes.
2. **Phase 2** — 2 500 episodes with destroyers. Fuzzy evasion guidance becomes critical.
3. **Phase 3** — 1 000 fine-tuning episodes.
4. **Record** — one greedy episode with frame-by-frame fuzzy score capture.
5. **Render** — animated visualisation saved to `uboot_v7.mp4` and displayed interactively.

---

## Visualisation

The animation renders a five-panel layout:

| Panel | Content |
|-------|---------|
| Main map | Ocean grid, coastlines, ship icons, submarine trails, torpedo lines, explosion markers |
| Sonar U-1 | Amber rotating sweep, range rings, contact blips colour-coded by ship type |
| Sonar U-2 | Same as above for the second submarine |
| Info panel | Position, torpedo bar, last action, alive status, **live fuzzy fire/evade bars** per sub, mission goal progress bar |
| Score timeline | Cumulative episode reward plotted frame by frame |

The **fuzzy bars** in the info panel display the real-time output of both FIS systems (fire readiness and evasion urgency, 0–10) for each submarine, making the advisory layer directly observable.

---

## Expected Results

After completing all three training phases:

- **Phase 1**: subs rapidly learn to move north and fire at unescorted targets. Average sinkings ≥ 3 per episode by ep 1 000.
- **Phase 2**: initial dip in sinking rate as destroyer avoidance is learned. Fuzzy evasion guidance accelerates recovery. Win rate begins to appear.
- **Phase 3**: convergence. Survival rate increases noticeably; win rate stabilises.

### Example Console Output

```
[ PHASE 1 ]  Intercept convoy without escort  (1500 eps)
  Ep   500  Reward   19.2  Sunk 2.9  Win  0.0%  ε=0.6034  Q-states 1312
  Ep  1000  Reward   33.7  Sunk 4.1  Win  4.2%  ε=0.3638  Q-states 2680
  Ep  1500  Reward   41.2  Sunk 5.0  Win 12.8%  ε=0.2193  Q-states 3240

[ PHASE 2 ]  Evade destroyers and hunt convoy  (2500 eps)
  Ep   500  Reward   22.1  Sunk 3.4  Win  8.5%  ε=0.3499  Q-states 4810
  ...

[ PHASE 3 ]  Coordinated refinement  (1000 eps)
  ...

[ REC ]  Recording greedy episode...
  Steps 163  |  Sunk 7  |  Score +198.4  |  VICTORY
    Transport : x2  (+80 pts)
    Tanker    : x3  (+75 pts)
    Cargo     : x2  (+20 pts)
```

---

## Known Limitations and Future Work

| Limitation | Potential improvement |
|------------|----------------------|
| Shared Q-table for both subs | Independent agents with a communication channel |
| Tabular Q-Learning does not generalise | Replace with DQN or PPO for continuous state handling |
| Fixed convoy spawn column | Randomise convoy routes; richer observation needed |
| Fuzzy rules hand-crafted | Learn membership function parameters via gradient descent (neuro-fuzzy) |
| No MP4 if ffmpeg unavailable | Add Pillow-based GIF fallback |

---

## Authors

- Iago Méndez García
- Pablo Crespo de la Cruz
- Carlos Segovia Domínguez

---
