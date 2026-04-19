# Reinforcement Learning AI - Frozen Lake

## Project Description

This project implements a **reinforcement learning** environment using the classic **Frozen Lake** game from Gymnasium. 
The objective is to develop intelligent agents capable of navigating across a frozen lake to reach the goal without falling into holes.

## Project Structure

```
Reinforcement_Learning_AI/
├── main.py                 # Main entry point that executes the agent's mission
├── agents/                 # Agents module
│   ├── __init__.py
│   ├── random_agent.py     # Agent that performs random actions
│   └── q_agent.py          # Q-Learning agent (in development)
├── env/                    # Environment module
│   └── frozen_lake_env.py  # Configuration and creation of Frozen Lake environment
├── resources/              # Resources and configurations
│   └── config.py           # Project configuration parameters
├── requirements.txt        # Project dependencies
└── README.md               # Brief description 
```

## Implemented Features

### Version 1

- **Frozen Lake Environment**: Configuration of `FrozenLake-v1` environment with 4x4 map
  - Non-slippery surface (`is_slippery=False`)
  - Human render mode for real-time visualization

- **Random Agent (`RandomAgent`)**:
  - Basic implementation that randomly selects actions from the action space
  - Serves as a baseline for comparing future more sophisticated agents

- **Mission Execution**:
  - Main loop that executes the agent until completing the mission (reaching the goal) or failing (falling into a hole)
  - Step-by-step rendering with configurable pause between steps
  - Final mission result report

- **Centralized Configuration**:
  - Configurable parameters in `resources/config.py`
  - Pause between steps: 0.2 seconds

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script to start the agent's mission:

```bash
python main.py
```

The agent will perform random actions until reaching the goal or falling into a hole.

