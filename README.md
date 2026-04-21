# Reinforcement Learning AI - Frozen Lake

## Project Description

This project implements a **reinforcement learning** environment using the classic **Frozen Lake** game from Gymnasium.
The objective is to develop intelligent agents capable of navigating across a frozen lake to reach the goal without falling into holes.

## Project Structure

```
workspace/
├── main.py                 # Main entry point that executes the agent's mission
├── agents/                 # Agents module
│   ├── __init__.py
│   └── random_agent.py     # Agent that performs random actions
├── env/                    # Environment module
│   └── frozen_lake_env.py  # Frozen Lake environment configuration and creation
├── resources/              # Resources and configurations
│   └── config.py           # Project configuration parameters
├── requirements.txt        # Project dependencies
└── README.md               # Project description
```

## Implemented Features

### Version 1

- **Frozen Lake Environment**: Configuration of `FrozenLake-v1` environment with 4x4 map
  - Non-slippery surface (`is_slippery=False`)
  - Human rendering mode for real-time visualization

- **Random Agent (`RandomAgent`)**:
  - Basic implementation that randomly selects actions from the action space
  - Serves as a baseline for comparing future more sophisticated agents

- **Mission Execution**:
  - Main loop that runs the agent until completing the mission (reaching the goal) or failing (falling into a hole)
  - Step-by-step rendering with configurable pause between steps
  - Final mission result report

- **Centralized Configuration**:
  - Configurable parameters in `resources/config.py`:
    - `MAP_NAME`: "4x4"
    - `IS_SLIPPERY`: False
    - `RENDER_MODE`: "human"
    - `PAUSE_BETWEEN_STEPS`: 0.2 seconds

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- gymnasium==1.2.3
- numpy==2.4.4
- cloudpickle==3.1.2
- typing_extensions==4.15.0
- Farama-Notifications==0.0.4

## Usage

Run the main script to start the agent's mission:

```bash
python main.py
```

The agent will perform random actions until reaching the goal or falling into a hole. Upon completion, the number of steps taken and the mission result will be displayed.

## Configuration

You can modify the environment parameters by editing the `resources/config.py` file:

- `MAP_NAME`: Map size ("4x4", "8x8", etc.)
- `IS_SLIPPERY`: Whether the surface is slippery (True/False)
- `RENDER_MODE`: Rendering mode ("human", "rgb_array", etc.)
- `PAUSE_BETWEEN_STEPS`: Pause time between each step (in seconds)

## Authors

- Iago Mendez García 
- Pablo Crespo de la Cruz
- Carlos Segovia Domínguez
