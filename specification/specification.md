# Floor Field Cellular Automata Simulation Specification

## 1. Overview

This specification defines a **Floor Field Cellular Automata (FFCA)** simulation for modeling pedestrian dynamics in constrained environments (supermarket aisles, train cabins). The system simulates emergent behaviors such as lane formation, clogging, and stop-and-go waves.

### 1.1 Domain

- **Primary Application**: Pedestrian flow simulation in narrow corridors
- **Use Cases**: Supermarket aisles, train cabin boarding, evacuation scenarios
- **Model Type**: 2D discrete-time cellular automata with floor fields

### 1.2 Expected Emergent Behaviors

| Behavior | Description |
|----------|-------------|
| Lane Formation | Self-organization of opposing pedestrian streams into lanes |
| Faster-is-Slower | Increased agent urgency leads to decreased throughput |
| Arching/Clogging | Semi-circular blockages at bottlenecks and hotspots |
| Stop-and-Go Waves | Backward-propagating waves of stopping in congested flow |

---

## 2. Functional Requirements

### 2.1 Core Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | Implement 2D discrete-time cellular automata model |
| FR-2 | Support static floor field (walls, goals, hotspots) |
| FR-3 | Support dynamic floor field (pheromone trails) |
| FR-4 | Model agent movement with conflict resolution |
| FR-5 | Extract simulation state at each discrete time step |

### 2.2 CLI Requirements

| ID | Requirement |
|----|-------------|
| CLI-1 | Single terminal command to start simulation |
| CLI-2 | Accept configuration file path as parameter |
| CLI-3 | Accept output directory as parameter |
| CLI-4 | Toggle CSV export (default: ON) |
| CLI-5 | Toggle final state snapshot (default: ON) |
| CLI-6 | Toggle animation GIF export (default: OFF) |
| CLI-7 | Toggle stdout summary report (default: ON, `--quiet` to disable) |

### 2.3 Export Requirements

| ID | Requirement |
|----|-------------|
| EXP-1 | CSV log: `(step, agent_id, x, y, state)` per time step |
| EXP-2 | Snapshot: PNG image of final grid state |
| EXP-3 | Animation: GIF of simulation progression |
| EXP-4 | Summary: Text report with throughput metrics |

---

## 3. Architecture

The system follows a **Model-View-Controller (MVC)** pattern adapted for scientific simulation.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Entry Point                          │
│                         (main.py)                               │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Configuration Loader                        │
│                        (config.py)                              │
│         - YAML/JSON parsing                                     │
│         - Dataclass validation                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SIMULATION ENGINE                           │
│                       (Controller)                              │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  for step in range(max_steps):                          │  │
│    │      state = engine.step()                              │  │
│    │      observers.notify(state)                            │  │
│    └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     MODEL       │  │     EXPORT      │  │   VISUALIZER    │
│   (model/)      │  │    (io/)        │  │    (io/)        │
│  - GridMap      │  │  - CSVWriter    │  │  - Snapshot     │
│  - Agent        │  │  - StateRecorder│  │  - GIF Builder  │
│  - FloorField   │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 4. Module Specifications

### 4.1 Project Structure

```
floor_field_ca/
├── main.py                 # CLI entry point
├── config.py               # Configuration dataclasses and loader
├── model/
│   ├── __init__.py
│   ├── grid.py             # GridMap class
│   ├── agent.py            # Agent class
│   ├── floor_field.py      # StaticField, DynamicField classes
│   └── engine.py           # SimulationEngine class
├── io/
│   ├── __init__.py
│   ├── csv_writer.py       # CSV export
│   ├── visualizer.py       # Snapshot and GIF generation
│   └── reporter.py         # Summary report generation
└── configs/
    ├── supermarket.yaml    # Example: supermarket layout
    └── train_cabin.yaml    # Example: train cabin layout
```

### 4.2 Configuration Module (`config.py`)

```python
@dataclass
class SimulationConfig:
    # Grid parameters
    grid_width: int
    grid_height: int

    # Simulation parameters
    max_steps: int
    agent_count: int

    # Floor field parameters
    static_field_strength: float  # kS: sensitivity to static field
    dynamic_field_strength: float # kD: sensitivity to dynamic field
    diffusion_rate: float         # alpha: pheromone diffusion rate
    decay_rate: float             # delta: pheromone decay rate

    # Agent behavior
    friends_probability: float    # Probability agents form groups
    patience: int                 # Steps before agent gives up waiting

    # Layout definition
    walls: List[Tuple[int, int]]        # Wall cell coordinates
    hotspots: List[Tuple[int, int]]     # Goal/attraction points
    spawn_zones: List[Tuple[int, int]]  # Agent entry points

    # Export flags
    csv_enabled: bool = True
    snapshot_enabled: bool = True
    gif_enabled: bool = False
    quiet: bool = False
```

### 4.3 Grid Module (`model/grid.py`)

```python
class GridMap:
    """
    Manages the 2D simulation environment with multiple layers.

    Attributes:
        width: int - Grid width in cells
        height: int - Grid height in cells
        static_field: np.ndarray - Distance-to-goal gradient (shape: height x width)
        dynamic_field: np.ndarray - Pheromone concentration (shape: height x width)
        occupancy: np.ndarray - Agent presence (0=empty, agent_id otherwise)
        walls: np.ndarray - Boolean mask of impassable cells

    Methods:
        is_walkable(x, y) -> bool
        is_occupied(x, y) -> bool
        get_neighbors(x, y) -> List[Tuple[int, int]]
        place_agent(agent_id, x, y) -> None
        remove_agent(x, y) -> None
        move_agent(agent_id, from_pos, to_pos) -> None
    """
```

### 4.4 Floor Field Module (`model/floor_field.py`)

```python
class StaticField:
    """
    Pre-computed distance field guiding agents toward goals.

    Methods:
        compute(grid: GridMap, goals: List[Tuple[int,int]]) -> np.ndarray
            Computes distance gradient using BFS/Dijkstra from goals

        get_value(x, y) -> float
            Returns attractiveness value at position
    """

class DynamicField:
    """
    Time-varying pheromone field for collective behavior.

    Attributes:
        field: np.ndarray - Current pheromone concentrations
        diffusion_rate: float - Alpha parameter (0.0-1.0)
        decay_rate: float - Delta parameter (0.0-1.0)

    Methods:
        deposit(x, y, amount: float) -> None
            Agent leaves pheromone trace at position

        update() -> None
            Apply diffusion (3x3 convolution) and decay

        get_value(x, y) -> float
            Returns pheromone concentration at position
    """
```

### 4.5 Agent Module (`model/agent.py`)

```python
class AgentState(Enum):
    MOVING = "moving"
    WAITING = "waiting"
    SHOPPING = "shopping"  # At hotspot
    EXITED = "exited"

class Agent:
    """
    Individual pedestrian entity.

    Attributes:
        id: int - Unique identifier
        position: Tuple[int, int] - Current (x, y) coordinates
        state: AgentState - Current behavioral state
        group_id: Optional[int] - Social group membership
        goal: Tuple[int, int] - Target destination
        patience: int - Remaining wait tolerance

    Methods:
        calculate_transition_probabilities(
            neighbors: List[Tuple[int, int]],
            static_field: StaticField,
            dynamic_field: DynamicField,
            occupied: Set[Tuple[int, int]]
        ) -> Dict[Tuple[int, int], float]
            Returns probability distribution over possible moves
            Formula: P(i→j) ∝ exp(kS * S_j + kD * D_j) * (1 - n_j)
            where n_j = 1 if cell j is occupied, 0 otherwise

        decide_next_move(...) -> Tuple[int, int]
            Samples from transition probabilities to select next cell
    """
```

### 4.6 Simulation Engine (`model/engine.py`)

```python
class SimulationEngine:
    """
    Orchestrates the discrete-time simulation loop.

    Attributes:
        grid: GridMap
        agents: List[Agent]
        static_field: StaticField
        dynamic_field: DynamicField
        current_step: int

    Methods:
        initialize(config: SimulationConfig) -> None
            Set up grid, fields, and spawn agents

        step() -> SimulationState
            Execute one discrete time step:
            1. Calculate desired moves for all agents (parallel update)
            2. Resolve conflicts (random priority or friction-based)
            3. Execute valid moves
            4. Update agent states
            5. Deposit pheromones
            6. Diffuse and decay dynamic field
            7. Return current state snapshot

        is_finished() -> bool
            Check termination conditions (max steps, all agents exited)
```

### 4.7 State Snapshot (`model/state.py`)

```python
@dataclass
class AgentSnapshot:
    agent_id: int
    x: int
    y: int
    state: str

@dataclass
class SimulationState:
    step: int
    agents: List[AgentSnapshot]
    grid_occupancy: np.ndarray
    dynamic_field: np.ndarray
    metrics: Dict[str, float]  # throughput, density, etc.
```

### 4.8 CSV Writer (`io/csv_writer.py`)

```python
class CSVWriter:
    """
    Exports simulation data to CSV format.

    Output format:
        step,agent_id,x,y,state
        0,1,5,10,moving
        0,2,6,10,waiting
        1,1,5,9,moving
        ...

    Methods:
        __init__(output_path: str)
        append(state: SimulationState) -> None
        close() -> None
    """
```

### 4.9 Visualizer (`io/visualizer.py`)

```python
class Visualizer:
    """
    Generates visual outputs using matplotlib.

    Methods:
        __init__(config: SimulationConfig)

        buffer_frame(state: SimulationState) -> None
            Store frame for GIF generation

        save_snapshot(state: SimulationState, output_path: str) -> None
            Save single PNG image showing:
            - Grid layout (walls, hotspots)
            - Agent positions (color-coded by state)
            - Dynamic field heatmap overlay

        generate_gif(output_path: str, fps: int = 10) -> None
            Compile buffered frames into animated GIF
    """
```

### 4.10 Reporter (`io/reporter.py`)

```python
class Reporter:
    """
    Generates summary statistics.

    Methods:
        __init__()

        update(state: SimulationState) -> None
            Accumulate metrics per step

        generate_summary() -> str
            Returns formatted text report:
            - Total simulation steps
            - Agents completed / total
            - Average throughput (agents/step)
            - Average travel time
            - Peak density observed
            - Clogging events detected
    """
```

---

## 5. CLI Interface

### 5.1 Command Syntax

```bash
python main.py --config <path> [options]
```

### 5.2 Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--config` | path | Yes | - | Path to YAML/JSON configuration file |
| `--steps` | int | No | from config | Override max simulation steps |
| `--out-dir` | path | No | `./output` | Output directory for exports |
| `--csv` | flag | No | ON | Enable CSV export |
| `--no-csv` | flag | No | - | Disable CSV export |
| `--snapshot` | flag | No | ON | Enable final snapshot |
| `--no-snapshot` | flag | No | - | Disable final snapshot |
| `--gif` | flag | No | OFF | Enable GIF animation |
| `--quiet` | flag | No | OFF | Suppress stdout output |
| `--seed` | int | No | None | Random seed for reproducibility |

### 5.3 Example Commands

```bash
# Basic run with defaults
python main.py --config configs/supermarket.yaml

# Full export with animation
python main.py --config configs/train_cabin.yaml --gif --out-dir results/

# Quiet mode for batch processing
python main.py --config configs/test.yaml --no-csv --no-snapshot --quiet

# Reproducible run
python main.py --config configs/supermarket.yaml --seed 42
```

---

## 6. Configuration File Format

### 6.1 YAML Schema

```yaml
# Grid dimensions
grid:
  width: 50
  height: 20

# Simulation parameters
simulation:
  max_steps: 500
  agent_count: 100

# Floor field parameters (Burstedde model)
floor_field:
  static_strength: 3.0    # kS
  dynamic_strength: 1.0   # kD
  diffusion_rate: 0.3     # alpha
  decay_rate: 0.1         # delta

# Agent behavior
agents:
  friends_probability: 0.2
  patience: 50
  speed: 1  # cells per step

# Layout definition
layout:
  walls:
    # Define walls as rectangles or point lists
    - type: rectangle
      x: 0
      y: 0
      width: 50
      height: 1
    - type: points
      coords: [[10, 5], [10, 6], [10, 7]]

  hotspots:
    - x: 25
      y: 10
      attraction: 1.0

  spawn_zones:
    - type: line
      start: [0, 10]
      end: [0, 10]
      direction: east  # Agent movement direction
    - type: line
      start: [49, 10]
      end: [49, 10]
      direction: west

# Export settings (can be overridden by CLI)
export:
  csv: true
  snapshot: true
  gif: false
```

---

## 7. Output Formats

### 7.1 CSV Output (`simulation_log.csv`)

```csv
step,agent_id,x,y,state
0,1,0,10,moving
0,2,49,10,moving
1,1,1,10,moving
1,2,48,10,moving
2,1,2,10,waiting
...
```

### 7.2 Summary Report (stdout)

```
================================================================================
                    FLOOR FIELD CA SIMULATION REPORT
================================================================================
Configuration: configs/supermarket.yaml
Random Seed: 42

SIMULATION METRICS
------------------
Total Steps:           500
Agents Completed:      87 / 100 (87.0%)
Average Travel Time:   156.3 steps
Peak Density:          0.45 agents/cell
Throughput:            0.174 agents/step

EMERGENT BEHAVIORS DETECTED
---------------------------
[✓] Lane Formation: Observed at step 45
[✓] Clogging Events: 3 events (steps 123, 267, 401)
[✓] Stop-and-Go Waves: 7 wave propagations detected

OUTPUT FILES
------------
CSV Log:    output/simulation_log.csv
Snapshot:   output/final_state.png
Animation:  output/simulation.gif (disabled)
================================================================================
```

### 7.3 Snapshot Image

The PNG snapshot displays:
- Grid cells with walls (black), walkable areas (white)
- Hotspots (yellow/orange markers)
- Agents (colored dots: blue=moving, red=waiting, green=shopping)
- Dynamic field overlay (semi-transparent heatmap)

---

## 8. Mathematical Model

### 8.1 Transition Probability (Burstedde Formula)

The probability of an agent moving from cell `i` to neighbor `j`:

```
P(i → j) = N * exp(kS * Sj + kD * Dj) * (1 - nj) * ξij
```

Where:
- `N` = Normalization constant
- `kS` = Static field sensitivity parameter
- `kD` = Dynamic field sensitivity parameter
- `Sj` = Static field value at cell j (distance to goal)
- `Dj` = Dynamic field value at cell j (pheromone)
- `nj` = Occupancy (1 if occupied, 0 if empty)
- `ξij` = 1 if j is walkable from i, 0 otherwise

### 8.2 Dynamic Field Update

At each time step:

```
D(t+1) = (1 - δ) * diffuse(D(t)) + deposits(t)
```

Where:
- `δ` = Decay rate
- `diffuse()` = 3x3 convolution kernel for spatial diffusion
- `deposits(t)` = Pheromone left by agents at step t

### 8.3 Conflict Resolution

When multiple agents desire the same cell:
1. **Friction-based**: Higher friction = lower probability of success
2. **Random priority**: Uniform random selection among competing agents
3. **Sequential update**: Process agents in random order

---

## 9. Implementation Notes

### 9.1 Dependencies

```
numpy>=1.20.0       # Grid operations and numerical computation
matplotlib>=3.4.0   # Visualization and snapshot generation
Pillow>=8.0.0       # GIF generation
PyYAML>=5.4.0       # Configuration parsing
dataclasses         # (stdlib) Configuration structures
```

### 9.2 Performance Considerations

- Use NumPy vectorized operations for field updates
- Pre-compute static field once at initialization
- Use scipy.ndimage.convolve for efficient diffusion
- Consider agent limit warnings for grids > 100x100 with >1000 agents

### 9.3 Testing Strategy

| Test Type | Coverage |
|-----------|----------|
| Unit | GridMap operations, Agent movement logic, Field calculations |
| Integration | Engine step execution, Export pipeline |
| Validation | Compare lane formation with published results |
| Performance | Benchmark with varying grid sizes and agent counts |

---

## 10. References

### 10.1 Core Floor Field Model

**Burstedde, C., Klauck, K., Schadschneider, A., & Zittartz, J. (2001)**
- Title: "Simulation of pedestrian dynamics using a two-dimensional cellular automaton"
- Journal: Physica A: Statistical Mechanics and its Applications, 295(3-4), 507-525
- Search: `Burstedde pedestrian cellular automata 2001`

### 10.2 Bi-directional Flow (Lane Formation)

**Blue, V. J., & Adler, J. L. (2001)**
- Title: "Cellular automata microsimulation for modeling bi-directional pedestrian walkways"
- Journal: Transportation Research Part B: Methodological, 35(3), 293-312
- Search: `Blue Adler cellular automata bi-directional pedestrian`

### 10.3 Social Groups Dynamics

**Moussaïd, M., Perozo, N., Garnier, S., Helbing, D., & Theraulaz, G. (2010)**
- Title: "The walking behaviour of pedestrian social groups and its impact on crowd dynamics"
- Journal: PLoS ONE, 5(4), e10047
- Search: `Moussaid pedestrian social groups walking behaviour 2010`

### 10.4 Faster-is-Slower Effect

**Helbing, D., Farkas, I., & Vicsek, T. (2000)**
- Title: "Simulating dynamical features of escape panic"
- Journal: Nature, 407(6803), 487-490
- Search: `Helbing escape panic faster is slower Nature 2000`

---

## Appendix A: Example Configuration Files

### A.1 Supermarket Aisle (`configs/supermarket.yaml`)

```yaml
grid:
  width: 60
  height: 15

simulation:
  max_steps: 1000
  agent_count: 80

floor_field:
  static_strength: 2.5
  dynamic_strength: 1.5
  diffusion_rate: 0.2
  decay_rate: 0.05

agents:
  friends_probability: 0.15
  patience: 30

layout:
  walls:
    - type: rectangle
      x: 0
      y: 0
      width: 60
      height: 1
    - type: rectangle
      x: 0
      y: 14
      width: 60
      height: 1
    # Shelf obstacles
    - type: rectangle
      x: 15
      y: 3
      width: 2
      height: 9
    - type: rectangle
      x: 35
      y: 3
      width: 2
      height: 9

  hotspots:
    - x: 16
      y: 7
      attraction: 0.8
    - x: 36
      y: 7
      attraction: 0.8

  spawn_zones:
    - type: line
      start: [0, 7]
      end: [0, 7]
      direction: east
    - type: line
      start: [59, 7]
      end: [59, 7]
      direction: west
```

### A.2 Train Cabin (`configs/train_cabin.yaml`)

```yaml
grid:
  width: 80
  height: 8

simulation:
  max_steps: 500
  agent_count: 40

floor_field:
  static_strength: 3.0
  dynamic_strength: 1.0
  diffusion_rate: 0.25
  decay_rate: 0.1

agents:
  friends_probability: 0.3
  patience: 20

layout:
  walls:
    # Cabin walls
    - type: rectangle
      x: 0
      y: 0
      width: 80
      height: 1
    - type: rectangle
      x: 0
      y: 7
      width: 80
      height: 1
    # Seat blocks
    - type: rectangle
      x: 10
      y: 2
      width: 3
      height: 4
    - type: rectangle
      x: 25
      y: 2
      width: 3
      height: 4
    - type: rectangle
      x: 40
      y: 2
      width: 3
      height: 4
    - type: rectangle
      x: 55
      y: 2
      width: 3
      height: 4
    - type: rectangle
      x: 70
      y: 2
      width: 3
      height: 4

  hotspots:
    # Luggage storage areas
    - x: 20
      y: 4
      attraction: 0.5
    - x: 50
      y: 4
      attraction: 0.5

  spawn_zones:
    - type: line
      start: [0, 3]
      end: [0, 4]
      direction: east
```

---

## Appendix B: State Machine

### Agent State Transitions

```
                    ┌─────────────┐
                    │   SPAWNED   │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │        MOVING          │◄────────────────┐
              │  (normal navigation)   │                 │
              └───────────┬────────────┘                 │
                          │                              │
          ┌───────────────┼───────────────┐              │
          │               │               │              │
          ▼               ▼               ▼              │
    ┌──────────┐   ┌──────────┐   ┌──────────────┐       │
    │ WAITING  │   │ SHOPPING │   │    EXITED    │       │
    │(blocked) │   │(at goal) │   │ (completed)  │       │
    └────┬─────┘   └────┬─────┘   └──────────────┘       │
         │              │                                │
         │              │ (after shopping_time)          │
         │              └────────────────────────────────┘
         │                                               │
         └───────────────────────────────────────────────┘
               (path cleared / patience timeout)
```
