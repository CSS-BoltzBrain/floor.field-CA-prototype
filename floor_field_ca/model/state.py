"""State snapshot dataclasses for Floor Field CA simulation."""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass(frozen=True)
class AgentSnapshot:
    """Immutable snapshot of an agent's state at a given time step."""
    agent_id: int
    x: int
    y: int
    state: str  # "moving", "waiting", "shopping", "exited"


@dataclass
class SimulationState:
    """Complete snapshot of simulation state at a given time step."""
    step: int
    agents: List[AgentSnapshot]
    grid_occupancy: np.ndarray  # Copy of occupancy grid
    dynamic_field: np.ndarray   # Copy of pheromone field
    metrics: Dict[str, float]   # throughput, density, etc.

    def to_csv_rows(self) -> List[Dict]:
        """Convert to CSV-compatible format."""
        return [
            {
                "step": self.step,
                "agent_id": a.agent_id,
                "x": a.x,
                "y": a.y,
                "state": a.state
            }
            for a in self.agents
        ]
