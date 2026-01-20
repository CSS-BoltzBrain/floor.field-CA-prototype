"""Agent implementation with Burstedde transition probability."""

from enum import Enum
from typing import Tuple, Dict, List, Optional, Set
import numpy as np

from .floor_field import StaticField, DynamicField


class AgentState(Enum):
    """Possible states for an agent."""
    MOVING = "moving"
    WAITING = "waiting"
    SHOPPING = "shopping"
    EXITED = "exited"


class Agent:
    """
    Individual pedestrian entity implementing the Burstedde model.

    The Burstedde formula for transition probability:
    P(i -> j) = N * exp(kS * S_j + kD * D_j) * (1 - n_j) * xi_ij

    Where:
    - N = normalization constant
    - kS = static field sensitivity
    - kD = dynamic field sensitivity
    - S_j = static field value (attractiveness to goal)
    - D_j = dynamic field value (pheromone)
    - n_j = 1 if occupied, 0 otherwise
    - xi_ij = 1 if walkable, 0 otherwise
    """

    def __init__(self, agent_id: int,
                 position: Tuple[int, int],
                 goal: Tuple[int, int],
                 direction: str,
                 patience: int,
                 group_id: Optional[int] = None):
        self.id = agent_id
        self.position = position
        self.goal = goal
        self.direction = direction  # "east", "west", "north", "south"
        self.state = AgentState.MOVING
        self.group_id = group_id
        self.initial_patience = patience
        self.patience = patience
        self.steps_taken = 0

    def calculate_transition_probabilities(
        self,
        neighbors: List[Tuple[int, int]],
        static_field: StaticField,
        dynamic_field: DynamicField,
        occupied: Set[Tuple[int, int]],
        kS: float,
        kD: float
    ) -> Dict[Tuple[int, int], float]:
        """
        Calculate probability distribution over possible moves.

        Uses the Burstedde formula with log-sum-exp for numerical stability.
        Returns dict mapping positions to probabilities (sum to 1).
        """
        # Collect log-probabilities (scores) for valid moves
        valid_moves = []
        scores = []

        for nx, ny in neighbors:
            # Occupancy factor: (1 - n_j)
            # If occupied by another agent, probability = 0
            # But staying in place is always allowed
            if (nx, ny) in occupied and (nx, ny) != self.position:
                continue  # Skip occupied cells

            # Get field values
            S_j = static_field.get_value(nx, ny)
            D_j = dynamic_field.get_value(nx, ny)

            # Handle unreachable cells
            if S_j == -np.inf:
                continue

            # Burstedde formula: score = kS * S_j + kD * D_j
            score = kS * S_j + kD * D_j
            valid_moves.append((nx, ny))
            scores.append(score)

        if not valid_moves:
            # No valid moves - stay in place with probability 1
            return {self.position: 1.0}

        # Use log-sum-exp for numerical stability
        scores = np.array(scores)
        max_score = np.max(scores)
        # Subtract max for numerical stability: softmax(x) = softmax(x - max(x))
        exp_scores = np.exp(scores - max_score)
        total = np.sum(exp_scores)
        probs = exp_scores / total

        return {pos: float(p) for pos, p in zip(valid_moves, probs)}

    def decide_next_move(
        self,
        probabilities: Dict[Tuple[int, int], float],
        rng: np.random.Generator
    ) -> Tuple[int, int]:
        """
        Sample from transition probabilities to select next cell.
        """
        if not probabilities:
            return self.position

        positions = list(probabilities.keys())
        probs = list(probabilities.values())

        # Ensure probabilities sum to 1 (handle floating point errors)
        probs = np.array(probs)
        probs = probs / probs.sum()

        # Weighted random choice
        idx = rng.choice(len(positions), p=probs)
        return positions[idx]

    def update_state(self, new_position: Tuple[int, int],
                     moved: bool, at_goal: bool) -> None:
        """Update agent state based on movement result."""
        if at_goal:
            self.state = AgentState.EXITED
        elif not moved:
            self.state = AgentState.WAITING
            self.patience -= 1
        else:
            self.state = AgentState.MOVING
            self.patience = self.initial_patience
            self.steps_taken += 1

        self.position = new_position

    def is_exhausted(self) -> bool:
        """Check if agent has run out of patience."""
        return self.patience <= 0

    def __repr__(self) -> str:
        return (f"Agent(id={self.id}, pos={self.position}, "
                f"state={self.state.value})")
