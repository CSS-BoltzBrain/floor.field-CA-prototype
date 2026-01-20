"""Grid map management for Floor Field CA simulation."""

import numpy as np
from typing import Tuple, List, Set


class GridMap:
    """
    Manages the 2D simulation environment with multiple data layers.

    Coordinate convention: (x, y) for API, [y, x] for array indexing.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Boolean mask: True = wall (impassable)
        self.walls = np.zeros((height, width), dtype=bool)

        # Occupancy: 0 = empty, positive int = agent_id
        self.occupancy = np.zeros((height, width), dtype=np.int32)

        # Goal cells for static field computation
        self.goals: Set[Tuple[int, int]] = set()

    def add_wall_rectangle(self, x: int, y: int, w: int, h: int) -> None:
        """Mark rectangular region as wall."""
        # Clamp to grid boundaries
        x_end = min(x + w, self.width)
        y_end = min(y + h, self.height)
        x = max(0, x)
        y = max(0, y)
        self.walls[y:y_end, x:x_end] = True

    def add_wall_points(self, coords: List[Tuple[int, int]]) -> None:
        """Mark specific cells as walls."""
        for x, y in coords:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.walls[y, x] = True

    def add_goal(self, x: int, y: int) -> None:
        """Register a goal cell for static field."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.goals.add((x, y))

    def is_walkable(self, x: int, y: int) -> bool:
        """Check if cell is within bounds and not a wall."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return not self.walls[y, x]

    def is_occupied(self, x: int, y: int) -> bool:
        """Check if cell contains an agent."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True  # Out of bounds treated as occupied
        return self.occupancy[y, x] != 0

    def get_neighbors(self, x: int, y: int,
                      include_diagonals: bool = False) -> List[Tuple[int, int]]:
        """
        Get walkable neighboring cells (von Neumann or Moore neighborhood).
        Always includes staying in place as an option.
        """
        if include_diagonals:
            # Moore neighborhood (8-connected)
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
        else:
            # Von Neumann neighborhood (4-connected)
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        neighbors = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                neighbors.append((nx, ny))

        # Always include staying in place (if walkable)
        if self.is_walkable(x, y):
            neighbors.append((x, y))

        return neighbors

    def place_agent(self, agent_id: int, x: int, y: int) -> None:
        """Place agent at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.occupancy[y, x] = agent_id

    def remove_agent(self, x: int, y: int) -> None:
        """Remove agent from position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.occupancy[y, x] = 0

    def move_agent(self, agent_id: int,
                   from_pos: Tuple[int, int],
                   to_pos: Tuple[int, int]) -> None:
        """Atomically move agent from one cell to another."""
        self.remove_agent(*from_pos)
        self.place_agent(agent_id, *to_pos)

    def get_occupied_positions(self) -> Set[Tuple[int, int]]:
        """Return set of all occupied cell positions."""
        ys, xs = np.where(self.occupancy != 0)
        return {(int(x), int(y)) for x, y in zip(xs, ys)}
