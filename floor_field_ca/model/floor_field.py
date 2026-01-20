"""Floor field implementations for Floor Field CA simulation."""

import numpy as np
from collections import deque
from typing import Set, Tuple
from scipy.ndimage import convolve


class StaticField:
    """
    Pre-computed distance field guiding agents toward goals.
    Lower distance values = closer to goal.
    Attractiveness is computed as (max_dist - distance) so higher = more attractive.
    """

    def __init__(self, grid_width: int, grid_height: int):
        self.width = grid_width
        self.height = grid_height
        self.field = np.full((grid_height, grid_width), np.inf)
        self.attractiveness = np.full((grid_height, grid_width), -np.inf)

    def compute(self, walls: np.ndarray, goals: Set[Tuple[int, int]]) -> None:
        """
        Compute distance gradient using multi-source BFS from all goals.
        Walls are impassable.
        """
        if not goals:
            return

        # Reset field
        self.field = np.full((self.height, self.width), np.inf)

        # BFS from all goals simultaneously (multi-source BFS)
        queue = deque()
        visited = np.zeros((self.height, self.width), dtype=bool)

        # Initialize goals with distance 0
        for gx, gy in goals:
            if 0 <= gx < self.width and 0 <= gy < self.height:
                self.field[gy, gx] = 0
                queue.append((gx, gy, 0))
                visited[gy, gx] = True

        # BFS expansion (4-connected)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y, dist = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height
                        and not visited[ny, nx] and not walls[ny, nx]):
                    visited[ny, nx] = True
                    self.field[ny, nx] = dist + 1
                    queue.append((nx, ny, dist + 1))

        # Convert distance to attractiveness (higher = closer to goal)
        finite_mask = np.isfinite(self.field)
        if np.any(finite_mask):
            max_dist = np.max(self.field[finite_mask])
            self.attractiveness = np.where(
                finite_mask,
                max_dist - self.field,
                -np.inf  # Unreachable cells
            )
        else:
            self.attractiveness = np.full((self.height, self.width), -np.inf)

    def get_value(self, x: int, y: int) -> float:
        """Return attractiveness value at position (for Burstedde formula)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.attractiveness[y, x]
        return -np.inf

    def get_distance(self, x: int, y: int) -> float:
        """Return raw distance value at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.field[y, x]
        return np.inf


class DynamicField:
    """
    Time-varying pheromone field for collective behavior.
    Agents deposit pheromones; field diffuses and decays over time.
    """

    def __init__(self, grid_width: int, grid_height: int,
                 diffusion_rate: float, decay_rate: float):
        self.width = grid_width
        self.height = grid_height
        self.diffusion_rate = diffusion_rate  # alpha
        self.decay_rate = decay_rate          # delta

        self.field = np.zeros((grid_height, grid_width), dtype=np.float64)

        # 3x3 diffusion kernel (normalized)
        self.diffusion_kernel = np.array([
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ], dtype=np.float64)

    def deposit(self, x: int, y: int, amount: float = 1.0) -> None:
        """Agent leaves pheromone trace at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.field[y, x] += amount

    def update(self, walls: np.ndarray) -> None:
        """
        Apply diffusion (3x3 convolution) and decay.
        Formula: D(t+1) = (1 - delta) * blend(D, diffused)
        where blend = (1 - alpha) * D + alpha * diffused
        """
        # Diffusion via convolution
        diffused = convolve(self.field, self.diffusion_kernel,
                            mode='constant', cval=0.0)

        # Blend original and diffused based on diffusion rate
        blended = (1 - self.diffusion_rate) * self.field + \
                  self.diffusion_rate * diffused

        # Apply decay
        self.field = (1 - self.decay_rate) * blended

        # Zero out pheromones on walls
        self.field[walls] = 0

    def get_value(self, x: int, y: int) -> float:
        """Return pheromone concentration at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.field[y, x]
        return 0.0

    def reset(self) -> None:
        """Reset the dynamic field to zero."""
        self.field.fill(0.0)
