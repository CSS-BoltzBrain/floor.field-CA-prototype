"""Visualization and export for Floor Field CA simulation."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pathlib import Path
from typing import List, Set, Tuple, TYPE_CHECKING
from PIL import Image
import io

if TYPE_CHECKING:
    from ..model.state import SimulationState


class Visualizer:
    """
    Generates visual outputs using matplotlib.

    Supports:
    - Single PNG snapshots
    - Animated GIF compilation
    """

    # Color scheme
    COLORS = {
        'wall': '#2C3E50',      # Dark blue-gray
        'floor': '#ECF0F1',     # Light gray
        'goal': '#F39C12',      # Orange
        'moving': '#3498DB',    # Blue
        'waiting': '#E74C3C',   # Red
        'shopping': '#27AE60',  # Green
        'exited': '#95A5A6',    # Gray
    }

    def __init__(self, grid_width: int, grid_height: int,
                 walls: np.ndarray, goals: Set[Tuple[int, int]]):
        self.width = grid_width
        self.height = grid_height
        self.walls = walls
        self.goals = goals
        self.frames: List[Image.Image] = []

    def _create_figure(self, state: "SimulationState",
                       show_pheromone: bool = True) -> plt.Figure:
        """Create matplotlib figure for state visualization."""
        # Determine figure size based on grid aspect ratio
        aspect = self.width / self.height
        fig_height = 6
        fig_width = max(8, fig_height * aspect)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Base layer: walls and floor
        base = np.ones((self.height, self.width, 3))
        floor_rgb = to_rgb(self.COLORS['floor'])
        wall_rgb = to_rgb(self.COLORS['wall'])

        # Fill with floor color
        base[:, :] = floor_rgb

        # Mark walls
        base[self.walls] = wall_rgb

        # Overlay pheromone heatmap
        if show_pheromone and np.max(state.dynamic_field) > 0:
            normalized = state.dynamic_field / np.max(state.dynamic_field)
            # Add red tint for pheromones
            for c in range(3):
                pheromone_color = [0.8, 0.2, 0.2]  # Red tint
                base[:, :, c] = np.clip(
                    base[:, :, c] * (1 - 0.5 * normalized) +
                    pheromone_color[c] * 0.5 * normalized,
                    0, 1
                )

        ax.imshow(base, origin='lower', aspect='equal',
                  extent=[-0.5, self.width - 0.5, -0.5, self.height - 0.5])

        # Draw goals
        for gx, gy in self.goals:
            ax.plot(gx, gy, 's', color=self.COLORS['goal'],
                    markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                    alpha=0.7)

        # Draw agents
        active_count = 0
        for agent in state.agents:
            if agent.state == 'exited':
                continue
            active_count += 1
            color = self.COLORS.get(agent.state, '#95A5A6')
            ax.plot(agent.x, agent.y, 'o', color=color,
                    markersize=5, markeredgecolor='white', markeredgewidth=0.3)

        # Title and labels
        ax.set_title(f'Step {state.step} | Active Agents: {active_count} | '
                     f'Completed: {int(state.metrics.get("completed", 0))}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Set axis limits
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Moving',
                       markerfacecolor=self.COLORS['moving'], markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Waiting',
                       markerfacecolor=self.COLORS['waiting'], markersize=8),
            plt.Line2D([0], [0], marker='s', color='w', label='Goal/Exit',
                       markerfacecolor=self.COLORS['goal'], markersize=8),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()
        return fig

    def buffer_frame(self, state: "SimulationState") -> None:
        """Store frame for GIF generation."""
        fig = self._create_figure(state)

        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        img = Image.open(buf).copy()
        self.frames.append(img)
        buf.close()
        plt.close(fig)

    def save_snapshot(self, state: "SimulationState", output_path: Path) -> None:
        """Save single PNG image of current state."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = self._create_figure(state)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def generate_gif(self, output_path: Path, fps: int = 10) -> None:
        """Compile buffered frames into animated GIF."""
        if not self.frames:
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration = int(1000 / fps)  # milliseconds per frame

        self.frames[0].save(
            output_path,
            save_all=True,
            append_images=self.frames[1:],
            duration=duration,
            loop=0
        )

    def clear_frames(self) -> None:
        """Clear buffered frames."""
        self.frames.clear()
