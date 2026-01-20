"""Summary report generation for Floor Field CA simulation."""

from typing import List, Dict, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..model.state import SimulationState


class Reporter:
    """Generates summary statistics and formatted text report."""

    def __init__(self, config_path: str, seed: Optional[int]):
        self.config_path = config_path
        self.seed = seed
        self.step_metrics: List[Dict] = []
        self.peak_density = 0.0
        self.clogging_events = 0
        self._prev_density = 0.0
        self._prev_throughput = 0.0
        self._stagnation_steps = 0

    def update(self, state: "SimulationState") -> None:
        """Accumulate metrics per step."""
        self.step_metrics.append(state.metrics.copy())

        # Track peak density
        current_density = state.metrics.get('density', 0)
        if current_density > self.peak_density:
            self.peak_density = current_density

        # Detect clogging (high density with no throughput increase)
        current_throughput = state.metrics.get('completed', 0)
        if current_density > 0.2:
            if current_throughput == self._prev_throughput:
                self._stagnation_steps += 1
                if self._stagnation_steps >= 10:
                    self.clogging_events += 1
                    self._stagnation_steps = 0
            else:
                self._stagnation_steps = 0

        self._prev_density = current_density
        self._prev_throughput = current_throughput

    def generate_summary(self, final_state: "SimulationState",
                         output_dir: Path,
                         csv_enabled: bool,
                         snapshot_enabled: bool,
                         gif_enabled: bool) -> str:
        """Returns formatted text report."""
        metrics = final_state.metrics
        total_agents = int(metrics.get('total_agents', 0))
        completed = int(metrics.get('completed', 0))
        avg_travel_time = metrics.get('avg_travel_time', 0)

        completion_pct = (completed / total_agents * 100) if total_agents > 0 else 0
        throughput = metrics.get('throughput', 0)

        # Build report
        lines = [
            "",
            "=" * 80,
            "                    FLOOR FIELD CA SIMULATION REPORT",
            "=" * 80,
            f"Configuration: {self.config_path}",
            f"Random Seed: {self.seed if self.seed else 'None (random)'}",
            "",
            "SIMULATION METRICS",
            "-" * 40,
            f"Total Steps:           {final_state.step}",
            f"Agents Completed:      {completed} / {total_agents} ({completion_pct:.1f}%)",
            f"Average Travel Time:   {avg_travel_time:.1f} steps",
            f"Peak Density:          {self.peak_density:.4f} agents/cell",
            f"Throughput:            {throughput:.4f} agents/step",
            "",
            "EMERGENT BEHAVIORS DETECTED",
            "-" * 40,
            f"[{'X' if self.clogging_events > 0 else ' '}] Clogging Events: {self.clogging_events} detected",
            "",
            "OUTPUT FILES",
            "-" * 40,
        ]

        # Output file paths
        if csv_enabled:
            lines.append(f"CSV Log:    {output_dir / 'simulation_log.csv'}")
        else:
            lines.append("CSV Log:    (disabled)")

        if snapshot_enabled:
            lines.append(f"Snapshot:   {output_dir / 'final_state.png'}")
        else:
            lines.append("Snapshot:   (disabled)")

        if gif_enabled:
            lines.append(f"Animation:  {output_dir / 'simulation.gif'}")
        else:
            lines.append("Animation:  (disabled)")

        lines.append("=" * 80)

        return "\n".join(lines)
