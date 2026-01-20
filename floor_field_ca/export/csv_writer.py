"""CSV export functionality for Floor Field CA simulation."""

import csv
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..model.state import SimulationState


class CSVWriter:
    """
    Exports simulation data to CSV format incrementally.

    Output format:
        step,agent_id,x,y,state
        0,1,5,10,moving
        ...
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.file: Optional[object] = None
        self.writer: Optional[csv.DictWriter] = None
        self._is_open = False

    def open(self) -> None:
        """Initialize file and write header."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_path, 'w', newline='')
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=['step', 'agent_id', 'x', 'y', 'state']
        )
        self.writer.writeheader()
        self._is_open = True

    def append(self, state: "SimulationState") -> None:
        """Write state data for current step."""
        if not self._is_open:
            self.open()
        for row in state.to_csv_rows():
            self.writer.writerow(row)
        self.file.flush()  # Ensure data is written

    def close(self) -> None:
        """Close file handle."""
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
            self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
