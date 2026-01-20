#!/usr/bin/env python3
"""
Floor Field Cellular Automata Simulation

A pedestrian dynamics simulator based on the Burstedde Floor Field model.

Usage:
    python main.py --config configs/supermarket.yaml [options]

Examples:
    python main.py --config configs/supermarket.yaml
    python main.py --config configs/train_cabin.yaml --gif --out-dir results/
    python main.py --config configs/test.yaml --no-csv --no-snapshot --quiet
    python main.py --config configs/supermarket.yaml --seed 42
"""

import argparse
import sys
from pathlib import Path

from config import load_config
from model.engine import SimulationEngine
from export.csv_writer import CSVWriter
from export.visualizer import Visualizer
from export.reporter import Reporter


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Floor Field Cellular Automata Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --config configs/supermarket.yaml
    python main.py --config configs/train_cabin.yaml --gif --out-dir results/
    python main.py --config configs/test.yaml --no-csv --no-snapshot --quiet
    python main.py --config configs/supermarket.yaml --seed 42
        """
    )

    # Required arguments
    parser.add_argument('--config', type=Path, required=True,
                        help='Path to YAML configuration file')

    # Optional overrides
    parser.add_argument('--steps', type=int, default=None,
                        help='Override max simulation steps')
    parser.add_argument('--out-dir', type=Path, default=Path('./output'),
                        help='Output directory for exports (default: ./output)')

    # Export toggles
    parser.add_argument('--csv', dest='csv', action='store_true', default=None,
                        help='Enable CSV export (default)')
    parser.add_argument('--no-csv', dest='csv', action='store_false',
                        help='Disable CSV export')

    parser.add_argument('--snapshot', dest='snapshot', action='store_true', default=None,
                        help='Enable final snapshot (default)')
    parser.add_argument('--no-snapshot', dest='snapshot', action='store_false',
                        help='Disable final snapshot')

    parser.add_argument('--gif', action='store_true', default=False,
                        help='Enable GIF animation export')

    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Suppress stdout output')

    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Apply CLI overrides
    if args.steps is not None:
        config.max_steps = args.steps
    if args.csv is not None:
        config.csv_enabled = args.csv
    if args.snapshot is not None:
        config.snapshot_enabled = args.snapshot
    if args.gif:
        config.gif_enabled = True
    config.quiet = args.quiet
    if args.seed is not None:
        config.seed = args.seed
    config.out_dir = args.out_dir

    # Initialize engine
    if not config.quiet:
        print(f"Initializing simulation...")
        print(f"  Grid: {config.grid.width}x{config.grid.height}")
        print(f"  Agents: {config.agent_count}")
        print(f"  Max steps: {config.max_steps}")

    engine = SimulationEngine(config)

    if not config.quiet:
        print(f"  Spawned: {len(engine.agents)} agents")

    # Initialize exporters
    csv_writer = None
    if config.csv_enabled:
        csv_writer = CSVWriter(config.out_dir / 'simulation_log.csv')
        csv_writer.open()

    visualizer = Visualizer(
        config.grid.width, config.grid.height,
        engine.grid.walls, engine.grid.goals
    )

    reporter = Reporter(str(args.config), config.seed)

    # Main simulation loop
    if not config.quiet:
        print(f"\nRunning simulation...")

    final_state = None
    try:
        while not engine.is_finished():
            state = engine.step()
            final_state = state

            # Export CSV
            if csv_writer:
                csv_writer.append(state)

            # Buffer GIF frame (every N steps to reduce memory)
            if config.gif_enabled:
                if state.step % 5 == 0 or engine.is_finished():
                    visualizer.buffer_frame(state)

            # Update reporter
            reporter.update(state)

            # Progress indicator
            if not config.quiet and state.step % 100 == 0:
                active = state.metrics.get('active_agents', 0)
                completed = int(state.metrics.get('completed', 0))
                print(f"  Step {state.step}: {active} active, {completed} completed")

    except KeyboardInterrupt:
        if not config.quiet:
            print("\nSimulation interrupted by user.")

    # Cleanup and final exports
    if csv_writer:
        csv_writer.close()
        if not config.quiet:
            print(f"\nCSV saved: {config.out_dir / 'simulation_log.csv'}")

    if config.snapshot_enabled and final_state:
        snapshot_path = config.out_dir / 'final_state.png'
        visualizer.save_snapshot(final_state, snapshot_path)
        if not config.quiet:
            print(f"Snapshot saved: {snapshot_path}")

    if config.gif_enabled:
        gif_path = config.out_dir / 'simulation.gif'
        if not config.quiet:
            print(f"Generating GIF ({len(visualizer.frames)} frames)...")
        visualizer.generate_gif(gif_path, fps=10)
        if not config.quiet:
            print(f"Animation saved: {gif_path}")

    # Print summary report
    if not config.quiet and final_state:
        report = reporter.generate_summary(
            final_state,
            config.out_dir,
            config.csv_enabled,
            config.snapshot_enabled,
            config.gif_enabled
        )
        print(report)

    return 0


if __name__ == '__main__':
    sys.exit(main())
