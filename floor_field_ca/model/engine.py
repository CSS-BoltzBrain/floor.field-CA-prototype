"""Simulation engine for Floor Field CA."""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, TYPE_CHECKING, Any
from collections import defaultdict

from .grid import GridMap
from .agent import Agent, AgentState
from .floor_field import StaticField, DynamicField
from .state import SimulationState, AgentSnapshot

if TYPE_CHECKING:
    from ..config import SimulationConfig


class SimulationEngine:
    """
    Orchestrates the discrete-time simulation loop.

    Implements:
    1. Grid and floor field initialization
    2. Agent spawning
    3. Parallel update with conflict resolution
    4. State snapshot generation
    """

    def __init__(self, config: "SimulationConfig"):
        self.config = config
        self.current_step = 0
        self.rng = np.random.default_rng(config.seed)

        # Initialize grid
        self.grid = GridMap(config.grid.width, config.grid.height)
        self._setup_walls()

        # Initialize direction-specific static fields
        self.static_fields: Dict[str, StaticField] = {}
        self._setup_static_fields()

        self.dynamic_field = DynamicField(
            config.grid.width, config.grid.height,
            config.floor_field.diffusion_rate,
            config.floor_field.decay_rate
        )

        # Initialize agents
        self.agents: List[Agent] = []
        self.active_agents: List[Agent] = []
        self._spawn_agents()

        # Metrics tracking
        self.completed_count = 0
        self.total_travel_time = 0

    def _setup_walls(self) -> None:
        """Configure walls from config."""
        for wall_spec in self.config.layout.walls:
            if wall_spec.wall_type == "rectangle":
                self.grid.add_wall_rectangle(
                    wall_spec.data['x'], wall_spec.data['y'],
                    wall_spec.data['width'], wall_spec.data['height']
                )
            elif wall_spec.wall_type == "points":
                self.grid.add_wall_points(wall_spec.data['coords'])

    def _setup_static_fields(self) -> None:
        """Create direction-specific static fields."""
        # Collect unique directions from spawn zones
        directions = set()
        for zone in self.config.layout.spawn_zones:
            directions.add(zone.direction)

        # Create static field for each direction
        for direction in directions:
            goals = self._get_exit_goals(direction)
            static_field = StaticField(self.config.grid.width, self.config.grid.height)
            static_field.compute(self.grid.walls, goals)
            self.static_fields[direction] = static_field

            # Also add goals to grid for visualization
            for gx, gy in goals:
                self.grid.add_goal(gx, gy)

    def _get_exit_goals(self, direction: str) -> Set[Tuple[int, int]]:
        """Get exit goals for a specific direction."""
        goals = set()
        if direction == "east":
            # Agents moving east, exit on right edge
            for y in range(self.config.grid.height):
                if not self.grid.walls[y, self.config.grid.width - 1]:
                    goals.add((self.config.grid.width - 1, y))
        elif direction == "west":
            # Agents moving west, exit on left edge
            for y in range(self.config.grid.height):
                if not self.grid.walls[y, 0]:
                    goals.add((0, y))
        elif direction == "north":
            # Agents moving north, exit on top edge
            for x in range(self.config.grid.width):
                if not self.grid.walls[self.config.grid.height - 1, x]:
                    goals.add((x, self.config.grid.height - 1))
        elif direction == "south":
            # Agents moving south, exit on bottom edge
            for x in range(self.config.grid.width):
                if not self.grid.walls[0, x]:
                    goals.add((x, 0))
        return goals

    def _get_line_positions(self, start: Tuple[int, int],
                            end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all positions along a line from start to end."""
        positions = []
        x1, y1 = start
        x2, y2 = end

        # Handle horizontal, vertical, or single point
        if x1 == x2 and y1 == y2:
            positions.append((x1, y1))
        elif x1 == x2:
            # Vertical line
            for y in range(min(y1, y2), max(y1, y2) + 1):
                positions.append((x1, y))
        elif y1 == y2:
            # Horizontal line
            for x in range(min(x1, x2), max(x1, x2) + 1):
                positions.append((x, y1))
        else:
            # Diagonal - use Bresenham's algorithm
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            x, y = x1, y1
            while True:
                positions.append((x, y))
                if x == x2 and y == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

        return positions

    def _get_goal_for_direction(self, direction: str) -> Tuple[int, int]:
        """Get a representative goal position for a direction."""
        mid_x = self.config.grid.width // 2
        mid_y = self.config.grid.height // 2

        if direction == "east":
            return (self.config.grid.width - 1, mid_y)
        elif direction == "west":
            return (0, mid_y)
        elif direction == "north":
            return (mid_x, self.config.grid.height - 1)
        elif direction == "south":
            return (mid_x, 0)
        return (mid_x, mid_y)

    def _spawn_agents(self) -> None:
        """Create and place initial agents at spawn zones."""
        agent_id = 1
        agents_per_zone = self.config.agent_count // max(1, len(self.config.layout.spawn_zones))

        for zone in self.config.layout.spawn_zones:
            positions = self._get_line_positions(zone.start, zone.end)
            spawned_in_zone = 0

            for pos in positions:
                if agent_id > self.config.agent_count:
                    break
                if spawned_in_zone >= agents_per_zone and zone != self.config.layout.spawn_zones[-1]:
                    break

                if not self.grid.is_occupied(*pos) and self.grid.is_walkable(*pos):
                    goal = self._get_goal_for_direction(zone.direction)
                    agent = Agent(
                        agent_id=agent_id,
                        position=pos,
                        goal=goal,
                        direction=zone.direction,
                        patience=self.config.agents.patience
                    )
                    self.agents.append(agent)
                    self.active_agents.append(agent)
                    self.grid.place_agent(agent_id, *pos)
                    agent_id += 1
                    spawned_in_zone += 1

        # If we didn't spawn enough agents, try to spawn more at random walkable positions
        # Use only directions that have static fields (from configured spawn zones)
        available_directions = list(self.static_fields.keys())
        if not available_directions:
            available_directions = ["east"]  # fallback

        while len(self.agents) < self.config.agent_count:
            # Find a random walkable, unoccupied position
            attempts = 0
            while attempts < 100:
                x = self.rng.integers(0, self.config.grid.width)
                y = self.rng.integers(0, self.config.grid.height)
                if self.grid.is_walkable(x, y) and not self.grid.is_occupied(x, y):
                    # Pick a random direction from available directions
                    direction = self.rng.choice(available_directions)
                    goal = self._get_goal_for_direction(direction)
                    agent = Agent(
                        agent_id=agent_id,
                        position=(x, y),
                        goal=goal,
                        direction=direction,
                        patience=self.config.agents.patience
                    )
                    self.agents.append(agent)
                    self.active_agents.append(agent)
                    self.grid.place_agent(agent_id, x, y)
                    agent_id += 1
                    break
                attempts += 1
            else:
                # Couldn't find space for more agents
                break

    def step(self) -> SimulationState:
        """
        Execute one discrete time step.

        1. Calculate desired moves for all agents (parallel update)
        2. Resolve conflicts (random priority)
        3. Execute valid moves
        4. Update agent states
        5. Deposit pheromones
        6. Diffuse and decay dynamic field
        7. Return current state snapshot
        """
        self.current_step += 1

        # Get current occupancy
        occupied: Set[Tuple[int, int]] = {
            a.position for a in self.active_agents
            if a.state != AgentState.EXITED
        }

        # Phase 1: Calculate desired moves for all agents
        desired_moves: Dict[int, Tuple[int, int]] = {}
        for agent in self.active_agents:
            if agent.state == AgentState.EXITED:
                continue

            neighbors = self.grid.get_neighbors(*agent.position)
            # Use direction-specific static field
            static_field = self.static_fields.get(agent.direction)
            if static_field is None:
                # Fallback: use first available static field
                static_field = next(iter(self.static_fields.values()))
            probs = agent.calculate_transition_probabilities(
                neighbors,
                static_field,
                self.dynamic_field,
                occupied,
                self.config.floor_field.static_strength,
                self.config.floor_field.dynamic_strength
            )
            desired_moves[agent.id] = agent.decide_next_move(probs, self.rng)

        # Phase 2: Resolve conflicts
        # Group agents by desired target cell
        target_to_agents: Dict[Tuple[int, int], List[Agent]] = defaultdict(list)
        for agent in self.active_agents:
            if agent.id in desired_moves:
                target = desired_moves[agent.id]
                target_to_agents[target].append(agent)

        # Resolve: random priority among competing agents
        winners: Dict[int, Tuple[int, int]] = {}
        for target, competing in target_to_agents.items():
            if len(competing) == 1:
                winners[competing[0].id] = target
            else:
                # Random selection among competitors
                # Agents trying to stay in place get priority
                staying = [a for a in competing if a.position == target]
                if staying:
                    winner = staying[0]
                else:
                    winner = self.rng.choice(competing)

                winners[winner.id] = target
                # Others stay in place
                for agent in competing:
                    if agent.id != winner.id:
                        winners[agent.id] = agent.position

        # Phase 3 & 4: Execute moves and update states
        newly_exited = []
        for agent in self.active_agents:
            if agent.state == AgentState.EXITED:
                continue

            old_pos = agent.position
            new_pos = winners.get(agent.id, old_pos)
            moved = (new_pos != old_pos)

            # Check if reached exit boundary
            at_goal = self._check_at_exit(agent, new_pos)

            # Update grid
            if moved:
                self.grid.move_agent(agent.id, old_pos, new_pos)

            # Update agent state
            agent.update_state(new_pos, moved, at_goal)

            if at_goal:
                newly_exited.append(agent)
                self.completed_count += 1
                self.total_travel_time += agent.steps_taken
                self.grid.remove_agent(*new_pos)

        # Remove exited agents from active list
        self.active_agents = [a for a in self.active_agents
                              if a.state != AgentState.EXITED]

        # Phase 5: Deposit pheromones (from agents who moved)
        for agent in self.active_agents:
            if agent.state == AgentState.MOVING:
                self.dynamic_field.deposit(*agent.position)

        # Phase 6: Update dynamic field (diffusion + decay)
        self.dynamic_field.update(self.grid.walls)

        # Phase 7: Create state snapshot
        return self._create_state_snapshot()

    def _check_at_exit(self, agent: Agent, position: Tuple[int, int]) -> bool:
        """Check if agent has reached an exit boundary."""
        x, y = position
        direction = agent.direction

        if direction == "east" and x >= self.config.grid.width - 1:
            return True
        elif direction == "west" and x <= 0:
            return True
        elif direction == "north" and y >= self.config.grid.height - 1:
            return True
        elif direction == "south" and y <= 0:
            return True

        return False

    def _create_state_snapshot(self) -> SimulationState:
        """Create immutable snapshot of current simulation state."""
        agent_snapshots = [
            AgentSnapshot(
                agent_id=a.id,
                x=a.position[0],
                y=a.position[1],
                state=a.state.value
            )
            for a in self.agents
        ]

        active_count = len(self.active_agents)
        total_cells = self.config.grid.width * self.config.grid.height

        metrics = {
            'density': active_count / total_cells if total_cells > 0 else 0,
            'completed': self.completed_count,
            'total_agents': len(self.agents),
            'active_agents': active_count,
            'throughput': self.completed_count / max(1, self.current_step),
            'avg_travel_time': (self.total_travel_time / self.completed_count
                                if self.completed_count > 0 else 0)
        }

        return SimulationState(
            step=self.current_step,
            agents=agent_snapshots,
            grid_occupancy=self.grid.occupancy.copy(),
            dynamic_field=self.dynamic_field.field.copy(),
            metrics=metrics
        )

    def is_finished(self) -> bool:
        """Check if simulation should terminate."""
        return (self.current_step >= self.config.max_steps or
                len(self.active_agents) == 0)

    def get_summary(self) -> Dict:
        """Get summary statistics for the simulation."""
        return {
            'total_steps': self.current_step,
            'agents_completed': self.completed_count,
            'agents_total': len(self.agents),
            'agents_remaining': len(self.active_agents),
            'avg_travel_time': (self.total_travel_time / self.completed_count
                                if self.completed_count > 0 else 0),
            'throughput': self.completed_count / max(1, self.current_step)
        }
