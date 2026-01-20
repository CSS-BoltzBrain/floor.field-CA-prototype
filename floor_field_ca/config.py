"""Configuration dataclasses and YAML loader for Floor Field CA simulation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class GridConfig:
    width: int
    height: int


@dataclass
class FloorFieldConfig:
    static_strength: float   # kS parameter
    dynamic_strength: float  # kD parameter
    diffusion_rate: float    # alpha (0.0-1.0)
    decay_rate: float        # delta (0.0-1.0)


@dataclass
class AgentConfig:
    friends_probability: float
    patience: int
    speed: int = 1


@dataclass
class WallSpec:
    wall_type: str  # "rectangle" or "points"
    data: Dict[str, Any]


@dataclass
class HotspotSpec:
    x: int
    y: int
    attraction: float = 1.0


@dataclass
class SpawnZoneSpec:
    zone_type: str  # "line"
    start: Tuple[int, int]
    end: Tuple[int, int]
    direction: str  # "east", "west", "north", "south"


@dataclass
class LayoutConfig:
    walls: List[WallSpec]
    hotspots: List[HotspotSpec]
    spawn_zones: List[SpawnZoneSpec]


@dataclass
class SimulationConfig:
    grid: GridConfig
    max_steps: int
    agent_count: int
    floor_field: FloorFieldConfig
    agents: AgentConfig
    layout: LayoutConfig

    # Export flags (can be overridden by CLI)
    csv_enabled: bool = True
    snapshot_enabled: bool = True
    gif_enabled: bool = False
    quiet: bool = False
    seed: Optional[int] = None
    out_dir: Path = field(default_factory=lambda: Path("./output"))


def _parse_walls(walls_raw: List[Dict]) -> List[WallSpec]:
    """Parse wall specifications from raw YAML data."""
    walls = []
    for w in walls_raw:
        wall_type = w.get('type', 'rectangle')
        if wall_type == 'rectangle':
            data = {
                'x': w['x'],
                'y': w['y'],
                'width': w['width'],
                'height': w['height']
            }
        elif wall_type == 'points':
            data = {'coords': [tuple(c) for c in w['coords']]}
        else:
            raise ValueError(f"Unknown wall type: {wall_type}")
        walls.append(WallSpec(wall_type=wall_type, data=data))
    return walls


def _parse_hotspots(hotspots_raw: List[Dict]) -> List[HotspotSpec]:
    """Parse hotspot specifications from raw YAML data."""
    return [
        HotspotSpec(
            x=h['x'],
            y=h['y'],
            attraction=h.get('attraction', 1.0)
        )
        for h in hotspots_raw
    ]


def _parse_spawn_zones(zones_raw: List[Dict]) -> List[SpawnZoneSpec]:
    """Parse spawn zone specifications from raw YAML data."""
    return [
        SpawnZoneSpec(
            zone_type=z.get('type', 'line'),
            start=tuple(z['start']),
            end=tuple(z['end']),
            direction=z['direction']
        )
        for z in zones_raw
    ]


def load_config(config_path: Path) -> SimulationConfig:
    """Load and validate YAML configuration file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Parse grid config
    grid = GridConfig(
        width=raw['grid']['width'],
        height=raw['grid']['height']
    )

    # Parse floor field config
    ff_raw = raw['floor_field']
    floor_field = FloorFieldConfig(
        static_strength=ff_raw['static_strength'],
        dynamic_strength=ff_raw['dynamic_strength'],
        diffusion_rate=ff_raw['diffusion_rate'],
        decay_rate=ff_raw['decay_rate']
    )

    # Parse agent config
    agents_raw = raw['agents']
    agents = AgentConfig(
        friends_probability=agents_raw.get('friends_probability', 0.0),
        patience=agents_raw.get('patience', 50),
        speed=agents_raw.get('speed', 1)
    )

    # Parse layout
    layout_raw = raw['layout']
    layout = LayoutConfig(
        walls=_parse_walls(layout_raw.get('walls', [])),
        hotspots=_parse_hotspots(layout_raw.get('hotspots', [])),
        spawn_zones=_parse_spawn_zones(layout_raw.get('spawn_zones', []))
    )

    # Parse simulation config
    sim_raw = raw['simulation']

    # Parse export config (optional)
    export_raw = raw.get('export', {})

    return SimulationConfig(
        grid=grid,
        max_steps=sim_raw['max_steps'],
        agent_count=sim_raw['agent_count'],
        floor_field=floor_field,
        agents=agents,
        layout=layout,
        csv_enabled=export_raw.get('csv', True),
        snapshot_enabled=export_raw.get('snapshot', True),
        gif_enabled=export_raw.get('gif', False)
    )
