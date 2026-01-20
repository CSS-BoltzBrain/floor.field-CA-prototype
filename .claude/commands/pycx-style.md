# PyCX Style Simulation Skill

You are an expert in the PyCX programming style for complex systems simulation. PyCX is a Python-based sample code repository for complex systems research and education, created by Hiroki Sayama at Binghamton University.

## Core Philosophy

**From the PyCX project:**
> "The core philosophy of PyCX is placed on the simplicity, readability, generalizability, and pedagogical values of sample codes. This is often achieved even at the cost of computational speed, efficiency, or maintainability."

**Key Design Principles:**
- **Simplicity over efficiency**: Code should be easy to understand, even if slower
- **Global variables are OK**: Used intentionally for educational clarity
- **Minimal OOP**: Simple classes, no complex inheritance hierarchies
- **Direct matplotlib**: Use `from pylab import *` for interactive plotting
- **Three-function pattern**: All simulations follow initialize/observe/update

## The Three-Function Pattern

Every PyCX simulation implements exactly three functions:

```python
import pycxsimulator
from pylab import *

# Model parameters as global variables
param1 = 10
param2 = 0.5

def initialize():
    """Set up initial system state using global variables."""
    global state1, state2, time
    time = 0
    state1 = zeros((50, 50))
    state2 = rand(100, 2)

def observe():
    """Visualize current state using matplotlib."""
    global state1, state2, time
    cla()  # Clear current axes
    # Plot current state...
    title('t = ' + str(time))

def update():
    """Update system state for one discrete time step."""
    global state1, state2, time
    time += 1
    # Update states...

# Start the interactive GUI
pycxsimulator.GUI().start(func=[initialize, observe, update])
```

## Model Categories

PyCX organizes models by prefix:

| Prefix | Category | Description |
|--------|----------|-------------|
| `abm-` | Agent-Based Models | Agents with position, state, behaviors |
| `ca-`  | Cellular Automata | Grid-based, discrete state systems |
| `ds-`  | Dynamical Systems | Differential equations, phase space |
| `net-` | Network Models | Graph-based dynamics |
| `pde-` | Partial Differential Equations | Spatial continuous dynamics |

## Agent-Based Model Template (abm-)

```python
import pycxsimulator
from pylab import *

# Model parameters
n = 100           # number of agents
speed = 0.01      # movement speed
noise = 0.1       # random noise level

class agent:
    """Simple agent class with minimal attributes."""
    def __init__(self):
        self.x = rand(2)      # position as 2D array
        self.v = rand(2) - 0.5  # velocity
        self.state = 0        # discrete state if needed

    def move(self):
        """Update position based on velocity."""
        self.x += self.v * speed
        # Periodic boundary conditions
        self.x = self.x % 1.0

def initialize():
    global agents, time
    time = 0
    agents = [agent() for _ in range(n)]

def observe():
    global agents, time
    cla()
    x = [a.x[0] for a in agents]
    y = [a.x[1] for a in agents]
    plot(x, y, 'bo', markersize=3)
    axis('image')
    axis([0, 1, 0, 1])
    title('t = ' + str(time))

def update():
    global agents, time
    time += 1

    # Compute interactions (do NOT update during iteration)
    for a in agents:
        # Compute new velocity based on neighbors...
        neighbors = [b for b in agents if b is not a
                     and norm(a.x - b.x) < 0.1]
        if neighbors:
            # Example: average neighbor positions
            center = mean([b.x for b in neighbors], axis=0)
            a.new_v = (center - a.x) * 0.1
        else:
            a.new_v = a.v

    # Apply updates after all computations
    for a in agents:
        a.v = a.new_v + uniform(-noise, noise, 2)
        a.move()

pycxsimulator.GUI().start(func=[initialize, observe, update])
```

## Cellular Automaton Template (ca-)

```python
import pycxsimulator
from pylab import *

# Model parameters
width = 50
height = 50
init_prob = 0.2

def initialize():
    global config, nextConfig, time
    time = 0
    config = zeros([height, width])
    for x in range(width):
        for y in range(height):
            config[y, x] = 1 if random() < init_prob else 0
    nextConfig = zeros([height, width])

def observe():
    global config, time
    cla()
    imshow(config, vmin=0, vmax=1, cmap=cm.binary)
    axis('image')
    title('t = ' + str(time))

def update():
    global config, nextConfig, time
    time += 1

    for x in range(width):
        for y in range(height):
            # Count neighbors (Moore neighborhood)
            count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    count += config[(y+dy) % height, (x+dx) % width]

            # Apply rules
            state = config[y, x]
            if state == 0 and count == 3:
                nextConfig[y, x] = 1
            elif state == 1 and (count < 2 or count > 3):
                nextConfig[y, x] = 0
            else:
                nextConfig[y, x] = state

    # Swap buffers
    config, nextConfig = nextConfig, config

pycxsimulator.GUI().start(func=[initialize, observe, update])
```

## Dynamical Systems Template (ds-)

```python
from pylab import *

# Model parameters
a = 0.1
b = 0.2
Dt = 0.01  # Time step

def initialize():
    global x, y, xresult, yresult, t, timesteps
    x, y = 1.0, 1.0
    xresult = [x]
    yresult = [y]
    t = 0.0
    timesteps = [t]

def observe():
    global x, xresult, y, yresult, t, timesteps
    xresult.append(x)
    yresult.append(y)
    timesteps.append(t)

def update():
    global x, y, t
    # Euler integration
    dx = a * x - b * x * y
    dy = -a * y + b * x * y
    x += dx * Dt
    y += dy * Dt
    t += Dt

# Run simulation
initialize()
while t < 100:
    update()
    observe()

# Plot results
subplot(2, 1, 1)
plot(timesteps, xresult, 'b', label='x')
plot(timesteps, yresult, 'r', label='y')
xlabel('Time')
legend()

subplot(2, 1, 2)
plot(xresult, yresult, 'g')
xlabel('x')
ylabel('y')
title('Phase Space')

tight_layout()
show()
```

## Network Model Template (net-)

```python
import pycxsimulator
from pylab import *
import networkx as nx

def initialize():
    global g, time
    time = 0
    g = nx.barabasi_albert_graph(100, 2)
    g.pos = nx.spring_layout(g)
    for i in g.nodes:
        g.nodes[i]['state'] = 1 if random() < 0.1 else 0

def observe():
    global g, time
    cla()
    node_colors = [g.nodes[i]['state'] for i in g.nodes]
    nx.draw(g, pos=g.pos, node_color=node_colors,
            cmap=cm.RdYlGn, vmin=0, vmax=1, node_size=50)
    title('t = ' + str(time))

p_spread = 0.3
p_recover = 0.1

def update():
    global g, time
    time += 1

    # Select random node
    node = choice(list(g.nodes))

    if g.nodes[node]['state'] == 0:  # Susceptible
        # Check if any neighbor is infected
        for neighbor in g.neighbors(node):
            if g.nodes[neighbor]['state'] == 1:
                if random() < p_spread:
                    g.nodes[node]['state'] = 1
                break
    else:  # Infected
        if random() < p_recover:
            g.nodes[node]['state'] = 0

pycxsimulator.GUI().start(func=[initialize, observe, update])
```

## Vicsek Model (Flocking/Alignment)

The Vicsek model is closely related to Boids - it implements velocity alignment:

```python
import pycxsimulator
from pylab import *

n = 500         # number of agents
v = 1           # speed of movement
r = 0.05        # perception range
r2 = r**2       # perception range squared
k = int(1/r)    # spatial bins
Dt = 0.01       # time step
eta = 0.1       # noise level

class agent:
    def __init__(self):
        self.x = rand(2)
        self.th = random() * 2 * pi  # heading angle

    def move(self):
        self.x += v * array([cos(self.th), sin(self.th)]) * Dt
        self.x = self.x % 1.0  # periodic boundaries

def initialize():
    global agents
    agents = [agent() for _ in range(n)]

def observe():
    global agents
    cla()
    plot([a.x[0] for a in agents], [a.x[1] for a in agents], 'bo', ms=2)
    axis('image')
    axis([0, 1, 0, 1])

def spatial_bin(a):
    """Return bin indices for spatial hashing."""
    return int(floor(a.x[0] / r)), int(floor(a.x[1] / r))

def update():
    global agents

    # Build spatial hash map for efficient neighbor finding
    spatial_map = [[[] for _ in range(k)] for _ in range(k)]
    for a in agents:
        i, j = spatial_bin(a)
        spatial_map[i][j].append(a)

    # Compute new headings
    for a in agents:
        i, j = spatial_bin(a)
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                neighbors.extend(spatial_map[(i+di)%k][(j+dj)%k])

        # Filter by actual distance
        neighbors = [b for b in neighbors
                     if sum((a.x - b.x)**2) < r2]

        # Average heading of neighbors
        avg = mean([[cos(b.th), sin(b.th)] for b in neighbors], axis=0)
        a.new_th = arctan2(avg[1], avg[0]) + uniform(-eta/2, eta/2)

    # Apply updates
    for a in agents:
        a.th = a.new_th
        a.move()

pycxsimulator.GUI().start(func=[initialize, observe, update])
```

## Converting Boids to PyCX Style

To analyze your Boids simulation using PyCX style:

```python
import pycxsimulator
from pylab import *

# Boids parameters (PyCX style: global variables)
n_boids = 50
bounds = 1.0
max_speed = 0.02
w_sep = 1.5
w_align = 1.0
w_coh = 1.0
r_sep = 0.05
r_align = 0.1
r_coh = 0.1

class boid:
    def __init__(self):
        self.x = rand(2) * bounds
        angle = random() * 2 * pi
        self.v = array([cos(angle), sin(angle)]) * max_speed * random()

    def limit_speed(self):
        speed = norm(self.v)
        if speed > max_speed:
            self.v = self.v / speed * max_speed

def initialize():
    global boids, time
    time = 0
    boids = [boid() for _ in range(n_boids)]

def observe():
    global boids, time
    cla()
    x = [b.x[0] for b in boids]
    y = [b.x[1] for b in boids]
    u = [b.v[0] for b in boids]
    v = [b.v[1] for b in boids]
    scatter(x, y, c='blue', s=10)
    quiver(x, y, u, v, color='gray', alpha=0.5, scale=1)
    axis('image')
    axis([0, bounds, 0, bounds])
    title('Boids t = ' + str(time))

def get_neighbors(b, radius):
    return [other for other in boids
            if other is not b and norm(other.x - b.x) < radius]

def separation(b):
    neighbors = get_neighbors(b, r_sep)
    if not neighbors:
        return zeros(2)
    center = mean([n.x for n in neighbors], axis=0)
    away = b.x - center
    n = norm(away)
    return away / n if n > 0 else zeros(2)

def alignment(b):
    neighbors = get_neighbors(b, r_align)
    if not neighbors:
        return zeros(2)
    avg_v = mean([n.v for n in neighbors], axis=0)
    n = norm(avg_v)
    return avg_v / n if n > 0 else zeros(2)

def cohesion(b):
    neighbors = get_neighbors(b, r_coh)
    if not neighbors:
        return zeros(2)
    center = mean([n.x for n in neighbors], axis=0)
    toward = center - b.x
    n = norm(toward)
    return toward / n if n > 0 else zeros(2)

def update():
    global boids, time
    time += 1

    # Compute all accelerations first
    for b in boids:
        b.accel = (w_sep * separation(b) +
                   w_align * alignment(b) +
                   w_coh * cohesion(b))

    # Apply updates
    for b in boids:
        b.v += b.accel * 0.1
        b.limit_speed()
        b.x += b.v
        b.x = b.x % bounds  # periodic boundaries

pycxsimulator.GUI().start(func=[initialize, observe, update])
```

## Metrics and Analysis (PyCX Style)

```python
from pylab import *

# Add to your simulation:

def compute_polarization():
    """Alignment of velocities (0=random, 1=perfect)."""
    global boids
    speeds = array([norm(b.v) for b in boids])
    speeds[speeds == 0] = 1e-10
    normalized = array([b.v / norm(b.v) if norm(b.v) > 0
                        else zeros(2) for b in boids])
    return norm(mean(normalized, axis=0))

def compute_cohesion():
    """Standard deviation of positions."""
    global boids
    positions = array([b.x for b in boids])
    return mean(std(positions, axis=0))

# In observe(), add metrics plotting:
def observe():
    global boids, time, pol_history, coh_history

    # Main visualization
    subplot(1, 2, 1)
    cla()
    # ... plot boids ...

    # Metrics panel
    subplot(1, 2, 2)
    cla()
    pol_history.append(compute_polarization())
    coh_history.append(compute_cohesion())
    plot(pol_history, 'b-', label='Polarization')
    plot(coh_history, 'r-', label='Cohesion')
    legend()
    xlabel('Time')
    ylim(0, 1)
```

## Running Without GUI (Batch Mode)

For parameter sweeps or headless execution:

```python
from pylab import *

# ... define initialize, observe, update ...

# Run without GUI
initialize()
results = []

for step in range(1000):
    update()
    if step % 10 == 0:
        results.append({
            'step': step,
            'polarization': compute_polarization(),
            'cohesion': compute_cohesion()
        })

# Plot results
steps = [r['step'] for r in results]
pol = [r['polarization'] for r in results]
coh = [r['cohesion'] for r in results]

plot(steps, pol, label='Polarization')
plot(steps, coh, label='Cohesion')
legend()
show()
```

## Key Differences: PyCX vs OOP Style

| Aspect | PyCX Style | OOP Style (your boids.py) |
|--------|------------|---------------------------|
| State | Global variables | Class attributes |
| Agents | Simple class, global list | Flock class manages list |
| Visualization | Inline in `observe()` | Separate Simulation class |
| Metrics | Functions accessing globals | Methods on Flock |
| GUI | pycxsimulator.GUI() | matplotlib.animation |
| Philosophy | Educational clarity | Production-ready |

## Best Practices

1. **Keep functions short**: Each function should do one thing
2. **Use comments liberally**: Explain the "why", not just the "what"
3. **Avoid premature optimization**: Clarity over speed
4. **Global variables are fine**: For educational code, they reduce boilerplate
5. **Test visually first**: Use the GUI before analyzing metrics
6. **Double-buffer updates**: Compute all new states, then apply all at once

## References

- Sayama, H. (2013) "PyCX: A Python-based simulation code repository for complex systems education." Complex Adaptive Systems Modeling 1:2
- Project website: https://github.com/hsayama/PyCX
- OpenSUNY Textbook: http://tinyurl.com/imacsbook
