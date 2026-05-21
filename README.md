# Antz - Evolving Ant Colony Simulation

A genetic algorithm-based ant colony simulation where ants evolve neural network "brains" to find food and return it to the hive. Over time, successful ants reproduce and pass on their neural patterns, leading to emergent foraging behaviors.

## How It Works

Each ant has a simple neural network brain with:
- **Inputs**: 23 sensors for direction, food location, pheromone trails (nest/food, 4 directions each), terrain density, hive direction/distance, carrying food state
- **Hidden Neurons**: 6 internal neurons that store and process information
- **Outputs**: Movement (forward/backward) and turning

Ants that successfully find food and return it to the hive gain fitness points. When ants die, the best performers are added to a leaderboard (top 200). New ants are spawned using mutations and crossovers of the best-performing brains.

## Installation

```bash
pip install pygame numpy
```

## Usage

```bash
python main.py [options]
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--load` | Load best ants from saved data files in `dataSave/best/` folder |
| `--headless` | Run without graphics. Auto-saves every 60 seconds. Press `Ctrl+C` to stop |
| `--test` | Test mode: 1 ant, 1 FPS, brain debug enabled |
| `--paths` | Start with trail/path view mode enabled |
| `--debug` | Enable debug mode with extra console output |
| `--pi` | Raspberry Pi mode: fullscreen, optimized for small screens |
| `--scale N` | Pixel scale factor for Pi / fullscreen mode (default: 2). `1` = native, higher = render at lower res and stretch |
| `--fullscreen` | Fullscreen on a desktop monitor; resolution auto-detected |
| `--monitor N` | Which monitor to use with `--fullscreen` (0=primary, 1=second, ...) |

### Examples

```bash
# Start fresh simulation with graphics
python main.py

# Continue training from saved ants
python main.py --load

# Fast training without graphics
python main.py --headless --load

# Debug a single ant step-by-step
python main.py --test

# Run on Raspberry Pi
python main.py --pi --load
```

## Parallel Training

Run multiple instances simultaneously for faster evolution:

```bash
# Run 10 parallel headless instances
python parallel_runner.py -n 10

# Run 4 instances for 30 minutes
python parallel_runner.py -n 4 --duration 1800

# Start fresh (no load)
python parallel_runner.py --no-load
```

Each instance gets a unique Run ID. All instances share the `dataSave/` folder.

## Data Organization

Use `organize_data.py` to clean up save files:

```bash
# Preview changes (dry run)
python organize_data.py --dry-run

# Keep top 1% performers, archive the rest
python organize_data.py --top-percent 1

# Keep only files with fitness >= 50000
python organize_data.py --min-fitness 50000

# Delete archive instead of moving
python organize_data.py --delete-archive
```

Or double-click `organize_best.bat` to keep top 1%.

**File format**: `{runID}_{timestamp}.json` (e.g., `a1b2c3d4_20260121-160604.json`)

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Save current best ants |
| `R` | Reset world (regenerate terrain, clear pheromones) |
| `T` | Toggle trail/path view mode |
| `P` | Toggle brain debug overlay (slows to 1 FPS) |
| `L` | Toggle lifeline view (per-ant timelines: pickup `X`, deliver `O`, death `\|`) |
| `F` | (Lifeline view only) Toggle follow-bottom mode |
| `H` | Halve the maximum ant count |
| `A` | Increase maximum ant count by 1000 |
| `U` | Fast-forward 1000 simulation steps |
| `C` | Cull bottom 50% of leaderboard |
| `+` / `=` | Increase target FPS |
| `-` | Decrease target FPS |
| `X` | Shutdown Raspberry Pi (Pi mode only) |

## Mouse Controls

| Action | Effect |
|--------|--------|
| **Right-click** | Add +5 food at mouse position |
| **Hover top edge** | Show leaderboard stats |
| **Hover bottom edge** | Show fitness timeline |
| **Left-click** | Show cell info (food, pheromones, terrain) |

## Display

Top-right corner shows:
- `FPS: 5.2/6` - Current FPS / Target FPS
- `Ants: 450/500` - Current ants / Max ants
- `Food: 1234` - Total food on field

## Data Files

Saved data is stored in the `dataSave/` folder:
- **dataSave/best/**: Best performers (load from here)
- **dataSave/archive/**: Older/lower-performing files
- **dataSave/deaths/**: Per-death JSONL log, one file per Run ID
- Each `BestAnts` JSON contains top 200 ant brains with fitness scores

## Observability: per-death log + analyzer

Every ant death appends one JSON line to `dataSave/deaths/{runID}.jsonl`. Each line includes:

- `step`, `antID`, `birth_step`, `lifespan`
- `food_consumed`, `farthest`, `carrying_at_death`
- `fitness_final` and the **`fitness_breakdown`** — fitness contribution per source: `pickup`, `deliver_base`, `deliver_distance`, `trail_step`, `death_nav`, `death_exploration`
- `brain_size`, `brain_hash`, `color` (RGB)
- `events` — list of `[step, "pickup"]` / `[step, "deliver"]` entries

This lets you answer the questions you actually care about — *what % of ants delivered food? what reward source dominates the winners?* — without staring at the simulation for days.

### Analyze a run

After (or during) a run:

```bash
python analyze_deaths.py dataSave/deaths/<runID>.jsonl
```

The analyzer prints six sections:

1. **Survival & trip completion** — % pickup, % deliver, multi-trip rate, lifespan/fitness percentiles
2. **Aggregate reward decomposition** — which fitness sources drove the entire run
3. **Top performers' reward decomposition** — which sources are *being selected for* in the winners
4. **Top N ants** by fitness with full breakdown
5. **Time-bucketed trends** — does %deliver / median fitness rise over the run? (this is your "is learning happening" answer)
6. **Brain-size vs fitness vs deliver rate** — does brain capacity correlate with success?

### Live lifeline view in-app

Press `L` during a graphical run to overlay a scrolling list of dead ants. Each row is a brain-colored line scaled to the ant's lifespan, with `X` markers for food pickups, `O` for deliveries, and a `|` cap at death.

Default mode is `[SCROLL]` — the viewport is anchored to the rows you scrolled to, so new deaths appended below don't move what you're reading. Press `F` to toggle `[FOLLOW]` mode where the view sticks to the latest deaths. Any scroll-wheel input drops follow mode. `HOME` jumps to the oldest entry, `END` snaps to the newest without enabling follow.

The death log file persists across runs, so you can also analyze prior runs offline.

## Ant Types

The second element of ant IDs indicates creation method:
- `L` - Loaded from file
- `M` - Mutated from a best ant
- `CL` - Clone of a best ant
- `CH` - Child (crossover of two parents)
- `EM` - Evolution mutated (heavy mutation during stagnation)
- `HY` - Hybrid (mixed from multiple parents)

## Tips

- Use `parallel_runner.py` for overnight training (10+ instances)
- Run `organize_best.bat` periodically to clean up data files
- Press `C` to cull leaderboard if evolution stagnates
- Adjust target FPS with `+`/`-` to balance speed vs ant count
- Right-click to add food and guide ants during debugging
