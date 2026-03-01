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
| `--scale N` | Pixel scale factor for Pi mode (default: 2) |

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
- Each JSON contains top 200 ant brains with fitness scores

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
