"""
Benchmark script for Antz simulation
Runs the simulation for a fixed number of steps and measures performance.
"""
import time
import sys
import argparse

def run_benchmark(module_name, num_steps=1000, num_ants=500):
    """Run benchmark on specified module"""
    
    # Import the module dynamically
    if module_name == "main":
        import main as sim_module
    elif module_name == "main_original":
        import main_original as sim_module
    elif module_name == "main_perf":
        import main_perf as sim_module
    else:
        raise ValueError(f"Unknown module: {module_name}")
    
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {module_name}")
    print(f"{'='*60}")
    print(f"Steps: {num_steps}, Target Ants: {num_ants}")
    
    # Create the game/colony without pygame display
    # We'll directly create an AntColony for pure simulation benchmarking
    
    screen_size = (800, 600)
    tile_size = 4
    
    print("\nInitializing colony...")
    init_start = time.perf_counter()
    colony = sim_module.AntColony(screen_size, num_ants, tile_size)
    init_time = time.perf_counter() - init_start
    print(f"Colony initialization: {init_time:.3f}s")
    
    # Pre-populate with some ants
    print(f"Spawning initial ants...")
    spawn_start = time.perf_counter()
    for _ in range(min(100, num_ants)):
        colony.Repopulate()
    spawn_time = time.perf_counter() - spawn_start
    print(f"Initial spawn: {spawn_time:.3f}s, Ants: {len(colony.ants)}")
    
    # Warm up - run a few steps to stabilize
    print("Warming up (100 steps)...")
    for _ in range(100):
        colony.update()
    
    print(f"After warmup: {len(colony.ants)} ants")
    
    # Main benchmark
    print(f"\nRunning benchmark ({num_steps} steps)...")
    
    update_times = []
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        colony.update()
        step_time = time.perf_counter() - step_start
        update_times.append(step_time)
        
        # Progress update every 200 steps
        if (step + 1) % 200 == 0:
            elapsed = time.perf_counter() - start_time
            rate = (step + 1) / elapsed
            print(f"  Step {step + 1}/{num_steps} - {rate:.1f} updates/sec - {len(colony.ants)} ants")
    
    total_time = time.perf_counter() - start_time
    
    # Calculate statistics
    avg_time = sum(update_times) / len(update_times)
    min_time = min(update_times)
    max_time = max(update_times)
    updates_per_sec = num_steps / total_time
    
    # Sort for percentiles
    sorted_times = sorted(update_times)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {module_name}")
    print(f"{'='*60}")
    print(f"Total time:      {total_time:.3f}s")
    print(f"Updates/second:  {updates_per_sec:.2f}")
    print(f"Avg step time:   {avg_time*1000:.3f}ms")
    print(f"Min step time:   {min_time*1000:.3f}ms")
    print(f"Max step time:   {max_time*1000:.3f}ms")
    print(f"P50 step time:   {p50*1000:.3f}ms")
    print(f"P95 step time:   {p95*1000:.3f}ms")
    print(f"P99 step time:   {p99*1000:.3f}ms")
    print(f"Final ant count: {len(colony.ants)}")
    print(f"{'='*60}\n")
    
    return {
        'module': module_name,
        'total_time': total_time,
        'updates_per_sec': updates_per_sec,
        'avg_step_ms': avg_time * 1000,
        'p50_ms': p50 * 1000,
        'p95_ms': p95 * 1000,
        'p99_ms': p99 * 1000,
        'final_ants': len(colony.ants)
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Antz simulation')
    parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps')
    parser.add_argument('--ants', type=int, default=500, help='Target number of ants')
    parser.add_argument('--module', type=str, default='main', 
                        choices=['main', 'main_original', 'main_perf', 'both'],
                        help='Which module to benchmark')
    args = parser.parse_args()
    
    results = []
    
    if args.module == 'both':
        # Run both and compare
        # Note: Need to restart Python between runs for clean comparison
        # For now, run them sequentially
        print("Running comparison benchmark...")
        print("Note: For most accurate results, run each separately")
        
        results.append(run_benchmark('main_original', args.steps, args.ants))
        
        # Force garbage collection between runs
        import gc
        gc.collect()
        
        results.append(run_benchmark('main', args.steps, args.ants))
        
        # Comparison
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        orig = results[0]
        new = results[1]
        speedup = orig['total_time'] / new['total_time']
        print(f"Original: {orig['updates_per_sec']:.2f} updates/sec")
        print(f"New:      {new['updates_per_sec']:.2f} updates/sec")
        print(f"Speedup:  {speedup:.2f}x")
        print("="*60)
    else:
        results.append(run_benchmark(args.module, args.steps, args.ants))
    
    return results


if __name__ == "__main__":
    main()
