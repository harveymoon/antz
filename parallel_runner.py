"""
Parallel Runner for Antz Simulation
Launches multiple headless instances to train in parallel.
All instances share the same dataSave folder, so best ants get merged.
"""
import subprocess
import sys
import time
import signal
import os
from multiprocessing import Process, Event
import argparse


def run_instance(instance_id, stop_event, load_data=True):
    """Run a single headless simulation instance"""
    args = [sys.executable, "main.py", "--headless"]
    if load_data:
        args.append("--load")
    
    # Set a unique random seed based on instance ID and time for diversity
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(instance_id + int(time.time()) % 10000)
    
    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Stream output with instance prefix
        while not stop_event.is_set():
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    # Process died - print exit code
                    exit_code = process.returncode
                    print(f"[{instance_id}] Process exited with code: {exit_code}")
                    break
                time.sleep(0.1)
                continue
            line_stripped = line.rstrip()
            # Print important lines and any errors/exceptions
            if any(x in line for x in ['Updates/sec', 'Best Fitness', 'STAGNATION', 'Saving', 'Leaderboard']):
                print(f"[{instance_id}] {line_stripped}")
            elif any(x in line for x in ['Error', 'Exception', 'Traceback', 'error', 'failed', 'File "']):
                print(f"[{instance_id}] ERROR: {line_stripped}")
        
        # Graceful shutdown
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            
    except Exception as e:
        print(f"[{instance_id}] Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run multiple Antz simulations in parallel')
    parser.add_argument('-n', '--num-instances', type=int, default=10,
                        help='Number of parallel instances (default: 10)')
    parser.add_argument('--no-load', action='store_true',
                        help='Start fresh without loading saved ants')
    parser.add_argument('--duration', type=int, default=0,
                        help='Run duration in seconds (0 = run until Ctrl+C)')
    args = parser.parse_args()
    
    num_instances = args.num_instances
    load_data = not args.no_load
    
    print(f"=" * 60)
    print(f"PARALLEL ANTZ RUNNER")
    print(f"=" * 60)
    print(f"Launching {num_instances} parallel instances...")
    print(f"Load saved data: {load_data}")
    print(f"All instances share dataSave/ folder")
    print(f"Press Ctrl+C to stop all instances and save")
    print(f"=" * 60)
    
    # Create stop event for graceful shutdown
    stop_event = Event()
    
    # Launch all instances
    processes = []
    for i in range(num_instances):
        p = Process(target=run_instance, args=(i, stop_event, load_data))
        p.start()
        processes.append(p)
        print(f"Started instance {i}")
        time.sleep(0.5)  # Stagger launches to avoid file conflicts
    
    print(f"\nAll {num_instances} instances running!")
    print("Monitoring... (Ctrl+C to stop)\n")
    
    start_time = time.time()
    
    try:
        while True:
            time.sleep(5)
            elapsed = time.time() - start_time
            alive = sum(1 for p in processes if p.is_alive())
            print(f"[Monitor] Running: {alive}/{num_instances} instances | Elapsed: {elapsed/60:.1f} min")
            
            # Check duration limit
            if args.duration > 0 and elapsed >= args.duration:
                print(f"\nDuration limit reached ({args.duration}s). Stopping...")
                break
            
            # Restart any crashed instances
            for i, p in enumerate(processes):
                if not p.is_alive():
                    print(f"[Monitor] Instance {i} died, restarting...")
                    p = Process(target=run_instance, args=(i, stop_event, load_data))
                    p.start()
                    processes[i] = p
                    
    except KeyboardInterrupt:
        print("\n\nShutting down all instances...")
    
    # Signal all instances to stop
    stop_event.set()
    
    # Wait for graceful shutdown
    print("Waiting for instances to save and exit...")
    for i, p in enumerate(processes):
        p.join(timeout=10)
        if p.is_alive():
            print(f"Force killing instance {i}...")
            p.terminate()
            p.join(timeout=2)
    
    print("\nAll instances stopped.")
    print(f"Total runtime: {(time.time() - start_time)/60:.1f} minutes")
    print("Check dataSave/ for merged results.")


if __name__ == "__main__":
    main()
