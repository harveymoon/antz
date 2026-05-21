"""
Data Organizer for Antz Simulation
Keeps the single best-performing save file from each sim run and deletes the rest.

For each run ID, analyzes all JSON files, keeps the one with the highest
max fitness, and deletes all other files (JSON + associated PNGs).

Filename format: {runID}_{timestamp}.json (e.g., abc12345_20260121-160604.json)
Legacy format: {timestamp}.json (e.g., 20260121-160604.json) - treated as unique runs
"""
import os
import json
import shutil
import argparse
import re
from collections import defaultdict


def parse_filename(filename):
    """Parse a filename to extract run ID and timestamp."""
    match = re.match(r'^([a-f0-9]{8})_(\d{8}-\d{6})\.json$', filename)
    if match:
        return match.group(1), match.group(2)
    
    match = re.match(r'^(\d{8}-\d{6})\.json$', filename)
    if match:
        return match.group(1), match.group(1)
    
    return filename, filename


def get_max_fitness(filepath):
    """Get the maximum fitness from a JSON save file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            best_ants = data.get('BestAnts', [])
        elif isinstance(data, list):
            best_ants = data
        else:
            return -1
            
        if not best_ants:
            return -1
        
        fitnesses = []
        for ant in best_ants:
            if isinstance(ant, dict):
                fitnesses.append(ant.get('fitness', 0))
        
        return max(fitnesses) if fitnesses else -1
    except (json.JSONDecodeError, IOError, TypeError):
        return -1


def main():
    parser = argparse.ArgumentParser(description='Keep best save per run, delete the rest')
    parser.add_argument('--data-dir', type=str, default='dataSave',
                        help='Directory containing JSON save files (default: dataSave)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without deleting files')
    parser.add_argument('--in-place', action='store_true',
                        help='Dedupe within --data-dir itself; do not move files into a best/ subfolder')
    args = parser.parse_args()

    data_dir = args.data_dir
    best_dir = data_dir if args.in_place else os.path.join(data_dir, 'best')
    
    json_files = [f for f in os.listdir(data_dir) 
                  if f.endswith('.json') and os.path.isfile(os.path.join(data_dir, f))]
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files in {data_dir}")
    
    runs = defaultdict(list)
    for filename in json_files:
        run_id, timestamp = parse_filename(filename)
        runs[run_id].append(filename)
    
    print(f"Found {len(runs)} unique runs")
    
    keep_files = []
    delete_files = []
    
    for run_id, files in runs.items():
        best_file = None
        best_fitness = -1
        
        for filename in files:
            filepath = os.path.join(data_dir, filename)
            fitness = get_max_fitness(filepath)
            if fitness > best_fitness:
                best_fitness = fitness
                best_file = filename
        
        if best_file:
            keep_files.append((best_file, best_fitness, run_id))
            for filename in files:
                if filename != best_file:
                    delete_files.append(filename)
        else:
            delete_files.extend(files)
    
    keep_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"BEST FILE PER RUN ({len(keep_files)} runs)")
    print(f"{'='*60}")
    for filename, fitness, run_id in keep_files:
        print(f"  [{run_id}] max_fitness={fitness:.0f}  {filename}")
    
    # Count PNGs that would be deleted
    png_delete_count = 0
    for filename in delete_files:
        png = os.path.join(data_dir, filename.replace('.json', '.png'))
        if os.path.exists(png):
            png_delete_count += 1
    
    # Also count orphan PNGs (no matching JSON)
    all_pngs = [f for f in os.listdir(data_dir) 
                if f.endswith('.png') and os.path.isfile(os.path.join(data_dir, f))]
    json_basenames = {f.replace('.json', '') for f in json_files}
    keep_basenames = {f.replace('.json', '') for f, _, _ in keep_files}
    orphan_pngs = [f for f in all_pngs if f.replace('.png', '') not in keep_basenames]
    
    print(f"\n{'='*60}")
    print(f"CLEANUP SUMMARY")
    print(f"{'='*60}")
    print(f"Keeping: {len(keep_files)} JSON files (best per run)")
    print(f"Deleting: {len(delete_files)} JSON files")
    print(f"Deleting: {len(orphan_pngs)} PNG files")
    
    if args.dry_run:
        print(f"\nDRY RUN - No files deleted")
        return
    
    # Move best files to best/ directory (skip when operating in-place)
    if not args.in_place:
        os.makedirs(best_dir, exist_ok=True)
        print(f"\nMoving {len(keep_files)} best files to {best_dir}...")
        for filename, _, _ in keep_files:
            src = os.path.join(data_dir, filename)
            dst = os.path.join(best_dir, filename)
            if os.path.exists(src):
                shutil.move(src, dst)
    
    # Delete non-best JSON files and their PNGs
    deleted_count = 0
    for filename in delete_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            deleted_count += 1
        png = filepath.replace('.json', '.png')
        if os.path.exists(png):
            os.remove(png)
    
    # Delete orphan PNGs
    for png_file in orphan_pngs:
        png_path = os.path.join(data_dir, png_file)
        if os.path.exists(png_path):
            os.remove(png_path)
    
    print(f"\nDone! Deleted {deleted_count} JSON + {len(orphan_pngs)} PNG files.")
    print(f"Best files are in {best_dir}/")


if __name__ == "__main__":
    main()
