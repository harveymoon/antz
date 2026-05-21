"""
One-shot cleanup of dataSave/best:
  1. Delete *_history.json files (fitness 0, skipped by --load).
  2. Parallel-scan every JSON for max BestAnts fitness.
  3. Print distribution and threshold = min + 0.6 * (max - min).
  4. Delete files below threshold + their paired PNGs.
  5. Print top N (default 20) so they can be copied elsewhere.
"""
import os
import json
import sys
import argparse
import multiprocessing as mp


def get_max_fitness(filepath):
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
        fitnesses = [a.get('fitness', 0) for a in best_ants if isinstance(a, dict)]
        return max(fitnesses) if fitnesses else -1
    except Exception:
        return -1


def scan_one(filepath):
    return (filepath, get_max_fitness(filepath))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='dataSave/best')
    parser.add_argument('--top-n', type=int, default=20)
    parser.add_argument('--threshold-pct', type=float, default=0.60,
                        help='Delete files with fitness below min + pct*(max-min). Default 0.60.')
    parser.add_argument('--percentile', type=float, default=None,
                        help='If set, threshold = this percentile of fitness values (e.g. 0.90). Overrides --threshold-pct.')
    parser.add_argument('--min-fitness', type=float, default=None,
                        help='If set, threshold = this absolute fitness value. Overrides --percentile and --threshold-pct.')
    parser.add_argument('--dedupe-runid-in-top', action='store_true',
                        help='In top-N, keep only the highest-fitness file per run ID prefix (legacy files treated as unique).')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument('--top-list-out', default=None,
                        help='If set, write top-N filenames (one per line) to this path.')
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f'Working dir: {data_dir}')

    all_entries = os.listdir(data_dir)
    history_files = [f for f in all_entries if f.endswith('_history.json')]
    json_files = [f for f in all_entries
                  if f.endswith('.json') and not f.endswith('_history.json')]
    png_files = [f for f in all_entries if f.endswith('.png')]

    print(f'Found {len(json_files)} regular JSONs, {len(history_files)} history JSONs, {len(png_files)} PNGs')

    # --- Step 1: history files
    print(f'\n[1] Deleting {len(history_files)} _history.json files...')
    if not args.dry_run:
        for f in history_files:
            os.remove(os.path.join(data_dir, f))
    print(f'    Done.' if not args.dry_run else '    (dry-run)')

    # --- Step 2: parallel scan
    print(f'\n[2] Scanning {len(json_files)} files with {args.workers} workers...')
    paths = [os.path.join(data_dir, f) for f in json_files]
    with mp.Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(scan_one, paths, chunksize=32), 1):
            results.append(r)
            if i % 1000 == 0:
                print(f'    {i}/{len(paths)}')
    print(f'    Done. {len(results)} entries.')

    fitnesses = [r[1] for r in results if r[1] >= 0]
    if not fitnesses:
        print('No valid fitness values found, aborting.')
        return
    fitnesses_sorted = sorted(fitnesses)
    n = len(fitnesses_sorted)
    fmin, fmax = fitnesses_sorted[0], fitnesses_sorted[-1]

    def pct(p):
        return fitnesses_sorted[min(n - 1, int(p * n))]

    print(f'\n[3] Fitness distribution (n={n}):')
    print(f'    min  = {fmin}')
    print(f'    p10  = {pct(0.10)}')
    print(f'    p25  = {pct(0.25)}')
    print(f'    p50  = {pct(0.50)}')
    print(f'    p75  = {pct(0.75)}')
    print(f'    p90  = {pct(0.90)}')
    print(f'    p99  = {pct(0.99)}')
    print(f'    max  = {fmax}')
    if args.min_fitness is not None:
        threshold = args.min_fitness
        print(f'    threshold (absolute --min-fitness) = {threshold:.0f}')
    elif args.percentile is not None:
        threshold = pct(args.percentile)
        print(f'    threshold (p{args.percentile*100:.0f}) = {threshold}')
    else:
        threshold = fmin + args.threshold_pct * (fmax - fmin)
        print(f'    threshold (min + {args.threshold_pct:.0%}*(max-min)) = {threshold:.2f}')

    to_delete = [r[0] for r in results if r[1] < threshold]
    to_keep = [r for r in results if r[1] >= threshold]
    print(f'    would delete {len(to_delete)} JSONs, keep {len(to_keep)}')

    # --- Step 4: delete below threshold + orphan PNGs
    print(f'\n[4] Deleting {len(to_delete)} files below threshold...')
    if not args.dry_run:
        deleted_json = 0
        deleted_png = 0
        for path in to_delete:
            try:
                os.remove(path)
                deleted_json += 1
            except OSError:
                pass
            png_path = path[:-5] + '.png'
            if os.path.exists(png_path):
                try:
                    os.remove(png_path)
                    deleted_png += 1
                except OSError:
                    pass
        # orphan PNGs (no matching kept JSON)
        kept_basenames = {os.path.basename(r[0])[:-5] for r in to_keep}
        for png in png_files:
            base = png[:-4]
            if base not in kept_basenames:
                p = os.path.join(data_dir, png)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        deleted_png += 1
                    except OSError:
                        pass
        print(f'    Deleted {deleted_json} JSON, {deleted_png} PNG.')
    else:
        print('    (dry-run)')

    # --- Step 5: top N
    to_keep.sort(key=lambda r: r[1], reverse=True)
    if args.dedupe_runid_in_top:
        seen_runs = set()
        deduped = []
        for path, fit in to_keep:
            base = os.path.basename(path)
            run_id = base.split('_', 1)[0] if len(base) >= 9 and base[8] == '_' else base
            if run_id in seen_runs:
                continue
            seen_runs.add(run_id)
            deduped.append((path, fit))
        top_n = deduped[:args.top_n]
    else:
        top_n = to_keep[:args.top_n]
    print(f'\n[5] Top {len(top_n)} files by fitness:')
    for path, fit in top_n:
        print(f'    {fit:8.2f}  {os.path.basename(path)}')

    if args.top_list_out:
        with open(args.top_list_out, 'w') as f:
            for path, fit in top_n:
                f.write(os.path.basename(path) + '\n')
        print(f'\n    Top-N filenames written to {args.top_list_out}')


if __name__ == '__main__':
    main()
