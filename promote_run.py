"""
Find the highest-fitness saves from a specific run ID and copy them to dataSave/best/.
The originals stay in dataSave/ untouched.

Usage: python promote_run.py <runID> [--top 30]
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys


def max_fitness(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ants = data.get('BestAnts', []) if isinstance(data, dict) else []
        fits = [a.get('fitness', 0) for a in ants if isinstance(a, dict)]
        return (filepath, max(fits) if fits else -1)
    except Exception:
        return (filepath, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_id', help='8-char run id, e.g. 1efe5c0d')
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--src', default='dataSave')
    ap.add_argument('--dst', default='dataSave/best')
    args = ap.parse_args()

    files = [os.path.join(args.src, f) for f in os.listdir(args.src)
             if f.startswith(args.run_id + '_') and f.endswith('.json')
             and not f.endswith('_history.json')
             and os.path.isfile(os.path.join(args.src, f))]
    if not files:
        print(f'No files found for run {args.run_id} in {args.src}')
        sys.exit(1)
    print(f'Scanning {len(files)} saves...')

    with mp.Pool(max(1, os.cpu_count() - 1)) as pool:
        results = pool.map(max_fitness, files, chunksize=16)
    results = [r for r in results if r[1] >= 0]
    results.sort(key=lambda r: r[1], reverse=True)
    top = results[:args.top]

    os.makedirs(args.dst, exist_ok=True)
    print(f'\nTop {len(top)} files (fitness, name):')
    copied = 0
    for path, fit in top:
        name = os.path.basename(path)
        dst_path = os.path.join(args.dst, name)
        if not os.path.exists(dst_path):
            shutil.copy2(path, dst_path)
            copied += 1
            mark = '+'
        else:
            mark = '='  # already there
        print(f'  {mark} {fit:>10.0f}  {name}')

    print(f'\nCopied {copied} new files to {args.dst}/. {len(top)-copied} already present.')


if __name__ == '__main__':
    main()
