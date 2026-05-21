"""
Analyze a per-death log produced by the antz simulation.

Each line in the JSONL is one ant's death summary:
  step, antID, food_consumed, lifespan, fitness_final,
  fitness_breakdown, brain_size, brain_hash, farthest, carrying_at_death

Run:
  python analyze_deaths.py dataSave/deaths/<runID>.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from statistics import median


def load(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def fmt(n):
    return f"{n:>10,}"


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = int(len(sorted_vals) * p)
    return sorted_vals[min(k, len(sorted_vals) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='Path to deaths JSONL file')
    ap.add_argument('--top', type=int, default=10, help='Show top N ants')
    ap.add_argument('--buckets', type=int, default=4,
                    help='Split run into N time buckets to track trends')
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"File not found: {args.path}")
        sys.exit(1)

    rows = load(args.path)
    if not rows:
        print('No data loaded.')
        return

    n = len(rows)
    last_step = rows[-1]['step']
    print(f'Loaded {n:,} death events spanning steps 0 - {last_step:,}')
    print()

    # === SURVIVAL & TRIP COMPLETION ===
    picked_up = sum(1 for r in rows if any(k in r.get('fitness_breakdown', {}) for k in ('pickup',)))
    delivered = sum(1 for r in rows if any(k in r.get('fitness_breakdown', {}) for k in ('deliver_base',)))
    food_consumed = sum(r.get('food_consumed', 0) for r in rows)
    multi_trip = sum(1 for r in rows if r.get('food_consumed', 0) >= 2)

    lifespans = sorted(r['lifespan'] for r in rows)
    fits = sorted(r['fitness_final'] for r in rows)

    print('=== Survival & trip completion ===')
    print(f'Total ants died:                 {fmt(n)}')
    print(f'Ants that picked up food:        {fmt(picked_up)}  ({picked_up/n:.1%})')
    print(f'Ants that delivered food:        {fmt(delivered)}  ({delivered/n:.1%})')
    print(f'Ants with >= 2 round trips:      {fmt(multi_trip)}  ({multi_trip/n:.1%})')
    print(f'Total food trips completed:      {fmt(food_consumed)}')
    print(f'Trips per ant (mean):            {food_consumed/n:>10.3f}')
    print(f'Lifespan p50/p90/max:            {percentile(lifespans,0.5):>4}  '
          f'{percentile(lifespans,0.9):>4}  {lifespans[-1]:>4}')
    print(f'Fitness p50/p90/max:             {percentile(fits,0.5):>10,}  '
          f'{percentile(fits,0.9):>10,}  {int(fits[-1]):>10,}')
    print()

    # === REWARD DECOMPOSITION (aggregate) ===
    print('=== Reward decomposition (sum of fitness across all deaths) ===')
    src_total = defaultdict(float)
    for r in rows:
        for src, amt in r.get('fitness_breakdown', {}).items():
            src_total[src] += amt
    grand = sum(src_total.values()) or 1
    for src, total in sorted(src_total.items(), key=lambda x: -abs(x[1])):
        print(f'  {src:<22}  {total:>14,.1f}  ({total/grand:.1%})')
    print()

    # === REWARD DECOMPOSITION (top performers only) ===
    rows_by_fit = sorted(rows, key=lambda r: r['fitness_final'], reverse=True)
    top_rows = rows_by_fit[:max(10, n // 100)]  # top 1% or top 10
    print(f'=== Reward decomposition for the top {len(top_rows)} performers ===')
    src_top = defaultdict(float)
    for r in top_rows:
        for src, amt in r.get('fitness_breakdown', {}).items():
            src_top[src] += amt
    grand_top = sum(src_top.values()) or 1
    for src, total in sorted(src_top.items(), key=lambda x: -abs(x[1])):
        print(f'  {src:<22}  {total:>14,.1f}  ({total/grand_top:.1%})')
    print()

    # === TOP PERFORMERS LIST ===
    print(f'=== Top {args.top} ants by final fitness ===')
    print(f'{"step":>8} {"id":>7} {"life":>5} {"food":>4} {"syn":>4} {"fitness":>10}  breakdown')
    for r in rows_by_fit[:args.top]:
        bd = r.get('fitness_breakdown', {})
        bd_str = ', '.join(f'{k}={int(v)}' for k, v in sorted(bd.items(), key=lambda x: -abs(x[1])))
        print(f'{r["step"]:>8} {r["antID"][0]:>7} {r["lifespan"]:>5} '
              f'{r["food_consumed"]:>4} {r["brain_size"]:>4} '
              f'{int(r["fitness_final"]):>10,}  {bd_str}')
    print()

    # === TRENDS OVER TIME (bucketed) ===
    print(f'=== Trends across {args.buckets} time buckets (is the system learning?) ===')
    bucket_size = max(1, last_step // args.buckets)
    buckets = [[] for _ in range(args.buckets)]
    for r in rows:
        b = min(args.buckets - 1, r['step'] // bucket_size)
        buckets[b].append(r)
    print(f'{"bucket":>6} {"steps":>14} {"n_died":>7} {"%pickup":>8} {"%deliver":>9} '
          f'{"%multi":>7} {"med_fit":>8} {"max_fit":>10} {"avg_food":>9}')
    for i, b in enumerate(buckets):
        if not b:
            continue
        bn = len(b)
        bp = sum(1 for r in b if 'pickup' in r.get('fitness_breakdown', {})) / bn
        bd = sum(1 for r in b if 'deliver_base' in r.get('fitness_breakdown', {})) / bn
        bm = sum(1 for r in b if r.get('food_consumed', 0) >= 2) / bn
        bf = sorted(r['fitness_final'] for r in b)
        med = percentile(bf, 0.5)
        mx = bf[-1]
        avg_food = sum(r.get('food_consumed', 0) for r in b) / bn
        print(f'{i:>6} {i*bucket_size:>6,}-{(i+1)*bucket_size:<6,}  {bn:>7,} '
              f'{bp:>7.1%}  {bd:>8.1%}  {bm:>6.1%}  {int(med):>8,} {int(mx):>10,}  {avg_food:>9.3f}')
    print()

    # === BRAIN SIZE vs OUTCOME ===
    print('=== Brain-size vs fitness ===')
    sizes_fit = defaultdict(list)
    for r in rows:
        bucket = (r['brain_size'] // 16) * 16  # group sizes in bins of 16
        sizes_fit[bucket].append(r['fitness_final'])
    for s in sorted(sizes_fit.keys()):
        fs = sorted(sizes_fit[s])
        med = percentile(fs, 0.5)
        mx = fs[-1]
        deliver_rate = sum(1 for r in rows if (r['brain_size'] // 16) * 16 == s
                            and 'deliver_base' in r.get('fitness_breakdown', {})) / len(fs)
        print(f'  brain_size {s:>3}-{s+15:<3}  n={len(fs):>5}  med_fit={int(med):>8,}  '
              f'max_fit={int(mx):>9,}  deliver_rate={deliver_rate:.1%}')


if __name__ == '__main__':
    main()
