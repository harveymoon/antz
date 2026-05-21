"""
Static brain audit: do champion ants actually use pheromones?

For each saved JSON in dataSave/best/, take the top ant's brain and check whether
any synapse PATH connects a pheromone input to a motor output (MOVE/TURN).
A brain that lacks any such path cannot be using pheromones to navigate at all.

Brain synapse format: [src, srcSel, dstSel, frc, dest]
  src  : True  -> source is an input sensor (indexed srcSel % NUM_INPUTS)
         False -> source is a neuron        (indexed srcSel % NUM_NEURONS)
  dest : True  -> destination is an output  (indexed dstSel % NUM_OUTPUTS)
         False -> destination is a neuron   (indexed dstSel % NUM_NEURONS)
  frc  : connection weight (-1 .. 1)

Input names (from main.py:2587-2593):
  0  direction    1  prevDir      2  blockedF     3  foodDir     4  foodFront
  5  oscillate    6  random       7  nestPherF    8  foodPherF   9  closeFood
 10  nestPherL   11  nestPherR   12  nestPherB   13  foodPherL  14  foodPherR
 15  foodPherB   16  hiveDir     17  hiveDist    18  carrying
 19  terrainF    20  terrainL    21  terrainR    22  terrainB

Pheromone inputs: {7, 8, 10, 11, 12, 13, 14, 15}
"""
import os
import json
import multiprocessing as mp
from collections import deque, Counter

DATA_DIR = 'dataSave/best'
NUM_INPUTS = 23
NUM_NEURONS = 6
NUM_OUTPUTS = 2
PHER_INPUTS = {7, 8, 10, 11, 12, 13, 14, 15}
NEST_PHER_INPUTS = {7, 10, 11, 12}
FOOD_PHER_INPUTS = {8, 13, 14, 15}
INPUT_NAMES = [
    "direction", "prevDir", "blockedF", "foodDir", "foodFront",
    "oscillate", "random", "nestPherF", "foodPherF", "closeFood",
    "nestPherL", "nestPherR", "nestPherB", "foodPherL", "foodPherR",
    "foodPherB", "hiveDir", "hiveDist", "carrying",
    "terrainF", "terrainL", "terrainR", "terrainB",
]
WEIGHT_EPS = 0.05  # synapses with |weight| < this treated as inactive


def graph_paths(brain, weight_eps=0.0):
    """Return set of pheromone input indices that have any path to an output.
    Edges with |weight| < weight_eps are ignored.
    Nodes encoded as ('I', idx), ('N', idx), ('O', idx)."""
    adj = {}
    for syn in brain:
        if not isinstance(syn, list) or len(syn) < 5:
            continue
        src, srcSel, dstSel, frc, dest = syn[0], syn[1], syn[2], syn[3], syn[4]
        if abs(frc) < weight_eps:
            continue
        if src:
            s = ('I', srcSel % NUM_INPUTS)
        else:
            s = ('N', srcSel % NUM_NEURONS)
        if dest:
            d = ('O', dstSel % NUM_OUTPUTS)
        else:
            d = ('N', dstSel % NUM_NEURONS)
        adj.setdefault(s, set()).add(d)

    reaching = set()
    for p in PHER_INPUTS:
        start = ('I', p)
        if start not in adj:
            continue
        # BFS from this pheromone input
        seen = {start}
        q = deque([start])
        while q:
            n = q.popleft()
            if n[0] == 'O':
                reaching.add(p)
                break
            for nxt in adj.get(n, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return reaching


def analyze_one(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        ants = data.get('BestAnts', [])
        if not ants:
            return None
        # pick top-fitness ant in file
        ants_with_fit = [(a.get('fitness', 0), a) for a in ants if isinstance(a, dict)]
        if not ants_with_fit:
            return None
        ants_with_fit.sort(key=lambda x: x[0], reverse=True)
        top_fit, top_ant = ants_with_fit[0]
        brain = top_ant.get('brain', [])
        if not isinstance(brain, list) or not brain:
            return None

        n_syn = len(brain)

        # Counts of synapses by source type
        pher_src_count = 0
        active_pher_src_count = 0
        for syn in brain:
            if not isinstance(syn, list) or len(syn) < 5:
                continue
            src, srcSel, _, frc, _ = syn[0], syn[1], syn[2], syn[3], syn[4]
            if src and (srcSel % NUM_INPUTS) in PHER_INPUTS:
                pher_src_count += 1
                if abs(frc) >= WEIGHT_EPS:
                    active_pher_src_count += 1

        reaching_raw = graph_paths(brain, weight_eps=0.0)
        reaching_active = graph_paths(brain, weight_eps=WEIGHT_EPS)

        return {
            'file': os.path.basename(filepath),
            'fitness': top_fit,
            'food': top_ant.get('food', 0),
            'n_syn': n_syn,
            'pher_src_count': pher_src_count,
            'active_pher_src_count': active_pher_src_count,
            'pher_inputs_reaching_motor_raw': len(reaching_raw),
            'pher_inputs_reaching_motor_active': len(reaching_active),
            'reaching_active_set': sorted(reaching_active),
        }
    except Exception:
        return None


def main():
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
             if f.endswith('.json') and not f.endswith('_history.json')]
    print(f'Scanning {len(files)} files...')
    with mp.Pool(max(1, os.cpu_count() - 1)) as pool:
        results = [r for r in pool.imap_unordered(analyze_one, files, chunksize=64) if r]
    print(f'Parsed {len(results)} brains.')

    results.sort(key=lambda r: r['fitness'], reverse=True)
    top_n = results[:100]

    print(f'\n=== Top 100 champions: pheromone wiring ===')
    print(f'{"#":>3} {"fitness":>9} {"food":>5} {"syn":>4} {"pher_src":>9} {"act_src":>8} '
          f'{"reach_raw":>9} {"reach_act":>9}  active_pher_inputs')
    for i, r in enumerate(top_n, 1):
        names = ','.join(INPUT_NAMES[p] for p in r['reaching_active_set'])
        print(f'{i:>3} {int(r["fitness"]):>9} {int(r["food"]):>5} {r["n_syn"]:>4} '
              f'{r["pher_src_count"]:>9} {r["active_pher_src_count"]:>8} '
              f'{r["pher_inputs_reaching_motor_raw"]:>9} {r["pher_inputs_reaching_motor_active"]:>9}  {names}')

    # Aggregate stats over top 100
    print(f'\n=== Aggregate stats over top 100 ===')
    n = len(top_n)
    has_any_raw = sum(1 for r in top_n if r['pher_inputs_reaching_motor_raw'] > 0)
    has_any_active = sum(1 for r in top_n if r['pher_inputs_reaching_motor_active'] > 0)
    print(f'Top brains with ANY pheromone->motor path (raw graph):    {has_any_raw}/{n}  ({has_any_raw/n:.0%})')
    print(f'Top brains with ANY pheromone->motor path (|w|>={WEIGHT_EPS}): {has_any_active}/{n}  ({has_any_active/n:.0%})')

    avg_reach = sum(r['pher_inputs_reaching_motor_active'] for r in top_n) / n
    avg_pher_src = sum(r['active_pher_src_count'] for r in top_n) / n
    avg_syn = sum(r['n_syn'] for r in top_n) / n
    print(f'Avg #pher inputs reaching motor (active):  {avg_reach:.2f} of 8')
    print(f'Avg #active synapses sourced from pher:    {avg_pher_src:.2f}')
    print(f'Avg total synapses:                        {avg_syn:.1f}')

    # Most common reaching pheromone inputs
    name_counter = Counter()
    for r in top_n:
        for p in r['reaching_active_set']:
            name_counter[INPUT_NAMES[p]] += 1
    print('\nWhich pheromone sensors reach motors most often (active edges, top 100):')
    for name, c in name_counter.most_common():
        print(f'  {name:<12}  {c:>3}/{n}  ({c/n:.0%})')

    # Bucket by fitness tertile to see if higher-fitness brains are more pheromone-wired
    print('\n=== Stigmergy rate by fitness bucket (over ALL parsed brains) ===')
    sorted_all = sorted(results, key=lambda r: r['fitness'])
    third = len(sorted_all) // 3
    buckets = [
        ('low  ', sorted_all[:third]),
        ('mid  ', sorted_all[third:2*third]),
        ('high ', sorted_all[2*third:]),
    ]
    for label, bucket in buckets:
        if not bucket:
            continue
        bn = len(bucket)
        any_path = sum(1 for r in bucket if r['pher_inputs_reaching_motor_active'] > 0)
        avg_r = sum(r['pher_inputs_reaching_motor_active'] for r in bucket) / bn
        fmin = int(bucket[0]['fitness']); fmax = int(bucket[-1]['fitness'])
        print(f'  {label} fitness [{fmin:>7} .. {fmax:>7}]  n={bn:>4}  '
              f'has_path={any_path/bn:>5.0%}  avg_reach={avg_r:.2f}')

    # Also: compare nest vs food pheromone reach
    nest_reach = sum(1 for r in top_n
                     if any(p in NEST_PHER_INPUTS for p in r['reaching_active_set']))
    food_reach = sum(1 for r in top_n
                     if any(p in FOOD_PHER_INPUTS for p in r['reaching_active_set']))
    print(f'\nIn top 100: nest-pheromone reaches motor: {nest_reach}/{n} '
          f'({nest_reach/n:.0%})   food-pheromone reaches motor: {food_reach}/{n} ({food_reach/n:.0%})')


if __name__ == '__main__':
    main()
