"""
Audit dataSave/best/ for duplicate brains across save files.

Each save file contains a list of BestAnts. We hash every brain we see and
report:
  - which brains appear in the most files (dominance)
  - their synapse count and fitness
  - per-file: how many of its BestAnts are unique vs duplicates of other files
"""
import json
import multiprocessing as mp
import os
from collections import Counter, defaultdict


DATA_DIR = 'dataSave/best'


def brain_key(brain):
    try:
        return hash(tuple(tuple(g) for g in brain))
    except Exception:
        return None


def scan_one(path):
    """Return (filename, list of (brain_key, synapse_count, fitness)) for top ants."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return (os.path.basename(path), [])
        ants = data.get('BestAnts', [])
        out = []
        for a in ants:
            if not isinstance(a, dict):
                continue
            brain = a.get('brain', [])
            if not isinstance(brain, list) or not brain:
                continue
            k = brain_key(brain)
            if k is None:
                continue
            out.append((k, len(brain), a.get('fitness', 0)))
        return (os.path.basename(path), out)
    except Exception:
        return (os.path.basename(path), [])


def main():
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
             if f.endswith('.json') and not f.endswith('_history.json')
             and os.path.isfile(os.path.join(DATA_DIR, f))]
    print(f'Scanning {len(files)} files in {DATA_DIR}/...')

    with mp.Pool(max(1, os.cpu_count() - 1)) as pool:
        results = pool.map(scan_one, files, chunksize=16)

    # Aggregate
    file_count_per_brain = Counter()        # brain_key -> #files containing it
    instances_per_brain = Counter()         # brain_key -> #BestAnts entries (across all files)
    brain_meta = {}                         # brain_key -> (synapse_count, max_fitness)
    file_to_brains = {}                     # filename -> set of brain_keys
    all_files_with_brain = defaultdict(list)

    for fname, ants in results:
        if not ants:
            continue
        unique_in_file = set()
        for k, n_syn, fit in ants:
            instances_per_brain[k] += 1
            unique_in_file.add(k)
            cur = brain_meta.get(k)
            if cur is None or fit > cur[1]:
                brain_meta[k] = (n_syn, fit)
        for k in unique_in_file:
            file_count_per_brain[k] += 1
            all_files_with_brain[k].append(fname)
        file_to_brains[fname] = unique_in_file

    total_files = sum(1 for _, ants in results if ants)
    n_unique_brains = len(file_count_per_brain)
    print(f'\nParsed {total_files} files. Distinct brains seen: {n_unique_brains:,}')
    total_instances = sum(instances_per_brain.values())
    print(f'Total BestAnts entries across all files: {total_instances:,}')
    print(f'Avg redundancy (entries / distinct brain): {total_instances / max(1,n_unique_brains):.1f}')

    # === Top "dominant" brains by file count ===
    print('\n=== Top 25 brains by number of FILES they appear in ===')
    print(f'{"rank":>4} {"#files":>7} {"%files":>7} {"#instances":>11} '
          f'{"synapses":>9} {"max_fitness":>12}')
    top = file_count_per_brain.most_common(25)
    for i, (k, n_files) in enumerate(top, 1):
        n_syn, max_fit = brain_meta.get(k, (0, 0))
        n_inst = instances_per_brain[k]
        print(f'{i:>4} {n_files:>7} {n_files/total_files:>6.1%}  {n_inst:>11} '
              f'{n_syn:>9} {int(max_fit):>12,}')

    # === Histogram of file-coverage for unique brains ===
    print('\n=== Distribution: how many files each unique brain appears in ===')
    coverage_hist = Counter()
    for k, n in file_count_per_brain.items():
        # bucket
        if n == 1: coverage_hist['1 file'] += 1
        elif n <= 5: coverage_hist['2-5 files'] += 1
        elif n <= 25: coverage_hist['6-25 files'] += 1
        elif n <= 100: coverage_hist['26-100 files'] += 1
        elif n <= 500: coverage_hist['101-500 files'] += 1
        else: coverage_hist['501+ files'] += 1
    for k in ['1 file', '2-5 files', '6-25 files', '26-100 files', '101-500 files', '501+ files']:
        n = coverage_hist.get(k, 0)
        if n:
            print(f'  {k:<15}  {n:>6}  ({n/n_unique_brains:.1%} of unique brains)')

    # === Synapse-count breakdown of dominant brains ===
    print('\n=== Synapse-count of brains appearing in 100+ files ===')
    syn_hist = Counter()
    for k, n_files in file_count_per_brain.items():
        if n_files >= 100:
            n_syn = brain_meta.get(k, (0, 0))[0]
            syn_hist[n_syn] += 1
    for n_syn in sorted(syn_hist.keys()):
        print(f'  {n_syn:>3} synapses : {syn_hist[n_syn]:>4} brains dominating 100+ files')

    # === Random selection simulation ===
    # If LoadBestAnts picks 25 files randomly, what fraction of TOP fitness brains do they see?
    print('\n=== If --load picks 25 random files (default max_files=25):')
    by_fitness = sorted(brain_meta.items(), key=lambda x: x[1][1], reverse=True)
    top10_brains = {k for k, _ in by_fitness[:10]}
    files_with_top10 = set()
    for k in top10_brains:
        files_with_top10.update(all_files_with_brain[k])
    p_any_top = len(files_with_top10) / total_files
    expected = 25 * p_any_top
    print(f'  Top-10 brains exist in {len(files_with_top10)}/{total_files} files ({p_any_top:.1%})')
    print(f'  Expected #files-with-a-top10-brain in a random pick of 25: {expected:.2f}')
    print(f'  Probability NONE are picked: {(1-p_any_top)**25:.1%}')

    # Show top 10 by fitness with their file coverage
    print('\n=== Top 10 brains by max fitness ===')
    print(f'{"rank":>4} {"max_fit":>10} {"synapses":>9} {"#files":>7} {"%files":>7}')
    for i, (k, (n_syn, fit)) in enumerate(by_fitness[:10], 1):
        nf = file_count_per_brain.get(k, 0)
        print(f'{i:>4} {int(fit):>10,} {n_syn:>9} {nf:>7} {nf/total_files:>6.1%}')


if __name__ == '__main__':
    main()
