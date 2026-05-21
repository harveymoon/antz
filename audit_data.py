import json
import os
from collections import defaultdict
from multiprocessing import Pool
import re
import glob

DATA_DIR = r"C:\Users\Harvey\Documents\Python\antz\dataSave\best"

def parse_filename(filename):
    """Extract run_id and timestamp from filename."""
    basename = os.path.basename(filename)
    match = re.match(r'^([a-f0-9]{8})?_?(\d{8})-(\d{6})\.json$', basename)
    if match:
        run_id, date_str, time_str = match.groups()
        return run_id, f"{date_str}-{time_str}"
    return None, None

def process_file(filepath):
    """Process single JSON file and extract metrics."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data.get('BestAnts'):
            return {'error': 'empty_ants', 'file': os.path.basename(filepath)}
        
        top_ant = data['BestAnts'][0]
        fitness = top_ant.get('fitness', 0)
        food = top_ant.get('food', 0)
        brain = top_ant.get('brain', [])
        
        # Validation checks
        issues = []
        if fitness == 0:
            issues.append('fitness_zero')
        if not brain:
            issues.append('empty_brain')
        
        # Check weight values (last element of each synapse tuple) are in [-1, 1]
        if brain:
            weights = []
            for synapse in brain:
                if isinstance(synapse, (list, tuple)) and len(synapse) >= 5:
                    # Weights are typically at indices 3 and 5
                    for idx in [3, 5]:
                        if idx < len(synapse):
                            w = synapse[idx]
                            if isinstance(w, (int, float)):
                                weights.append(w)
                                if w < -1 or w > 1:
                                    issues.append('weight_oob')
                                    break
        
        run_id, timestamp = parse_filename(filepath)
        date_only = timestamp[:8] if timestamp else None
        
        # Brain hash based on synapse structure
        brain_hash = hash(tuple(tuple(s) if isinstance(s, (list, tuple)) else s for s in brain))
        
        return {
            'file': os.path.basename(filepath),
            'fitness': fitness,
            'food': food,
            'brain_len': len(brain),
            'is_modern': run_id is not None,
            'timestamp': timestamp,
            'date': date_only,
            'ratio': fitness / food if food > 0 else 0,
            'brain_hash': brain_hash,
            'issues': issues,
        }
    except Exception as e:
        return {'error': str(e), 'file': os.path.basename(filepath)}

def main():
    pattern = os.path.join(DATA_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files")
    
    # Process in parallel
    with Pool(processes=8) as pool:
        results = pool.map(process_file, files)
    
    valid = [r for r in results if 'error' not in r]
    errors = [r for r in results if 'error' in r]
    
    print(f"\n=== ERRORS & RED FLAGS ===")
    print(f"Total errors/malformed: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"  {e['file']}: {e.get('error')}")
    
    if not valid:
        print("No valid files to analyze!")
        return
    
    # 1. TIME SERIES
    print(f"\n=== TIME SERIES (Best Fitness by Date) ===")
    by_date = defaultdict(list)
    for r in valid:
        if r['date']:
            by_date[r['date']].append(r['fitness'])
    
    date_max = {}
    for date in sorted(by_date.keys()):
        max_fit = max(by_date[date])
        date_max[date] = max_fit
        print(f"{date}: {max_fit:,.0f}")
    
    sorted_dates = sorted(date_max.keys())
    if len(sorted_dates) > 1:
        first = date_max[sorted_dates[0]]
        last = date_max[sorted_dates[-1]]
        trend = ((last - first) / first * 100) if first > 0 else 0
        print(f"Overall trend: {trend:+.1f}% (from {first:,.0f} to {last:,.0f})")
    
    # 2. RUN vs LEGACY
    print(f"\n=== RUN vs LEGACY ===")
    modern = [r for r in valid if r['is_modern']]
    legacy = [r for r in valid if not r['is_modern']]
    
    print(f"Modern files: {len(modern)}, Legacy: {len(legacy)}")
    if modern:
        modern_fit = [r['fitness'] for r in modern]
        print(f"  Modern - max: {max(modern_fit):,.0f}, median: {sorted(modern_fit)[len(modern_fit)//2]:,.0f}")
    if legacy:
        legacy_fit = [r['fitness'] for r in legacy]
        print(f"  Legacy - max: {max(legacy_fit):,.0f}, median: {sorted(legacy_fit)[len(legacy_fit)//2]:,.0f}")
    
    # 3. BRAIN COMPLEXITY vs FITNESS
    print(f"\n=== BRAIN COMPLEXITY vs FITNESS ===")
    brain_lens = [r['brain_len'] for r in valid]
    fitnesses = [r['fitness'] for r in valid]
    
    print(f"Brain length range: {min(brain_lens)} to {max(brain_lens)} synapses")
    print(f"Fitness range: {min(fitnesses):,.0f} to {max(fitnesses):,.0f}")
    
    # Correlation: top vs bottom 10% by brain size
    threshold_top = sorted(brain_lens)[int(0.9*len(brain_lens))]
    threshold_bot = sorted(brain_lens)[int(0.1*len(brain_lens))]
    big_brain_fit = [valid[i]['fitness'] for i, bl in enumerate(brain_lens) if bl >= threshold_top]
    small_brain_fit = [valid[i]['fitness'] for i, bl in enumerate(brain_lens) if bl <= threshold_bot]
    
    if big_brain_fit and small_brain_fit:
        print(f"Top 10% brain size (>={threshold_top}): avg fitness {sum(big_brain_fit)/len(big_brain_fit):,.0f}")
        print(f"Bottom 10% brain size (<={threshold_bot}): avg fitness {sum(small_brain_fit)/len(small_brain_fit):,.0f}")
    
    # 4. DIVERSITY CHECK
    print(f"\n=== DIVERSITY CHECK (Top 50 Ants) ===")
    top_50 = sorted(valid, key=lambda r: r['fitness'], reverse=True)[:50]
    brain_hashes = [r['brain_hash'] for r in top_50]
    unique_brains = len(set(brain_hashes))
    print(f"Top 50 unique brain structures: {unique_brains}/50")
    
    if unique_brains < 15:
        print("WARNING: Severe bottleneck! Top ants share only {}/{} distinct architectures.".format(unique_brains, 50))
    
    # 5. FITNESS vs FOOD RATIO
    print(f"\n=== FITNESS vs FOOD RATIO ===")
    ratios = [r['ratio'] for r in valid if r['ratio'] > 0]
    if ratios:
        print(f"Ratio range: {min(ratios):.1f} to {max(ratios):.1f}")
        print(f"Mean ratio: {sum(ratios)/len(ratios):.1f}")
        
        top_10 = sorted(valid, key=lambda r: r['fitness'], reverse=True)[:10]
        top_ratios = [r['ratio'] for r in top_10]
        print(f"Top 10 ants - ratios: {[f'{x:.1f}' for x in top_ratios]}")
        
        if sum(ratios)/len(ratios) > 15:
            print("NOTE: Fitness heavily weighted by bonuses, not just food collection.")
    
    # Issue summary
    issue_counts = defaultdict(int)
    for r in valid:
        for issue in r.get('issues', []):
            issue_counts[issue] += 1
    
    print(f"\n=== ISSUE SUMMARY ===")
    if issue_counts:
        for issue, count in sorted(issue_counts.items()):
            print(f"{issue}: {count}")
    else:
        print("No issues detected")
    
    print(f"\n=== OVERALL STATS ===")
    print(f"Total valid files: {len(valid)}")
    print(f"Max fitness: {max(fitnesses):,.0f}")
    print(f"Mean fitness: {sum(fitnesses)/len(fitnesses):,.0f}")
    print(f"Median fitness: {sorted(fitnesses)[len(fitnesses)//2]:,.0f}")

if __name__ == '__main__':
    main()
