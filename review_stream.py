"""Streaming death-log review: ant types, reward decomposition, learning trend.
Memory-bounded: never holds more than top-N records."""
import json, sys, heapq

path = sys.argv[1]
SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # process every Nth line

types = {}          # antID[1] -> count
type_deliver = {}   # antID[1] -> deliveries
n = 0
pickups = 0
delivers = 0
multi = 0
src_total = {}
top = []  # heap of (fitness, breakdown, brain_size, food, antID1)
buckets = []  # per 1M steps: [n, delivered, fit_sum]
last_step = 0
lifespan_sum = 0
carry_death = 0
brain_sz_sum = 0

with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        if i % SAMPLE:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        n += 1
        t = d.get('antID', [0, '?'])[1]
        types[t] = types.get(t, 0) + 1
        ev = d.get('events', [])
        npick = sum(1 for e in ev if e[1] == 'pickup')
        ndel = sum(1 for e in ev if e[1] == 'deliver')
        if npick: pickups += 1
        if ndel:
            delivers += 1
            type_deliver[t] = type_deliver.get(t, 0) + 1
        if ndel > 1: multi += 1
        fb = d.get('fitness_breakdown', {})
        for k, v in fb.items():
            src_total[k] = src_total.get(k, 0) + v
        fit = d.get('fitness_final', 0)
        step = d.get('step', 0)
        last_step = max(last_step, step)
        b = step // 1_000_000
        while len(buckets) <= b:
            buckets.append([0, 0, 0.0])
        buckets[b][0] += 1
        buckets[b][1] += 1 if ndel else 0
        buckets[b][2] += fit
        lifespan_sum += d.get('lifespan', 0)
        carry_death += 1 if d.get('carrying_at_death') else 0
        brain_sz_sum += d.get('brain_size', 0)
        item = (fit, json.dumps(fb), d.get('brain_size', 0), d.get('food_consumed', 0), str(t), ndel)
        if len(top) < 50:
            heapq.heappush(top, item)
        elif fit > top[0][0]:
            heapq.heapreplace(top, item)

print(f'=== {path} (sampled 1/{SAMPLE}) ===')
print(f'deaths={n}  last_step={last_step:,}')
print(f'pickup%={100*pickups/max(1,n):.2f}  deliver%={100*delivers/max(1,n):.2f}  multi-trip%={100*multi/max(1,n):.3f}')
print(f'avg lifespan={lifespan_sum/max(1,n):.0f}  carrying_at_death%={100*carry_death/max(1,n):.2f}  avg brain={brain_sz_sum/max(1,n):.1f}')
print('\n-- ant types (count, deliver% within type) --')
for t, c in sorted(types.items(), key=lambda x: -x[1]):
    dl = type_deliver.get(t, 0)
    print(f'  {t:>3}: {c:>9}  ({100*c/n:5.1f}%)  delivered: {100*dl/max(1,c):5.2f}%')
print('\n-- aggregate reward sources --')
tot = sum(src_total.values()) or 1
for k, v in sorted(src_total.items(), key=lambda x: -x[1]):
    print(f'  {k:>18}: {v:>14,.0f}  ({100*v/tot:5.1f}%)')
print('\n-- top-50 ants reward sources --')
top_src = {}
for fit, fbj, bs, food, t, ndel in top:
    for k, v in json.loads(fbj).items():
        top_src[k] = top_src.get(k, 0) + v
tt = sum(top_src.values()) or 1
for k, v in sorted(top_src.items(), key=lambda x: -x[1]):
    print(f'  {k:>18}: {v:>14,.0f}  ({100*v/tt:5.1f}%)')
tops = sorted(top, reverse=True)[:10]
print('\n-- top 10 ants --')
for fit, fbj, bs, food, t, ndel in tops:
    print(f'  fit={fit:>9,.0f} type={t:>2} brain={bs:>3} food={food:>4} delivers={ndel:>4}  {fbj[:120]}')
print('\n-- per-1M-step buckets: deaths, deliver%, mean fitness --')
for bi, (bn, bd, bf) in enumerate(buckets):
    if bn == 0: continue
    print(f'  {bi:>3}M: n={bn:>9}  deliver%={100*bd/bn:6.2f}  meanFit={bf/bn:10.1f}')
