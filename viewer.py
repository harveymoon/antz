"""Antz sim history viewer.

A dependency-free local web app for browsing current and past runs:
summary stats, learning-curve charts, reset/epoch markers, and a timeline
scrubber synced to the run's timelapse video (when one exists).

Usage:
    python viewer.py            # index new data, then serve on http://localhost:8008
    python viewer.py --index    # just (re)build caches and exit
    python viewer.py --port N   # serve on another port

Death logs are distilled ONCE into small cached summaries in
dataSave/viewer_cache/{runID}.json (a multi-GB log becomes ~100 KB of
binned series). After a run's cache exists, the raw death log is no longer
needed by the viewer and may be deleted to save disk. Caches update
incrementally if a log grows (live runs).
"""
import json
import os
import re
import sys
import glob
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'dataSave')
DEATHS = os.path.join(DATA, 'deaths')
CACHE = os.path.join(DATA, 'viewer_cache')
REPORTS = os.path.join(ROOT, 'reports')
CAPTURES = os.path.join(DATA, 'captures')
TIMELAPSES = os.path.join(DATA, 'timelapses')

BIN = 1000  # steps per series bin

# ---------------------------------------------------------------- fast line parse

def _num_after(line, key, cast=int):
    """Extract the number following '"key": ' without full JSON parsing."""
    i = line.find(key)
    if i < 0:
        return None
    i += len(key)
    j = i
    n = len(line)
    while j < n and line[j] not in ',}]':
        j += 1
    try:
        return cast(line[i:j].strip())
    except ValueError:
        return None

SRC_KEYS = ('"pickup":', '"deliver_base":', '"deliver_distance":',
            '"death_nav":', '"death_exploration":', '"trail_step":')

# Events are [step, "pickup"] in old logs and [step, "pickup", dist] in new ones
PICK_DIST = re.compile(r'"pickup", (\d+)\]')
DELIV_DIST = re.compile(r'"deliver", (\d+)\]')
NBIN = 10  # deaths, pickups, delivers, lifeSum, fitSum, pkDistSum, pkDistN, dvDistSum, dvDistN, dvDistMax


def parse_log_lines(fh, state):
    """Stream death-log lines into the aggregate state dict."""
    bins = state['bins']
    bytype = state['bytype']
    sources = state['sources']
    for raw in fh:
        line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
        if '"step"' not in line:
            continue
        step = _num_after(line, '"step": ')
        if step is None:
            continue
        lifespan = _num_after(line, '"lifespan": ') or 0
        fitness = _num_after(line, '"fitness_final": ', float) or 0.0
        food = _num_after(line, '"food_consumed": ') or 0
        pdists = [int(x) for x in PICK_DIST.findall(line)]
        ddists = [int(x) for x in DELIV_DIST.findall(line)]
        pk = line.count('"pickup"]') + len(pdists)
        dv = line.count('"deliver"]') + len(ddists)
        # ant type: '"antID": [123, "M", -1]'
        atype = '?'
        i = line.find('"antID": [')
        if i >= 0:
            j = line.find('"', i + 10)
            if j >= 0:
                k = line.find('"', j + 1)
                atype = line[j + 1:k][:2]
        b = step // BIN
        row = bins.get(b)
        if row is None:
            row = bins[b] = [0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0]
        elif len(row) < NBIN:
            row.extend([0] * (NBIN - len(row)))  # cache from before distance logging
        row[0] += 1
        row[1] += pk
        row[2] += dv
        row[3] += lifespan
        row[4] += fitness
        if pdists:
            row[5] += sum(pdists)
            row[6] += len(pdists)
        if ddists:
            row[7] += sum(ddists)
            row[8] += len(ddists)
            if max(ddists) > row[9]:
                row[9] = max(ddists)
        t = bytype.get(atype)
        if t is None:
            t = bytype[atype] = [0, 0]
        t[0] += 1
        if dv:
            t[1] += 1
        if '"fitness_breakdown": {}' not in line:
            for k in SRC_KEYS:
                v = _num_after(line, k + ' ', float)
                if v:
                    sources[k.strip('":')] = sources.get(k.strip('":'), 0.0) + v
        tot = state['totals']
        tot['deaths'] += 1
        tot['pickups'] += pk
        tot['delivers'] += dv
        if dv > 1:
            tot['multitrip_ants'] += 1
        if food > tot['maxFood']:
            tot['maxFood'] = food
        if fitness > tot['maxFitness']:
            tot['maxFitness'] = fitness
        if step > state['lastStep']:
            state['lastStep'] = step
        if state['firstStep'] < 0 or step < state['firstStep']:
            state['firstStep'] = step


# ---------------------------------------------------------------- report parsing

REPORT_PATTERNS = {
    'step': re.compile(r'Step: ([\d,]+)'),
    'fit': re.compile(r'Min: ([\d,]+)\s+\|\s+25%: ([\d,]+)\s+\|\s+Median: ([\d,]+)\s+\|\s+75%: ([\d,]+)\s+\|\s+Max: ([\d,]+)'),
    'div': re.compile(r'Diversity Index: ([\d.]+)'),
    'brain': re.compile(r'Synapse count: Min=(\d+), Avg=([\d.]+), Max=(\d+)'),
}


def parse_reports(run_id):
    out = []
    for path in sorted(glob.glob(os.path.join(REPORTS, f'{run_id}_report_*.txt'))):
        try:
            txt = open(path, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        rec = {}
        m = REPORT_PATTERNS['step'].search(txt)
        if not m:
            continue
        rec['step'] = int(m.group(1).replace(',', ''))
        m = REPORT_PATTERNS['fit'].search(txt)
        if m:
            rec['fitMin'], rec['fit25'], rec['fitMed'], rec['fit75'], rec['fitMax'] = (
                int(g.replace(',', '')) for g in m.groups())
        m = REPORT_PATTERNS['div'].search(txt)
        if m:
            rec['diversity'] = float(m.group(1))
        m = REPORT_PATTERNS['brain'].search(txt)
        if m:
            rec['brainAvg'] = float(m.group(2))
        out.append(rec)
    return out


# ---------------------------------------------------------------- cache build

def discover_runs():
    ids = set()
    for pat, trim in (
            (os.path.join(DEATHS, '*.jsonl'), '.jsonl'),
            (os.path.join(DATA, '*.manifest.json'), '.manifest.json')):
        for p in glob.glob(pat):
            name = os.path.basename(p)
            if name.endswith('.resets.jsonl'):
                name = name[:-len('.resets.jsonl')]
            else:
                name = name[:-len(trim)]
            ids.add(name)
    for p in glob.glob(os.path.join(CAPTURES, '*')):
        if os.path.isdir(p):
            ids.add(os.path.basename(p))
    for p in glob.glob(os.path.join(REPORTS, '*_report_*.txt')):
        ids.add(os.path.basename(p).split('_report_')[0])
    return sorted(ids)


def cache_path(run_id):
    return os.path.join(CACHE, f'{run_id}.json')


def load_cache(run_id):
    try:
        with open(cache_path(run_id), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fresh_state():
    return {'bins': {}, 'bytype': {}, 'sources': {},
            'totals': {'deaths': 0, 'pickups': 0, 'delivers': 0,
                       'multitrip_ants': 0, 'maxFood': 0, 'maxFitness': 0.0},
            'firstStep': -1, 'lastStep': 0, 'bytes_parsed': 0}


def build_cache(run_id, verbose=True):
    """Create or incrementally update one run's cache. Returns the cache dict."""
    os.makedirs(CACHE, exist_ok=True)
    cached = load_cache(run_id)
    state = fresh_state()
    if cached and 'state' in cached:
        st = cached['state']
        st['bins'] = {int(k): v for k, v in st['bins'].items()}
        state = st
    log = os.path.join(DEATHS, f'{run_id}.jsonl')
    log_size = os.path.getsize(log) if os.path.exists(log) else 0
    if log_size > state['bytes_parsed']:
        if verbose:
            mb = (log_size - state['bytes_parsed']) / 1e6
            print(f'[index] {run_id}: parsing {mb:,.0f} MB...')
        with open(log, 'rb') as f:
            f.seek(state['bytes_parsed'])
            if state['bytes_parsed']:
                f.readline()  # skip possible partial line
            parse_log_lines(f, state)
            state['bytes_parsed'] = f.tell()
    # resets
    resets = []
    rp = os.path.join(DEATHS, f'{run_id}.resets.jsonl')
    if os.path.exists(rp):
        for ln in open(rp, encoding='utf-8', errors='replace'):
            try:
                resets.append(json.loads(ln))
            except ValueError:
                pass
    # manifest
    manifest = None
    mp = os.path.join(DATA, f'{run_id}.manifest.json')
    if os.path.exists(mp):
        try:
            manifest = json.load(open(mp, encoding='utf-8'))
        except (OSError, ValueError):
            pass
    # media
    video = None
    for p in glob.glob(os.path.join(TIMELAPSES, f'timelapse_{run_id}_*.mp4')):
        video = os.path.basename(p)
    frames = len(glob.glob(os.path.join(CAPTURES, run_id, 'frame_*.png')))
    cache = {
        'runID': run_id,
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'state': state,
        'resets': resets,
        'manifest': manifest,
        'reports': parse_reports(run_id),
        'video': video,
        'frames': frames,
        'log_size': log_size,
        'log_mtime': os.path.getmtime(log) if os.path.exists(log) else None,
    }
    # Atomic write: never leave a half-written cache for a reader to trip on
    tmp = cache_path(run_id) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path(run_id))
    return cache


def run_summary(cache):
    st = cache['state']
    man = cache.get('manifest') or {}
    started = man.get('started')
    if not started and cache.get('log_mtime'):
        # Pre-manifest run: fall back to the death log's timestamp
        started = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache['log_mtime']))
    return {
        'runID': cache['runID'],
        'started': started,
        'argv': man.get('argv'),
        'commit': man.get('commit'),
        'steps': st['lastStep'],
        'deaths': st['totals']['deaths'],
        'delivers': st['totals']['delivers'],
        'maxFood': st['totals']['maxFood'],
        'maxFitness': st['totals']['maxFitness'],
        'resets': len(cache.get('resets') or []),
        'video': cache.get('video'),
        'frames': cache.get('frames', 0),
        'reports': len(cache.get('reports') or []),
        'log_size': cache.get('log_size', 0),
    }


def index_all(verbose=True):
    out = []
    for rid in discover_runs():
        try:
            out.append(build_cache(rid, verbose=verbose))
        except Exception as e:  # keep indexing the rest
            print(f'[index] {rid}: FAILED ({e})')
    return out


def refresher_loop():
    """Single background thread that keeps caches current. HTTP handlers only
    ever read caches - they never parse - so the UI stays instant no matter
    how much a live run's log has grown."""
    while True:
        try:
            for rid in discover_runs():
                c = load_cache(rid)
                log = os.path.join(DEATHS, f'{rid}.jsonl')
                grown = (os.path.exists(log)
                         and (c is None or os.path.getsize(log) > c['state']['bytes_parsed']))
                if c is None or grown:
                    build_cache(rid, verbose=False)
        except Exception as e:
            print(f'[refresh] error: {e}')
        time.sleep(10)


# ---------------------------------------------------------------- http server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json', extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_file_ranged(self, path, ctype):
        size = os.path.getsize(path)
        rng = self.headers.get('Range')
        start, end = 0, size - 1
        code = 200
        if rng and rng.startswith('bytes='):
            part = rng[6:].split('-')
            if part[0]:
                start = int(part[0])
            if len(part) > 1 and part[1]:
                end = min(int(part[1]), size - 1)
            code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(length))
        if code == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with open(path, 'rb') as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionAbortedError, BrokenPipeError):
                    return
                remaining -= len(chunk)

    def do_GET(self):
        path = self.path.split('?')[0]
        try:
            if path == '/':
                return self._send(200, PAGE, 'text/html; charset=utf-8')
            if path == '/api/runs':
                runs = []
                for rid in discover_runs():
                    c = load_cache(rid)
                    if c is None:
                        continue  # background refresher will index it shortly
                    s = run_summary(c)
                    # Hide data-less ghosts (old report-only run IDs)
                    if s['deaths'] or s['video'] or s['frames'] or s['reports'] >= 2:
                        runs.append(s)
                runs.sort(key=lambda r: r.get('started') or '', reverse=True)
                return self._send(200, runs)
            m = re.match(r'^/api/run/([A-Za-z0-9_-]+)$', path)
            if m:
                c = load_cache(m.group(1))
                if c is None:
                    return self._send(404, {'error': 'indexing - retry shortly'})
                return self._send(200, c)
            m = re.match(r'^/video/([A-Za-z0-9_-]+)$', path)
            if m:
                c = load_cache(m.group(1))
                if c and c.get('video'):
                    p = os.path.join(TIMELAPSES, c['video'])
                    if os.path.exists(p):
                        return self._send_file_ranged(p, 'video/mp4')
                return self._send(404, {'error': 'no video'})
            m = re.match(r'^/frame/([A-Za-z0-9_-]+)/(\d+)$', path)
            if m:
                p = os.path.join(CAPTURES, m.group(1), f'frame_{int(m.group(2)):06d}.png')
                if os.path.exists(p):
                    return self._send(200, open(p, 'rb').read(), 'image/png')
                return self._send(404, {'error': 'no frame'})
            return self._send(404, {'error': 'not found'})
        except (ConnectionAbortedError, BrokenPipeError):
            pass
        except Exception as e:
            try:
                self._send(500, {'error': str(e)})
            except Exception:
                pass


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Antz History Viewer</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font:13px/1.5 system-ui, sans-serif; background:#15171c; color:#cfd3dc; display:flex; height:100vh; }
  #side { width:270px; min-width:270px; overflow-y:auto; background:#1b1e25; border-right:1px solid #2a2e38; }
  #side h1 { font-size:14px; padding:12px 14px 6px; margin:0; color:#8eff9a; }
  .run { padding:9px 14px; border-bottom:1px solid #22262f; cursor:pointer; }
  .run:hover { background:#232834; }
  .run.sel { background:#28304a; }
  .run .id { font-weight:600; color:#e8ebf2; font-family:monospace; }
  .run .meta { color:#8a90a0; font-size:11px; }
  .badge { display:inline-block; padding:0 6px; border-radius:8px; font-size:10px; margin-left:4px; }
  .b-video { background:#3a2e55; color:#c9b3ff; }
  .b-live { background:#214a2a; color:#8eff9a; }
  #main { flex:1; overflow-y:auto; padding:16px 22px; }
  .cards { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }
  .card { background:#1b1e25; border:1px solid #2a2e38; border-radius:8px; padding:8px 14px; min-width:110px; }
  .card .v { font-size:19px; font-weight:700; color:#e8ebf2; }
  .card .k { font-size:10px; color:#8a90a0; text-transform:uppercase; letter-spacing:.5px; }
  canvas.chart { width:100%; height:110px; background:#1b1e25; border:1px solid #2a2e38; border-radius:8px; margin-bottom:10px; display:block; }
  #scrub { width:100%; }
  video { width:100%; max-height:46vh; background:#000; border-radius:8px; }
  h2 { font-size:13px; color:#8a90a0; margin:14px 0 6px; }
  .label { font-size:11px; color:#8a90a0; margin:2px 0; }
  #manifest { font-family:monospace; font-size:11px; color:#8a90a0; white-space:pre-wrap; }
</style></head><body>
<div id="side"><h1>&#128028; Antz Runs</h1><div id="runs"></div></div>
<div id="main"><div class="label">Select a run</div></div>
<script>
let runs = [], sel = null, cache = null, prevSizes = {};
const $ = s => document.querySelector(s);

async function loadRuns() {
  runs = await (await fetch('/api/runs')).json();
  const box = $('#runs'); box.innerHTML = '';
  for (const r of runs) {
    const live = prevSizes[r.runID] !== undefined && r.log_size > prevSizes[r.runID];
    prevSizes[r.runID] = r.log_size;
    const d = document.createElement('div');
    d.className = 'run' + (sel === r.runID ? ' sel' : '');
    d.innerHTML = `<div class="id">${r.runID}` +
      (r.video ? '<span class="badge b-video">video</span>' : '') +
      (live ? '<span class="badge b-live">LIVE</span>' : '') + `</div>` +
      `<div class="meta">${r.started || 'date unknown'}</div>` +
      `<div class="meta">${(r.steps||0).toLocaleString()} steps - ${(r.deaths||0).toLocaleString()} deaths - ${(r.delivers||0).toLocaleString()} delivers</div>`;
    d.onclick = () => select(r.runID);
    box.appendChild(d);
  }
}

async function select(id) {
  sel = id;
  cache = await (await fetch('/api/run/' + id)).json();
  render();
  loadRuns();
}

function series() {
  const bins = cache.state.bins, keys = Object.keys(bins).map(Number).sort((a,b)=>a-b);
  const s = { step:[], deaths:[], pickups:[], delivers:[], life:[], fit:[],
              dStep:[], dAvg:[], dMax:[] };
  for (const k of keys) {
    const [d,p,v,ls,fs,,,dds=0,ddn=0,ddm=0] = bins[k];
    s.step.push(k*1000); s.deaths.push(d); s.pickups.push(p); s.delivers.push(v);
    s.life.push(d ? ls/d : 0); s.fit.push(d ? fs/d : 0);
    if (ddn) { s.dStep.push(k*1000); s.dAvg.push(dds/ddn); s.dMax.push(ddm); }
  }
  return s;
}

function drawChart(cv, xs, ys, color, title, resets, cursor) {
  const ctx = cv.getContext('2d'), W = cv.width = cv.clientWidth*2, H = cv.height = 220;
  ctx.clearRect(0,0,W,H);
  if (!xs.length) return;
  const x0 = xs[0], x1 = xs[xs.length-1] || 1, ymax = Math.max(...ys, 1);
  const X = v => (v-x0)/(x1-x0||1)*(W-20)+10, Y = v => H-14-(v/ymax)*(H-40);
  ctx.strokeStyle = '#2a2e38'; ctx.strokeRect(0.5,0.5,W-1,H-1);
  for (const r of resets||[]) { const x = X(r.step);
    ctx.strokeStyle = r.move_nest ? '#ff5d5d' : '#b08a3e'; ctx.beginPath(); ctx.moveTo(x,16); ctx.lineTo(x,H-14); ctx.stroke(); }
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  xs.forEach((x,i) => i ? ctx.lineTo(X(x),Y(ys[i])) : ctx.moveTo(X(x),Y(ys[i])));
  ctx.stroke(); ctx.lineWidth = 1;
  if (cursor != null) { const x = X(cursor);
    ctx.strokeStyle = '#8eff9a'; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  ctx.fillStyle = '#8a90a0'; ctx.font = '20px system-ui';
  ctx.fillText(title + '  (max ' + Math.round(ymax).toLocaleString() + ')', 14, 26);
}

let charts = [];
function render(cursorStep) {
  const st = cache.state, t = st.totals, man = cache.manifest || {};
  const s = series();
  const main = $('#main');
  if (!main.dataset.run || main.dataset.run !== sel) {
    main.dataset.run = sel;
    main.innerHTML = `
      <div class="cards">
        <div class="card"><div class="v">${(st.lastStep||0).toLocaleString()}</div><div class="k">steps</div></div>
        <div class="card"><div class="v">${t.deaths.toLocaleString()}</div><div class="k">deaths</div></div>
        <div class="card"><div class="v">${t.delivers.toLocaleString()}</div><div class="k">deliveries</div></div>
        <div class="card"><div class="v">${t.maxFood}</div><div class="k">top food</div></div>
        <div class="card"><div class="v">${Math.round(t.maxFitness).toLocaleString()}</div><div class="k">top fitness</div></div>
        <div class="card"><div class="v">${(cache.resets||[]).length}</div><div class="k">world resets</div></div>
      </div>
      ${cache.video ? `<video id="vid" src="/video/${sel}" controls muted></video>` : (cache.frames ? `<img id="frameimg" style="width:100%;border-radius:8px">` : '')}
      <input type="range" id="scrub" min="0" max="1000" value="0">
      <div class="label" id="scrublabel"></div>
      <canvas class="chart" id="c1"></canvas>
      <canvas class="chart" id="c2"></canvas>
      <canvas class="chart" id="c3"></canvas>
      <canvas class="chart" id="c4"></canvas>
      <canvas class="chart" id="c5" style="display:none"></canvas>
      <canvas class="chart" id="c6" style="display:none"></canvas>
      <h2>Run manifest</h2><div id="manifest">${Object.keys(man).length ? JSON.stringify(man, null, 1) : 'none (pre-manifest run)'}</div>`;
    const vid0 = $('#vid');
    if (vid0) vid0.onloadedmetadata = () => {
      if (vid0.dataset.pending) { vid0.currentTime = vid0.dataset.pending * vid0.duration; delete vid0.dataset.pending; }
    };
    $('#scrub').oninput = e => {
      const frac = e.target.value/1000, step = Math.round(st.lastStep*frac);
      $('#scrublabel').textContent = 'step ' + step.toLocaleString();
      const vid = $('#vid');
      if (vid) { if (vid.duration) vid.currentTime = frac*vid.duration; else vid.dataset.pending = frac; }
      const img = $('#frameimg');
      if (img && cache.frames) img.src = '/frame/' + sel + '/' + Math.max(1, Math.round(frac*cache.frames));
      drawAll(step);
    };
  }
  drawAll(cursorStep);
  function drawAll(cur) {
    drawChart($('#c1'), s.step, s.delivers, '#8eff9a', 'Deliveries / 1k steps', cache.resets, cur);
    drawChart($('#c2'), s.step, s.pickups, '#ffd36e', 'Pickups / 1k steps', cache.resets, cur);
    drawChart($('#c3'), s.step, s.life, '#6ec4ff', 'Avg lifespan at death', cache.resets, cur);
    const reps = cache.reports || [];
    if (reps.length > 1)
      drawChart($('#c4'), reps.map(r=>r.step), reps.map(r=>r.fitMax||0), '#c9b3ff', 'Board max fitness (reports)', cache.resets, cur);
    else
      drawChart($('#c4'), s.step, s.deaths, '#ff9d6e', 'Deaths / 1k steps', cache.resets, cur);
    if (s.dStep.length > 1) {
      $('#c5').style.display = 'block'; $('#c6').style.display = 'block';
      drawChart($('#c5'), s.dStep, s.dAvg, '#ff7ad9', 'Avg delivered-food distance from nest (tiles)', cache.resets, cur);
      drawChart($('#c6'), s.dStep, s.dMax, '#ffb27a', 'Farthest delivery per 1k steps (tiles)', cache.resets, cur);
    }
  }
}

setInterval(async () => {
  await loadRuns();
  if (sel) { cache = await (await fetch('/api/run/' + sel)).json(); render(); }
}, 8000);
loadRuns();
</script></body></html>
"""


def main():
    port = 8008
    if '--port' in sys.argv:
        port = int(sys.argv[sys.argv.index('--port') + 1])
    if '--index' in sys.argv:
        print('[index] updating caches...')
        index_all()
        print('[index] done.')
        return
    # Serve immediately from existing caches; one background thread keeps
    # them current (initial catch-up included) so requests never parse.
    threading.Thread(target=refresher_loop, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Antz history viewer: http://localhost:{port} (background indexer running)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
