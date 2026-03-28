/* ── Eidolon Dashboard — Application Logic ───────────────────────────────── */

/* ── State ─────────────────────────────────────────────────────────────── */
let ws = null;
let pipelineRunning = false;
let ghostNodeCount = 0;
let nodeStartTimes = {};
let pipelineStartTime = null;
let mttdInterval = null;
let tokenMappings = []; // [{real, ghost}] for code mirror

const DEFAULT_EVENT = "401 Unauthorized on POST /verify-token";
const DEFAULT_SERVICE = "demo-svc";
const NODES = ["triage", "solution_replay", "context_fetch", "hypothesis", "validation", "hitl", "patch_synthesis"];

/* ── WebSocket ─────────────────────────────────────────────────────────── */
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    setWsState(true);
    setStatus("Connected — Eidolon ready");
    loadStatus();
  };

  ws.onmessage = (e) => {
    try { handleEvent(JSON.parse(e.data)); }
    catch (err) { console.warn("Bad event:", e.data); }
  };

  ws.onclose = () => {
    setWsState(false);
    setStatus("Disconnected — retrying in 3s");
    setTimeout(connectWS, 3000);
  };

  ws.onerror = () => ws.close();
}

function setWsState(connected) {
  document.getElementById('ws-dot').className = connected ? 'connected' : '';
  document.getElementById('ws-label').textContent = connected ? 'live' : 'disconnected';
}

/* ── Event dispatcher ──────────────────────────────────────────────────── */
function handleEvent(ev) {
  switch (ev.type) {
    case 'pipeline_start':      onPipelineStart(ev); break;
    case 'node_start':          onNodeStart(ev); break;
    case 'node_complete':       onNodeComplete(ev); break;
    case 'ghost_node':          onGhostNode(ev); break;
    case 'ghost_stack_trace':   onGhostStackTrace(ev); break;
    case 'hypotheses':          onHypotheses(ev); break;
    case 'stream_token':        onStreamToken(ev); break;
    case 'hitl_ready':          onHitlReady(ev); break;
    case 'hitl_decision':       onHitlDecision(ev); break;
    case 'patch_ready':         onPatchReady(ev); break;
    case 'patch_reconstructed': onPatchReconstructed(ev); break;
    case 'pipeline_complete':   onPipelineComplete(ev); break;
    case 'duplicate':           onDuplicate(ev); break;
    case 'solution_replay':     onSolutionReplay(ev); break;
    case 'memory_wiped':        onMemoryWiped(ev); break;
    case 'error':               onError(ev); break;
  }
}

/* ── Pipeline events ───────────────────────────────────────────────────── */
function onPipelineStart(ev) {
  pipelineRunning = true;
  pipelineStartTime = Date.now();
  ghostNodeCount = 0;
  tokenMappings = [];
  resetUI();

  const evLabel = typeof ev.event === 'object' && ev.event !== null
    ? (ev.event.error_message || JSON.stringify(ev.event))
    : (ev.event || DEFAULT_EVENT);
  document.getElementById('event-label').textContent = evLabel;
  document.getElementById('service-gem').classList.add('active');
  document.getElementById('service-label').textContent = (ev.service || DEFAULT_SERVICE).toUpperCase();
  document.getElementById('pipeline-phase').textContent = 'RUNNING';
  document.getElementById('run-btn').disabled = true;

  clearInterval(mttdInterval);
  mttdInterval = setInterval(() => {
    const s = ((Date.now() - pipelineStartTime) / 1000).toFixed(1);
    document.getElementById('mttd-value').textContent = s + 's';
    const pct = Math.min(100, (parseFloat(s) / 120) * 100);
    document.getElementById('mttd-fill').style.width = pct + '%';
  }, 200);

  const evStr = typeof ev.event === 'object' && ev.event !== null
    ? (ev.event.error_message || JSON.stringify(ev.event))
    : String(ev.event || '');
  setStatus("Analysis started — " + evStr.slice(0, 60));
}

function onNodeStart(ev) {
  nodeStartTimes[ev.node] = Date.now();
  setNodeState(ev.node, 'active');
  setNodeStatus(ev.node, '<span class="scanning">Analysing...</span>');

  const idx = NODES.indexOf(ev.node);
  if (idx > 0) {
    const conn = document.getElementById('conn-' + idx);
    if (conn) conn.classList.add('flowing');
  }

  setStatus(ev.label || ev.node);
}

function onNodeComplete(ev) {
  const node = ev.node;
  const elapsed = nodeStartTimes[node]
    ? ((Date.now() - nodeStartTimes[node]) / 1000).toFixed(1) + 's'
    : '';

  setNodeState(node, 'done');
  const el = document.getElementById('time-' + node);
  if (el) el.textContent = elapsed;

  const idx = NODES.indexOf(node);
  if (idx >= 0) {
    const conn = document.getElementById('conn-' + idx);
    if (conn) { conn.classList.remove('flowing'); conn.classList.add('done'); }
  }

  if (node === 'triage') {
    const typeTag = `<span class="tag tag-${(ev.error_type||'unknown').toLowerCase()}">${ev.error_type||'?'}</span>`;
    const sevTag  = `<span class="tag tag-${(ev.severity||'medium').toLowerCase()}">${ev.severity||'?'}</span>`;
    setNodeStatus(node, typeTag + ' ' + sevTag);
    if (ev.anchor_source === 'stack_trace_exact') {
      addBrainNote('Exact node located via stack trace line number');
    }
  } else if (node === 'context_fetch') {
    const anchor = ev.anchor_source === 'stack_trace_exact' ? ' — stack trace anchor' : '';
    setNodeStatus(node, `${ev.similar_count || 0} nodes · ${ev.graph_edges || 0} edges${anchor}`);
  } else if (node === 'hypothesis') {
    setNodeStatus(node, `${ev.count || 0} hypotheses generated`);
  } else if (node === 'validation') {
    const pct = ev.composite_confidence ? (ev.composite_confidence * 100).toFixed(0) + '%' : '0%';
    setNodeStatus(node, `Best: ${pct} · ${ev.gate_label || ''}`);
  } else if (node === 'patch_synthesis') {
    setNodeStatus(node, 'Patch synthesised');
  } else if (node === 'solution_replay') {
    setNodeState(node, 'replay');
    setNodeStatus(node, 'Known fix retrieved from library');
  } else {
    setNodeStatus(node, 'Complete');
  }
}

function onGhostNode(ev) {
  ghostNodeCount++;
  document.getElementById('ghost-count').textContent = ghostNodeCount + ' nodes';

  const data = ev.data || {};
  const id   = data.node_id || data.id || '?';
  const type = data.node_type || data.type || 'function';
  const disc = data.structural_discriminators || {};
  const lineStart = data.line_start;
  const lineEnd   = data.line_end;

  // Add to ghost node list
  const card = document.createElement('div');
  card.className = 'ghost-node-card';

  let metaHtml = `<span class="meta-tag meta-tag-func">${type}</span>`;
  if (disc.kind === 'http_endpoint') {
    metaHtml += ` <span class="meta-tag meta-tag-route">${disc.http_method} ${disc.route}</span>`;
  } else if (disc.kind === 'celery_task') {
    metaHtml += ` <span class="meta-tag meta-tag-route">task</span>`;
  } else if (disc.kind === 'test_function') {
    metaHtml += ` <span class="meta-tag meta-tag-route">test</span>`;
  }
  if (lineStart) {
    metaHtml += ` <span class="meta-tag meta-tag-line">L${lineStart}${lineEnd ? '-'+lineEnd : ''}</span>`;
  }

  card.innerHTML = `
    <div class="ghost-node-id">${id}</div>
    <div class="ghost-node-meta">${metaHtml}</div>
  `;

  const body = document.getElementById('ghost-body');
  const empty = body.querySelector('.empty-state');
  if (empty) body.removeChild(empty);
  body.appendChild(card);
  body.scrollTop = body.scrollHeight;
}

/* ── Ghost Stack Trace → Ghost Mirror ─────────────────────────────────── */
function onGhostStackTrace(ev) {
  const gst     = ev.data || {};
  const excType = gst.exception_type || gst.exception_class || 'Error';
  const frames  = gst.frames || [];
  const hints   = gst.structural_hints || {};

  const mirrorBody = document.getElementById('mirror-body');
  if (!mirrorBody) return;

  // Clear placeholder text on first event
  const placeholder = mirrorBody.querySelector('div');
  if (placeholder && placeholder.textContent.includes('Token mappings')) {
    mirrorBody.innerHTML = '';
  }

  // Exception type row — preserved (not hashed)
  const excRow = document.createElement('div');
  excRow.className = 'token-row';
  excRow.innerHTML = `
    <span class="token-real" style="border-color:var(--crimson);background:var(--crimson-dim);">Exception: ${escHtml(excType)}</span>
    <span class="token-arrow">&#8594;</span>
    <span class="token-ghost" style="color:var(--crimson);">${escHtml(excType)} <em style="font-size:9px;opacity:.6;">(preserved)</em></span>
  `;
  mirrorBody.appendChild(excRow);

  // Ghost message if present
  if (gst.ghost_message) {
    const msgRow = document.createElement('div');
    msgRow.className = 'token-row';
    msgRow.style.marginBottom = '8px';
    msgRow.innerHTML = `
      <span class="token-real" style="font-size:10px;">message</span>
      <span class="token-arrow">&#8594;</span>
      <span class="token-ghost" style="font-size:10px;">${escHtml(gst.ghost_message.slice(0,60))}</span>
    `;
    mirrorBody.appendChild(msgRow);
  }

  // Frame rows — line number preserved, function name hashed
  frames.slice(-6).forEach(f => {
    const lineRow = document.createElement('div');
    lineRow.className = 'token-row';
    lineRow.innerHTML = `
      <span class="token-real">L${f.line_number} · ${escHtml(f.file_basename || '?')}</span>
      <span class="token-arrow">&#8594;</span>
      <span class="token-ghost">${escHtml(f.ghost_function || '?')}</span>
    `;
    mirrorBody.appendChild(lineRow);

    if (f.ghost_code) {
      const codeRow = document.createElement('div');
      codeRow.className = 'token-row';
      codeRow.style.cssText = 'margin-left:14px;opacity:0.8;';
      codeRow.innerHTML = `
        <span class="token-real" style="font-size:10px;color:var(--ink-dim);">code</span>
        <span class="token-arrow">&#8594;</span>
        <span class="token-ghost" style="font-size:10px;">${escHtml((f.ghost_code||'').slice(0,44))}</span>
      `;
      mirrorBody.appendChild(codeRow);
    }
  });

  // Structural hints banner
  if (hints.retrieval_strategy || hints.error_category) {
    const hint = document.createElement('div');
    hint.style.cssText = 'margin-top:8px;padding:4px 8px;background:var(--sage-dim);border-left:2px solid var(--sage);font-size:10px;color:var(--sage);font-family:var(--font-mono);border-radius:0 2px 2px 0;';
    hint.textContent = `strategy: ${hints.retrieval_strategy || hints.error_category}`;
    mirrorBody.appendChild(hint);
  }
}

/* ── Code Mirror Visualization ─────────────────────────────────────────── */
function addTokenMapping(ghostId, disc, lineStart, lineEnd, type) {
  tokenMappings.push({ ghostId, disc, lineStart, lineEnd, type });
  renderMirror();
}

function renderMirror() {
  const container = document.getElementById('mirror-body');
  if (!container) return;

  // Show at most 8 entries to keep it clean
  const items = tokenMappings.slice(-8);
  container.innerHTML = '';

  items.forEach(m => {
    const row = document.createElement('div');
    row.className = 'token-row';

    // Figure out what the "real" representation would look like
    let realLabel = m.type;
    if (m.disc.kind === 'http_endpoint') {
      realLabel = `${m.disc.http_method} ${m.disc.route}`;
    } else if (m.disc.kind === 'celery_task') {
      realLabel = `@task (${m.type})`;
    } else if (m.disc.kind === 'test_function') {
      realLabel = `test_*()`;
    } else if (m.disc.qualified_name) {
      realLabel = m.disc.qualified_name.slice(0, 20) + '...';
    }
    if (m.lineStart) realLabel += ` :${m.lineStart}`;

    row.innerHTML = `
      <span class="token-real">${escHtml(realLabel)}</span>
      <span class="token-arrow">&#8594;</span>
      <span class="token-ghost">${escHtml(m.ghostId)}</span>
    `;
    container.appendChild(row);
  });
}

/* Show ghost stack trace frames in the mirror panel */
function showStackTrace(gst) {
  const panel = document.getElementById('mirror-body');
  if (!panel) return;

  const card = document.createElement('div');
  card.className = 'stack-card';

  const excType = gst.exception_type || 'Error';
  const frames  = gst.frames || [];

  let framesHtml = frames.slice(-4).map((f, i) => `
    <div class="stack-frame">
      <span class="frame-line">L${f.line_number}</span>
      <span class="frame-fn">${escHtml(f.ghost_function || '?')}</span>
      <span class="frame-code">${escHtml((f.ghost_code||'').slice(0, 40))}</span>
    </div>
  `).join('');

  card.innerHTML = `
    <div class="stack-exception">${excType}</div>
    <div style="font-size:11px;color:var(--ink-dim);margin-bottom:6px;font-style:italic;font-family:var(--font-body)">
      ${escHtml((gst.ghost_message||'').slice(0, 80))}
    </div>
    ${framesHtml}
  `;

  // Insert before the node list
  const body = document.getElementById('ghost-body');
  const empty = body.querySelector('.empty-state');
  if (empty) body.removeChild(empty);

  const existing = body.querySelector('.stack-card');
  if (existing) body.removeChild(existing);

  body.insertBefore(card, body.firstChild);
}

/* ── Hypothesis events ────────────────────────────────────────────────── */
function onHypotheses(ev) {
  const hypos = ev.data || [];
  const brainBody = document.getElementById('brain-body');
  const empty = brainBody.querySelector('.empty-state');
  if (empty) brainBody.removeChild(empty);

  hypos.forEach((h, i) => {
    const pct   = Math.round((h.confidence || 0) * 100);
    const level = pct >= 70 ? 'high' : pct >= 40 ? 'medium' : 'low';
    const isBest = i === 0;
    const card = document.createElement('div');
    card.className = 'hypothesis-card' + (isBest ? ' best' : '');
    card.innerHTML = `
      <div class="hypo-rank">
        Hypothesis ${h.rank || i + 1}
        ${isBest ? '<span class="best-badge">Best</span>' : ''}
      </div>
      <div class="hypo-text">${escHtml(h.hypothesis || '—')}</div>
      <div class="hypo-confidence">
        <div class="conf-track"><div class="conf-fill ${level}" style="width:${pct}%"></div></div>
        <div class="conf-pct">${pct}%</div>
      </div>
    `;
    brainBody.appendChild(card);
  });
}

function onStreamToken(ev) {
  setNodeStatus('hypothesis', `Reasoning... ${ev.token_count || 0} tokens`);
}

/* ── Solution Replay ──────────────────────────────────────────────────── */
function onSolutionReplay(ev) {
  const brainBody = document.getElementById('brain-body');
  const empty = brainBody.querySelector('.empty-state');
  if (empty) brainBody.removeChild(empty);

  const card = document.createElement('div');
  card.className = 'replay-card';
  card.innerHTML = `
    <div class="replay-title">Solution Library — Known Pattern</div>
    <div style="font-size:13px;color:var(--ink-mid);margin-bottom:6px;">
      Pattern: <strong>${escHtml(ev.fingerprint_key || '')}</strong>
    </div>
    <div style="font-size:13px;color:var(--bronze);">
      Fix: ${escHtml(ev.patch_type || '?')}
    </div>
    <div style="font-size:12px;color:var(--ink-dim);margin-top:4px;font-style:italic;">
      Confirmed ${ev.times_replayed || 0} time(s) · Confidence: ${Math.round((ev.confidence || 0) * 100)}%
    </div>
  `;
  brainBody.insertBefore(card, brainBody.firstChild);

  // Show banner in mirror panel too
  const banner = document.createElement('div');
  banner.className = 'solution-hit-banner';
  banner.textContent = 'Known error — solution retrieved from library';
  document.getElementById('ghost-body').insertBefore(banner, document.getElementById('ghost-body').firstChild);
}

/* ── HITL ─────────────────────────────────────────────────────────────── */
function onHitlReady(ev) {
  clearInterval(mttdInterval);
  setNodeState('hitl', 'active');
  setNodeStatus('hitl', '<span class="scanning">Awaiting your decision...</span>');
  document.getElementById('pipeline-phase').textContent = 'AWAITING REVIEW';

  const h    = ev.hypothesis || {};
  const conf = ev.confidence || 0;
  const pct  = Math.round(conf * 100);

  document.getElementById('hitl-hypothesis').textContent   = h.hypothesis || 'No hypothesis available';
  document.getElementById('hitl-root-cause').textContent   = h.root_cause || '—';
  document.getElementById('hitl-patch-sketch').textContent = h.patch_sketch || '—';
  document.getElementById('hitl-conf-fill').style.width    = pct + '%';
  document.getElementById('hitl-conf-value').textContent   = pct + '%';

  const nodes = h.affected_nodes || [];
  document.getElementById('hitl-nodes').innerHTML = nodes.length
    ? 'Affected nodes: ' + nodes.map(n => `<span class="node-id">${escHtml(n)}</span>`).join(' ')
    : '';

  document.getElementById('hitl-overlay').classList.add('active');
  setStatus("Human review required — approve or reject the proposed patch");
}

function onHitlDecision(ev) {
  document.getElementById('hitl-overlay').classList.remove('active');
  setNodeState('hitl', 'done');
  setNodeStatus('hitl', ev.approved ? 'Approved' : 'Rejected');
  document.getElementById('time-hitl').textContent = '';
  setStatus(ev.approved ? "Approved — synthesising patch..." : "Rejected — pipeline ended");
  document.getElementById('pipeline-phase').textContent = ev.approved ? 'PATCHING' : 'REJECTED';
}

function onPatchReady(ev) {
  const patch = ev.patch || {};
  const card  = document.createElement('div');
  card.className = 'patch-card';
  card.innerHTML = `
    <div style="font-family:var(--font-title);font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:var(--ink-dim);margin-bottom:8px;">Patch Generated</div>
    <div class="patch-type">${escHtml(patch.change_type || '—')}</div>
    <div class="patch-target">${escHtml(patch.target_node || '—')}</div>
    <div class="patch-desc">${escHtml(patch.description || ev.rationale || '—')}</div>
  `;
  document.getElementById('brain-body').appendChild(card);
}

function onPatchReconstructed(ev) {
  setStatus("Patch reconstructed — analysis complete");
  if (ev.diff) {
    addBrainNote('Reconstructed diff available');
  }
}

function onPipelineComplete(ev) {
  clearInterval(mttdInterval);
  pipelineRunning = false;
  document.getElementById('run-btn').disabled = false;
  document.getElementById('service-gem').classList.remove('active');
  document.getElementById('pipeline-phase').textContent = 'COMPLETE';

  const s = ev.mttd_seconds || 0;
  document.getElementById('mttd-value').textContent = s + 's';
  document.getElementById('mttd-fill').style.width = Math.min(100, (s / 120) * 100) + '%';

  const col = s < 60 ? 'var(--sage)' : s < 120 ? 'var(--bronze)' : 'var(--crimson)';
  document.getElementById('mttd-value').style.color = col;

  setStatus(`Complete — MTTD: ${s}s`);
}

function onDuplicate(ev) {
  setNodeStatus('triage', `Duplicate pattern (${(ev.score * 100).toFixed(0)}%) — pipeline skipped`);
  document.getElementById('pipeline-phase').textContent = 'DUPLICATE';
  clearInterval(mttdInterval);
  pipelineRunning = false;
  document.getElementById('run-btn').disabled = false;
}

function onError(ev) {
  setStatus("Error: " + (ev.message || '?'));
  clearInterval(mttdInterval);
  pipelineRunning = false;
  document.getElementById('run-btn').disabled = false;
}

/* ── HITL action ──────────────────────────────────────────────────────── */
async function hitlDecide(approved) {
  const feedback = document.getElementById('hitl-feedback').value.trim();
  document.getElementById('hitl-overlay').classList.remove('active');
  try {
    await fetch('/api/hitl/decide', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approved, feedback })
    });
  } catch (e) { console.error(e); }
}

/* ── API calls ────────────────────────────────────────────────────────── */
async function runPipeline() {
  if (pipelineRunning) return;

  // Detect input mode: structured JSON or plain string
  const inputEl = document.getElementById('event-input');
  let rawVal  = inputEl ? inputEl.value.trim() : DEFAULT_EVENT;
  if (!rawVal) return;

  // Sometimes users paste json wrapped in quotes
  if ((rawVal.startsWith('"') && rawVal.endsWith('"')) || (rawVal.startsWith("'") && rawVal.endsWith("'"))) {
    rawVal = rawVal.slice(1, -1).replace(/\\"/g, '"');
  }

  // Try to parse as JSON first (structured format)
  let event = rawVal;
  try {
    const parsed = JSON.parse(rawVal);
    if (parsed && typeof parsed === 'object' && parsed.error_message) {
      event = parsed; // pass structured dict
    }
  } catch (e) {
    // plain string — use as-is
  }

  const service = DEFAULT_SERVICE;
  try {
    const r = await fetch('/api/run-pipeline', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event, service })
    });
    const d = await r.json();
    if (d.error) setStatus("Error: " + d.error);
  } catch (e) { setStatus("Failed to start pipeline: " + e); }
}

async function injectBug() {
  setStatus("Injecting bug into demo file...");
  await fetch('/api/inject-bug', { method: 'POST' });
}

async function restoreBug() {
  setStatus("Restoring demo file...");
  await fetch('/api/restore', { method: 'POST' });
}

async function shutdownServer() {
  if (confirm("Shut down the Eidolon server?")) {
    setStatus("Shutting down...");
    try { await fetch('/api/shutdown', { method: 'POST' }); } catch (e) {}
    document.body.innerHTML = '<div style="display:flex;height:100vh;align-items:center;justify-content:center;background:#F7F2E8;font-family:Cinzel,serif;font-size:18px;letter-spacing:.2em;color:#5C4030;">EIDOLON TERMINATED</div>';
  }
}

async function wipeMemory() {
  if (pipelineRunning) { setStatus('Cannot wipe while pipeline is running'); return; }
  try {
    const r = await fetch('/api/wipe-memory', { method: 'POST' });
    const d = await r.json();
    if (d.error) { setStatus('Wipe failed: ' + d.error); return; }
    // Visual feedback inline — no confirm dialog needed for demo
    setStatus(`Memory wiped — ${d.solutions_removed} solution(s) cleared. Next run uses full pipeline.`);
  } catch (e) { setStatus('Wipe error: ' + e); }
}

function onMemoryWiped(ev) {
  const n = ev.solutions_removed || 0;
  // Flash a banner in the Ghost Mirror panel
  const body = document.getElementById('ghost-body');
  const banner = document.createElement('div');
  banner.style.cssText = 'padding:10px 14px;background:var(--hash-dim);border:1px solid var(--hash-color);border-left:3px solid var(--hash-color);border-radius:2px;margin-bottom:8px;animation:card-appear 0.3s ease;';
  banner.innerHTML = `
    <div style="font-family:var(--font-title);font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--hash-color);margin-bottom:3px;">Solution Library Cleared</div>
    <div style="font-size:13px;color:var(--ink-mid);">${n} stored solution${n!==1?'s':''} removed. The next run will execute the full diagnostic pipeline.</div>
  `;
  const empty = body.querySelector('.empty-state');
  if (empty) body.removeChild(empty);
  const existing = body.querySelector('[data-wipe]');
  if (existing) existing.remove();
  banner.setAttribute('data-wipe', '1');
  body.insertBefore(banner, body.firstChild);
}

async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const model = d.models?.triage || '';
    const el = document.getElementById('model-label');
    if (el && model) el.textContent = model.split('/').pop();
  } catch (e) {}
}

/* ── UI helpers ───────────────────────────────────────────────────────── */
function setStatus(msg) {
  document.getElementById('status-msg').textContent = msg;
}

function setNodeState(node, state) {
  const el = document.getElementById('node-' + node);
  if (!el) return;
  el.classList.remove('active', 'done', 'replay');
  if (state) el.classList.add(state);
}

function setNodeStatus(node, html) {
  const el = document.getElementById('status-' + node);
  if (el) el.innerHTML = html;
}

function addBrainNote(text) {
  const brainBody = document.getElementById('brain-body');
  const empty = brainBody.querySelector('.empty-state');
  if (empty) brainBody.removeChild(empty);
  const note = document.createElement('div');
  note.style.cssText = 'font-size:12px;color:var(--ink-dim);font-style:italic;padding:4px 0;font-family:var(--font-body);';
  note.textContent = text;
  brainBody.appendChild(note);
}

function resetUI() {
  ghostNodeCount = 0;
  tokenMappings = [];

  const ghostBody = document.getElementById('ghost-body');
  ghostBody.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">&#9670;</div>
      <div class="empty-label">Scanning codebase...</div>
      <div class="empty-desc">Ghost nodes will appear as the CPG is traversed</div>
    </div>`;

  const mirrorBody = document.getElementById('mirror-body');
  if (mirrorBody) mirrorBody.innerHTML = '';

  document.getElementById('brain-body').innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">&#9670;</div>
      <div class="empty-label">Awaiting hypotheses</div>
      <div class="empty-desc">The agent is building structural context</div>
    </div>`;

  document.getElementById('ghost-count').textContent = '0 nodes';
  document.getElementById('mttd-value').textContent = '0.0s';
  document.getElementById('mttd-value').style.color = 'var(--terracotta)';
  document.getElementById('mttd-fill').style.width = '0%';
  document.getElementById('hitl-feedback').value = '';

  NODES.forEach(n => {
    setNodeState(n, null);
    setNodeStatus(n, 'Waiting');
    const t = document.getElementById('time-' + n);
    if (t) t.textContent = '';
    const conn = document.getElementById('conn-' + NODES.indexOf(n));
    if (conn) { conn.classList.remove('flowing', 'done'); }
  });
}

function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ── Boot ─────────────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
  connectWS();

  // If ghost_stack_trace arrives embedded in a node_complete event, show it
  // We hook into the original handleEvent to check for stack trace data
  const _orig = handleEvent;

  // Sponsor name persistence
  const sponsorEl = document.getElementById('sponsor-name');
  const saved = localStorage.getItem('eidolon_sponsor');
  if (saved && sponsorEl) sponsorEl.value = saved;
  if (sponsorEl) {
    sponsorEl.addEventListener('input', () => {
      localStorage.setItem('eidolon_sponsor', sponsorEl.value);
    });
  }

  // Event input — populate from default
  const eventInput = document.getElementById('event-input');
  if (eventInput) eventInput.value = DEFAULT_EVENT;
});
