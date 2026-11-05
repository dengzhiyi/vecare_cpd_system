'use strict';

/* ------------------------------------------------------------------ */
/* 全局状态                                                             */
const $ = (id) => document.getElementById(id);

const S = {
  view: 'live',
  lastNonAboutView: 'live',
  scenarios: [],
  params: {},
  selected: 'sleeping',
  sensitivity: 1.0,
  speed: 1.0,
  hwFMax: 20000,    // 默认 20 kHz；可选 24k/20k/18k/16k/14k
  bands: [],
  truthVisible: true,
  live: {
    frames: [], events: [], status: null, scenario: null,
    lastSeq: 0, sessionId: null, polling: false,
  },
  replay: {
    frames: [], events: [], meta: null, summary: null, sessionId: null,
    index: 0, playing: false, speed: 1, timer: null,
  },
  audio: { ctx: null, lastBeep: 0 },
  audioViz: { hwFMax: 20000, seconds: null, data: null, busy: false },
};

const LIVE_WINDOW_S = 30;
const LIVE_FRAME_LIMIT = 900;
const HEAT_ROWS = 120;
const HEAT_STOPS = [
  [0.0, [7, 20, 30]],
  [0.25, [13, 66, 82]],
  [0.5, [30, 140, 132]],
  [0.75, [126, 216, 183]],
  [1.0, [242, 179, 92]],
];

const CAT_LABEL = { baseline: 'BASELINE', presence: 'PRESENCE', critical: 'CRITICAL' };

/* ------------------------------------------------------------------ */
/* 工具函数                                                             */
async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  let body = null;
  try { body = await res.json(); } catch (_) { /* 解析失败时忽略 */ }
  if (!res.ok) throw new Error((body && body.error) || `HTTP ${res.status}`);
  return body;
}

function esc(v) {
  return String(v == null ? '' : v)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt1 = (v) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(1));
const fmt2 = (v) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(2));
const fmt3 = (v) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(3));
const pct = (v) => (v == null ? 0 : clamp(Number(v) * 100, 0, 100));

function toast(message) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.classList.add('show'), 10);
  setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 300);
  }, 3200);
}

function debounce(fn, ms) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

/* ------------------------------------------------------------------ */
/* 启动                                                                 */
document.addEventListener('DOMContentLoaded', init);

async function init() {
  bindUI();
  try {
    /* 初始化：GET /api/scenarios 一次取回场景清单、默认参数与硬件频带档位。
       真正的“仿真”由下方 /api/live/start 触发，Web UI 本身不做声学计算。 */
    const data = await api('/api/scenarios');
    S.scenarios = data.scenarios || [];
    S.params = data.params || {};
    S.bands = S.params.bands || [];
    renderBandChips();
    if (S.scenarios.length && !S.scenarios.some((s) => s.key === S.selected)) {
      S.selected = S.scenarios[0].key;
    }
    renderScenarioList();
  } catch (err) {
    $('selectedScenarioNote').textContent = `无法读取场景：${err.message}`;
  }
  try {
    await refreshStatus();
    renderLive();
  } catch (err) {
    console.warn(err);
  }
  loadSessions();
  setInterval(pollLive, 320);
  window.addEventListener('resize', debounce(renderCurrent, 180));
}

function bindUI() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => setView(tab.dataset.view));
  });

  $('startBtn').addEventListener('click', startMonitor);
  $('pauseBtn').addEventListener('click', togglePause);
  $('stopBtn').addEventListener('click', stopMonitor);
  $('calibrateBtn').addEventListener('click', calibrateEmpty);
  $('ackBtn').addEventListener('click', acknowledge);

  $('bandChips').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    const raw = chip.dataset.hw;
    S.hwFMax = raw == null || raw === '' ? 20000 : Number(raw);
    renderBandChips();
  });

  $('audioBandChips').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    const raw = chip.dataset.hw;
    S.audioViz.hwFMax = raw == null || raw === '' ? 20000 : Number(raw);
    $('audioBandChips').querySelectorAll('.chip').forEach((c) => {
      c.classList.toggle('active', c === chip);
    });
  });

  $('audioFile').addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    $('audioFileName').textContent = file ? file.name : '选择 WAV 录音';
  });
  $('audioLength').addEventListener('change', (e) => {
    const raw = e.target.value;
    /* 空值表示总时长；其余为只分析录音开头的秒数，由后端截取。 */
    S.audioViz.seconds = raw === '' ? null : Number(raw);
  });
  $('audioAnalyzeBtn').addEventListener('click', analyzeAudioFile);

  $('sensitivity').addEventListener('input', (e) => {
    S.sensitivity = Number(e.target.value);
    $('sensitivityValue').textContent = S.sensitivity.toFixed(2);
  });

  $('liveSpeedChips').querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      S.speed = Number(chip.dataset.speed);
      $('liveSpeedChips').querySelectorAll('.chip').forEach((c) => c.classList.toggle('active', c === chip));
      if (S.live.status && S.live.status.running) {
        /* 运行中改倍速：POST /api/live/speed，请求体 {speed: 1/2/4/8/16}。
           后端只改变仿真推进速率，不重置 CSI 缓存和检测状态。 */
        api('/api/live/speed', { method: 'POST', body: JSON.stringify({ speed: S.speed }) })
          .catch((err) => toast(err.message));
      }
    });
  });

  $('replaySpeedChips').querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      S.replay.speed = Number(chip.dataset.speed);
      $('replaySpeedChips').querySelectorAll('.chip').forEach((c) => c.classList.toggle('active', c === chip));
      if (S.replay.playing) scheduleReplay();
    });
  });

  $('replayPlay').addEventListener('click', toggleReplay);
  $('replaySeek').addEventListener('input', (e) => {
    if (!S.replay.frames.length) return;
    S.replay.index = clamp(Number(e.target.value), 0, S.replay.frames.length - 1);
    renderReplayFrame();
  });

  $('truthToggle').addEventListener('change', (e) => {
    S.truthVisible = e.target.checked;
    document.querySelector('.truth-card').classList.toggle('dim', !S.truthVisible);
  });

  $('aboutClose').addEventListener('click', () => setView(S.lastNonAboutView));
}

/* ------------------------------------------------------------------ */
/* 实时增量拉流：GET /api/live?since=<lastSeq>&session=<sessionId>
   后端返回自 since 之后的新决策帧、当前检测器状态和事件列表。仿真线程在
   /api/live/start 后异步运行，因此 UI 用 320 ms 轮询模拟“实时流”；since
   与 session 共同做增量过滤，切场景后 session 变化会触发本地缓存清空。 */
async function pollLive() {
  if (S.live.polling) return;
  S.live.polling = true;
  try {
    const query = new URLSearchParams({ since: String(S.live.lastSeq || 0) });
    if (S.live.sessionId) query.set('session', S.live.sessionId);
    const data = await api(`/api/live?${query.toString()}`);
    const status = data.status || {};
    if (status.session_id !== S.live.sessionId) {
      S.live.sessionId = status.session_id;
      S.live.frames = [];
      S.live.events = [];
      S.live.lastSeq = 0;
    }
    if (Array.isArray(data.frames) && data.frames.length) {
      S.live.frames.push(...data.frames);
      if (S.live.frames.length > LIVE_FRAME_LIMIT) {
        S.live.frames = S.live.frames.slice(-LIVE_FRAME_LIMIT);
      }
      S.live.lastSeq = S.live.frames[S.live.frames.length - 1].seq;
    }
    S.live.status = status;
    S.live.scenario = data.scenario || null;
    S.live.events = data.events || [];
    syncControls();
    if (S.view === 'live') {
      renderLive();
      maybeBeep(S.live.frames.length ? S.live.frames[S.live.frames.length - 1].state : status.detector);
    }
  } catch (_) {
    /* 服务器可能仍在启动，下一轮轮询会自动重试 */
  } finally {
    S.live.polling = false;
  }
}

async function refreshStatus() {
  S.live.status = await api('/api/status');
  S.live.sessionId = S.live.status.session_id;
  syncControls();
}

/* ------------------------------------------------------------------ */
/* 实时监测控件                                                         */
function renderScenarioList() {
  const list = $('scenarioList');
  list.innerHTML = S.scenarios.map((s) => `
    <button class="scenario-card ${s.key === S.selected ? 'selected' : ''}" data-key="${esc(s.key)}">
      <b>${esc(s.name)}</b>
      <small>${esc(s.subtitle)} · ${s.duration_s}s</small>
      <span class="cat ${esc(s.category)}">${CAT_LABEL[s.category] || esc(s.category)}</span>
    </button>
  `).join('');
  list.querySelectorAll('.scenario-card').forEach((btn) => {
    btn.addEventListener('click', () => {
      S.selected = btn.dataset.key;
      list.querySelectorAll('.scenario-card').forEach((b) => b.classList.toggle('selected', b === btn));
      renderScenarioNote();
    });
  });
  renderScenarioNote();
}

function renderScenarioNote() {
  const sc = S.scenarios.find((s) => s.key === S.selected);
  if (!sc) return;
  $('selectedScenarioNote').textContent = `${sc.description} 预期：${sc.expected}`;
}

/* 硬件频响上限档位。频带由后端 plan_band 统一规划，前端只展示后端返回的
   载波数与上下边界，不在浏览器里重复推导。 */
const fmtKHz = (v) => (v / 1000).toFixed(1);

function renderBandChips() {
  const box = $('bandChips');
  if (!box) return;
  if (!S.bands.length) {
    box.innerHTML = '<button class="chip active" data-hw="20000">20k</button>';
    return;
  }
  box.innerHTML = S.bands.map((b) => `
    <button class="chip ${b.hw_f_max === S.hwFMax ? 'active' : ''}"
            data-hw="${b.hw_f_max}"
            title="${esc(b.describe)}">${esc(b.label)}</button>
  `).join('');
  syncBandNote();
}

function syncBandNote() {
  const band = S.bands.find((b) => b.hw_f_max === S.hwFMax)
    || S.bands.find((b) => b.hw_f_max === 20000)
    || S.bands[0];
  if (!band) return;
  $('bandValue').textContent = `${band.hw_f_max / 1000}k`;
  $('bandRange').textContent = `${fmtKHz(band.f_lo_hz)}–${fmtKHz(band.f_hi_hz)} kHz`;
  $('bandCars').textContent = `${band.n_carriers} 路 · ${Math.round(band.spacing_hz)} Hz`;
}

/* 启动仿真：POST /api/live/start
   请求体 {scenario, sensitivity, speed, hw_f_max}：
   - scenario：场景键（见 GET /api/scenarios）
   - sensitivity：检测灵敏度倍率
   - speed：仿真倍速（1/2/4/8/16）
   - hw_f_max：硬件频响上限（14k/16k/18k/20k/24k，默认 20k），决定载波频带
   后端创建 SimulationEngine 线程，按 50 Hz 合成 CSI 并驱动 SAS 检测器；
   返回 status.session_id，同时把本次运行写入 data/sessions 供回放。 */
async function startMonitor() {
  ensureAudio();
  $('startBtn').disabled = true;
  try {
    const status = await api('/api/live/start', {
      method: 'POST',
      body: JSON.stringify({
        scenario: S.selected,
        sensitivity: S.sensitivity,
        speed: S.speed,
        hw_f_max: S.hwFMax,
      }),
    });
    S.live.status = status;
    S.live.sessionId = status.session_id;
    S.live.frames = [];
    S.live.events = [];
    S.live.lastSeq = 0;
    S.replay.index = 0;
    setView('live');
    await pollLive();
    renderLive();
  } catch (err) {
    toast(`启动失败：${err.message}`);
    syncControls();
  }
}

/* 暂停/继续：POST /api/live/pause，请求体 {paused: true|false}。
   只冻结仿真时钟，不清空已有帧和检测状态。 */
async function togglePause() {
  if (!S.live.status || !S.live.status.running) return;
  try {
    const status = await api('/api/live/pause', {
      method: 'POST',
      body: JSON.stringify({ paused: !S.live.status.paused }),
    });
    S.live.status = status;
    syncControls();
    renderLive();
    if (S.live.frames.length) {
      renderDashboard(S.live.frames, S.live.status.detector, {
        mode: 'live', events: S.live.events, animated: !status.paused,
      });
    }
  } catch (err) {
    toast(err.message);
  }
}

/* 停止：POST /api/live/stop，结束仿真线程并写入会话 meta（状态、总时长、
   帧数、事件数与评估摘要）。停止后本次会话出现在“日志回放”列表。 */
async function stopMonitor() {
  try {
    await api('/api/live/stop', { method: 'POST', body: '{}' });
    await refreshStatus();
    await pollLive();
    loadSessions();
  } catch (err) {
    toast(err.message);
  }
}

/* 确认静音：POST /api/live/ack，只取消告警声与告警态，不解除存在锁存；
   若随后连续 25 s 无运动/呼吸证据，后端会自行清除告警并回到监测中。 */
async function acknowledge() {
  try {
    await api('/api/live/ack', { method: 'POST', body: '{}' });
    await pollLive();
  } catch (err) {
    toast(err.message);
  }
}

function syncControls() {
  const st = S.live.status || {};
  const running = !!st.running;
  const paused = !!st.paused;
  $('startBtn').disabled = running;
  $('startBtn').textContent = running ? '监测进行中' : (st.scenario ? '重新开始' : '开始监测');
  $('pauseBtn').disabled = !running;
  $('pauseBtn').textContent = paused ? '继续' : '暂停';
  $('stopBtn').disabled = !running;
  $('calibrateBtn').disabled = running;
  $('sensitivity').disabled = running;
  $('scenarioList').classList.toggle('locked', running);
  $('bandChips').classList.toggle('locked', running);
  $('sessionId').textContent = st.session_id || '—';
  renderBandRun(st);
  renderCalibration(st.calibration);
}

/* 空车标定状态：GET /api/calibration 返回基线文件内容。
   已标定时显示采集时长与基线能量；未标定时明确说明冷启动自适应。 */
function renderCalibration(cal) {
  if (!cal) return;
  const p = cal.profile;
  const stored = cal.stored;
  if (p) {
    const sourceLabel = p.source === 'wav' ? '真实录音' : '仿真';
    $('calibrationState').textContent = `已标定 ${p.duration_s || 5} s · ${sourceLabel}`;
    $('calibrationNote').textContent =
      `基线能量 ${Number(p.baseline_energy).toFixed(4)} · 相干门限 ` +
      `${Number(p.coherence_threshold).toFixed(3)} · ${p.n_carriers} 路载波`;
  } else if (stored) {
    $('calibrationState').textContent = '未用于仿真';
    $('calibrationNote').textContent =
      '检测到真实录音基线。它只适用于 WAV 回放/实车，仿真监测仍用冷启动自适应。';
  } else {
    $('calibrationState').textContent = '未标定';
    $('calibrationNote').textContent =
      '未标定时从开机第一帧起自适应估计背景，不要求先静置 5 s。';
  }
}

/* 执行空车标定：POST /api/calibration {duration_s, hw_f_max}。
   后端在空车场景下离线跑 5 s 仿真并写 data/calibration/empty_cabin.json；
   完成后下一次监测启动会自动加载这份基线。 */
async function calibrateEmpty() {
  if (S.live.status && S.live.status.running) {
    toast('请先停止当前监测，再执行空车标定');
    return;
  }
  const btn = $('calibrateBtn');
  btn.disabled = true;
  btn.textContent = '标定中…';
  try {
    const data = await api('/api/calibration', {
      method: 'POST',
      body: JSON.stringify({ duration_s: 5, hw_f_max: S.hwFMax })
    });
    renderCalibration(data);
    toast('空车标定完成，下次启动将加载该基线');
    await loadSessions();
  } catch (err) {
    toast(err.message);
  } finally {
    btn.textContent = '执行空车标定';
    btn.disabled = !! (S.live.status && S.live.status.running);
  }
}

function renderBandRun(st) {
  const r = st.band_range_hz;
  if (!r || r[0] == null) {
    $('bandRun').textContent = '—';
    return;
  }
  const planned = st.carriers_planned;
  const used = st.carriers_used;
  let txt = `${fmtKHz(r[0])}–${fmtKHz(r[1])} kHz`;
  if (used == null) txt += planned ? ` · ${planned} 路` : '';
  else if (planned && used < planned) txt += ` · ${planned}→${used} 路（已裁剪）`;
  else txt += ` · ${used} 路`;
  $('bandRun').textContent = txt;
}

/* ------------------------------------------------------------------ */
/* 视图切换                                                              */
function setView(view) {
  if (view === 'about') {
    if (S.view !== 'about') S.lastNonAboutView = S.view;
    S.view = 'about';
    $('aboutView').classList.remove('hidden');
  } else {
    S.view = view;
    $('aboutView').classList.add('hidden');
    $('liveControls').classList.toggle('hidden', view !== 'live');
    $('replayControls').classList.toggle('hidden', view !== 'replay');
    $('mainLayout').classList.toggle('hidden', view === 'audio');
    $('audioView').classList.toggle('hidden', view !== 'audio');
  }
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.view === S.view));

  if (view === 'live') {
    renderLive();
  } else if (view === 'replay') {
    loadSessions();
    renderReplayFrame();
  } else if (view === 'audio') {
    renderAudioViz();
  }
}

function renderCurrent() {
  if (S.view === 'live') renderLive();
  else if (S.view === 'replay') renderReplayFrame();
  else if (S.view === 'audio') renderAudioViz();
}

/* ------------------------------------------------------------------ */
/* 仪表盘渲染                                                            */
function renderLive() {
  syncControls();
  if (S.view !== 'live') return;
  const st = S.live.status;
  const det = (st && st.detector) || (S.live.frames.length ? S.live.frames[S.live.frames.length - 1].state : null);
  renderDashboard(S.live.frames, det, {
    mode: 'live',
    events: S.live.events,
    animated: !!(st && st.running && !st.paused),
  });
}

function renderDashboard(frames, det, opts = {}) {
  const last = frames.length ? frames[frames.length - 1] : null;
  renderHero(det, last, opts);
  renderCabin(det, opts);
  renderTruth(last);
  drawCharts(frames, det, last);
  renderEvents(opts.events || [], last, opts);
  renderAlarmBanner(det, opts);
}

/* 状态文案 */
function stateMeta(state, det, last, opts) {
  const t = last ? last.t : (det ? det.t : 0);
  const presenceFor = det && det.presence_since != null && last ? Math.max(0, last.t - det.presence_since) : null;
  switch (state) {
    case 'PRESENCE': {
      const extra = presenceFor != null ? `已持续 ${fmt1(presenceFor)} s` : '存在证据持续';
      const left = det && det.countdown != null ? `，${fmt1(det.countdown)} s 后触发告警` : '';
      return { label: '检测到儿童', cls: 'presence', sub: `${extra}${left}。系统将在 10 s 响应窗口内报警。` };
    }
    case 'ALARM': {
      const reason = det && det.alarm_reason === 'APNEA' ? '呼吸暂停' : '儿童存在';
      const since = det && det.alarm_since != null && last ? fmt1(Math.max(0, last.t - det.alarm_since)) : null;
      return {
        label: reason === '呼吸暂停' ? '呼吸暂停告警' : '儿童存在告警', cls: 'alarm',
        sub: `已触发${reason}告警${since != null ? ` ${since} s` : ''}。请立即检查后排儿童并按需静音。`,
      };
    }
    case 'ACKED':
      return {
        label: '告警已静音', cls: 'acked',
        sub: '用户已确认告警，但儿童仍在车内；存在状态将保持，直至证据消失。',
      };
    case 'MONITORING':
      return {
        label: opts.mode === 'replay' ? '监测中 · 无儿童' : '监测中 · 无儿童', cls: 'safe',
        sub: '未检测到持续存在证据，系统保持安全监测；5 s 证据窗口内占比低于 30%。',
      };
    default:
      return {
        label: '待机', cls: 'idle',
        sub: '选择一个仿真场景，开始车内儿童存在检测。',
      };
  }
}

function renderHero(det, last, opts) {
  const state = det ? det.state : 'IDLE';
  const meta = stateMeta(state, det, last, opts);
  const stateLabel = $('stateLabel');
  stateLabel.textContent = meta.label;
  $('stateSub').textContent = meta.sub;
  $('stateMode').textContent = state;
  $('stateDot').className = `state-dot ${meta.cls}`;
  $('hero').classList.toggle('presence', state === 'PRESENCE' || state === 'ACKED');
  $('hero').classList.toggle('alarm', state === 'ALARM');

  const bpm = det ? det.bpm : null;
  const conf = det ? det.breath_confidence : 0;
  $('bpmValue').textContent = bpm == null ? '—' : fmt1(bpm);
  $('confBar').style.width = `${pct(conf)}%`;
  $('confText').textContent = bpm == null ? '置信度 —' : `置信度 ${(conf * 100).toFixed(0)}%`;

  const motion = last && last.features ? last.features.motion : null;
  const threshold = det ? det.motion_threshold : (last ? last.state.motion_threshold : null);
  $('motionValue').textContent = motion == null ? '—' : fmt3(motion);
  const motionRatio = motion == null || !threshold ? 0 : motion / Math.max(0.001, threshold * 2);
  $('motionBar').style.width = `${pct(motionRatio)}%`;
  $('motionBar').style.background = motion != null && threshold != null && motion > threshold
    ? 'linear-gradient(90deg, #f2b35c, #ff8a6b)' : 'linear-gradient(90deg, #47e0c0, #5cc8ff)';
  $('motionText').textContent = threshold == null ? '阈值 —' : `能量比阈值 ${fmt2(threshold)}${motion != null && motion > threshold ? ' · 已超阈' : ''}`;

  const score = det ? det.presence_score : null;
  $('scoreValue').textContent = score == null ? '—' : (score * 100).toFixed(0);
  $('scoreBar').style.width = `${pct(score)}%`;
  $('scoreBar').style.background = state === 'ALARM'
    ? 'linear-gradient(90deg, #ff6070, #ff8a6b)' : 'linear-gradient(90deg, #47e0c0, #5cc8ff)';

  const budget = det && det.response_budget_s ? det.response_budget_s : 10;
  let cdValue = '—';
  let cdText = `${budget.toFixed(0)} s 响应预算`;
  let cdRatio = 0;
  if (det && det.alarm) {
    cdText = det.acknowledged ? '告警已静音' : '告警已触发';
    cdRatio = 1;
  } else if (det && det.presence && det.countdown != null) {
    cdValue = fmt1(det.countdown);
    cdRatio = clamp(det.countdown / Math.max(1, budget), 0, 1);
    cdText = '响应预算倒计时（自首个证据）';
  }
  $('countdownValue').textContent = cdValue;
  $('countdownText').textContent = cdText;
  $('countdownBar').style.width = `${cdRatio * 100}%`;
  $('countdownBar').style.background = det && det.alarm
    ? 'linear-gradient(90deg, #ff6070, #ff9a6b)' : 'linear-gradient(90deg, #f2b35c, #ffca7a)';

  const bootstrapping = !!(det && !det.calibrated);
  const progress = det ? det.calibration_progress || 0 : 0;
  $('calibBar').style.width = bootstrapping ? `${progress * 100}%` : (det ? '100%' : '0%');
  $('calibText').textContent = bootstrapping
    ? `冷启动背景估计 ${(progress * 100).toFixed(0)}%`
    : (det ? `背景基线已建立（${det.background_source}）· 运动阈值 ${fmt2(det.motion_threshold)}` : '背景噪声基线');
}

function renderCabin(det, opts) {
  const stage = $('cabinStage');
  const present = !!(det && det.presence);
  const alarm = !!(det && det.alarm && !det.acknowledged);
  const breathing = !!(present && det && det.bpm != null);
  stage.classList.toggle('present', present);
  stage.classList.toggle('breathing', breathing);
  stage.classList.toggle('alarm', alarm);
  $('childSeat').classList.toggle('current', present);
  $('waves').classList.toggle('active', !!opts.animated);
  $('waves').classList.toggle('detected', present);
  $('speakerNode').classList.toggle('transmitting', !!opts.animated);
  $('micNode').classList.toggle('transmitting', !!opts.animated);
}

function renderTruth(last) {
  const truth = last && last.truth ? last.truth : null;
  const set = (id, text, cls) => {
    const el = $(id);
    el.textContent = text;
    el.className = cls || '';
  };
  if (!truth) {
    set('truthPresent', '—');
    set('truthBreathing', '—');
    set('truthBpm', '—');
    set('truthDistance', '—');
    return;
  }
  set('truthPresent', truth.present ? '是' : '否', truth.present ? 'ok' : '');
  set('truthBreathing', truth.breathing ? '是' : '否', truth.breathing ? 'ok' : '');
  set('truthBpm', truth.bpm == null ? '—' : `${fmt1(truth.bpm)} BPM`, truth.breathing ? 'ok' : '');
  set('truthDistance', truth.distance_m == null ? '—' : `${fmt2(truth.distance_m)} m`, '');
}

/* 事件流每次数据刷新（10 Hz）都会调用本函数。若每帧重写 innerHTML，
   所有条目的 event-in 动画会反复从头播放，看起来就是整列一直在闪缩。
   这里按内容签名比对：内容没变就不碰 DOM；只有新到的条目播放入场动画。 */
let renderedEventKeys = [];

function renderEvents(events, last, opts) {
  const list = $('eventList');
  let evs = events || [];
  if (opts.mode === 'replay' && last) {
    evs = evs.filter((ev) => ev.t <= last.t + 0.05);
  }
  $('eventCount').textContent = `${evs.length} 条`;
  if (!evs.length) {
    if (renderedEventKeys.length) {
      list.innerHTML = '<li class="event-empty">尚无事件。开始监测后，校准、检测与告警事件会记录在此。</li>';
      renderedEventKeys = [];
    }
    return;
  }

  const items = evs.slice(-40).reverse();
  const keys = items.map((ev) => `${ev.t}|${ev.kind || 'state'}|${ev.message}`);
  if (keys.length === renderedEventKeys.length
      && keys.every((key, i) => key === renderedEventKeys[i])) {
    return;
  }

  const previous = new Set(renderedEventKeys);
  list.innerHTML = items.map((ev, i) => `
    <li class="event-item ${esc(ev.kind || 'state')}${previous.has(keys[i]) ? ' static' : ''}">
      <time>${fmt1(ev.t)}s</time>
      <p>${esc(ev.message)}</p>
    </li>
  `).join('');
  renderedEventKeys = keys;
}

function renderAlarmBanner(det, opts) {
  const banner = $('alarmBanner');
  if (!det || !det.alarm) {
    banner.classList.add('hidden');
    return;
  }
  banner.classList.remove('hidden');
  banner.classList.toggle('muted', !!det.acknowledged);
  const apnea = det.alarm_reason === 'APNEA';
  $('alarmTitle').textContent = apnea ? '呼吸暂停告警' : '儿童存在告警';
  const since = det.alarm_since == null ? '' : ` · 触发于 ${fmt1(det.alarm_since)} s`;
  $('alarmDetail').textContent = det.acknowledged
    ? `告警已静音，但儿童仍在车内，请持续关注${since}`
    : (apnea
      ? `检测到儿童呼吸停止，请立即救援${since}`
      : `检测到儿童被留在车内，请立即检查后排${since}`);
  $('ackBtn').classList.toggle('hidden', opts.mode === 'replay' || !!det.acknowledged);
}

/* ------------------------------------------------------------------ */
/* 图表                                                                 */
function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, rect.width);
  const h = Math.max(1, rect.height);
  const pw = Math.round(w * dpr);
  const ph = Math.round(h * dpr);
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function plotRect(w, h, pad = {}) {
  const l = pad.l == null ? 44 : pad.l;
  const r = pad.r == null ? 12 : pad.r;
  const t = pad.t == null ? 10 : pad.t;
  const b = pad.b == null ? 24 : pad.b;
  return { l, t, r: w - r, b: h - b, w: Math.max(1, w - l - r), h: Math.max(1, h - t - b) };
}

function drawAxes(ctx, rect, xMin, xMax, yMin, yMax, opts = {}) {
  const fmtX = opts.fmtX || ((v) => `${Math.round(v)}s`);
  const fmtY = opts.fmtY || ((v) => v.toFixed(2));
  const xTicks = opts.xTicks || 5;
  const yTicks = opts.yTicks || 4;
  ctx.save();
  ctx.font = '9.5px "JetBrains Mono", Consolas, monospace';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  ctx.lineWidth = 1;

  for (let i = 0; i <= yTicks; i++) {
    const val = yMin + (yMax - yMin) * i / yTicks;
    const y = rect.b - rect.h * i / yTicks;
    ctx.strokeStyle = 'rgba(140, 175, 205, 0.075)';
    ctx.beginPath();
    ctx.moveTo(rect.l, y);
    ctx.lineTo(rect.r, y);
    ctx.stroke();
    ctx.fillStyle = '#5d6b80';
    ctx.fillText(fmtY(val), rect.l - 7, y);
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= xTicks; i++) {
    const val = xMin + (xMax - xMin) * i / xTicks;
    const x = rect.l + rect.w * i / xTicks;
    ctx.strokeStyle = 'rgba(140, 175, 205, 0.05)';
    ctx.beginPath();
    ctx.moveTo(x, rect.t);
    ctx.lineTo(x, rect.b);
    ctx.stroke();
    ctx.fillStyle = '#5d6b80';
    ctx.fillText(fmtX(val), x, rect.b + 7);
  }

  ctx.strokeStyle = 'rgba(140, 175, 205, 0.16)';
  ctx.beginPath();
  ctx.moveTo(rect.l, rect.b + 0.5);
  ctx.lineTo(rect.r, rect.b + 0.5);
  ctx.stroke();
  ctx.restore();
}

function xOf(v, rect, xMin, xMax) {
  return rect.l + rect.w * (v - xMin) / Math.max(1e-9, xMax - xMin);
}
function yOf(v, rect, yMin, yMax) {
  return rect.b - rect.h * (v - yMin) / Math.max(1e-9, yMax - yMin);
}

function drawSeries(ctx, pts, rect, xMin, xMax, yMin, yMax, color, width = 2, gapS = 2.5) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.l, rect.t, rect.w, rect.h);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  let pen = false;
  let prev = null;
  for (const p of pts) {
    if (p.y == null || !Number.isFinite(p.y)) { pen = false; prev = null; continue; }
    if (prev && Math.abs(p.x - prev.x) > gapS) pen = false;
    const X = xOf(p.x, rect, xMin, xMax);
    const Y = yOf(clamp(p.y, yMin, yMax), rect, yMin, yMax);
    if (pen) ctx.lineTo(X, Y); else { ctx.moveTo(X, Y); pen = true; }
    prev = p;
  }
  ctx.stroke();
  ctx.restore();
}

function drawDots(ctx, pts, rect, xMin, xMax, yMin, yMax, color, radius = 2, every = 1) {
  ctx.save();
  ctx.fillStyle = color;
  for (let i = 0; i < pts.length; i += every) {
    const p = pts[i];
    if (p.y == null || !Number.isFinite(p.y)) continue;
    ctx.beginPath();
    ctx.arc(xOf(p.x, rect, xMin, xMax), yOf(clamp(p.y, yMin, yMax), rect, yMin, yMax), radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawPresenceBands(ctx, frames, rect, xMin, xMax, yMin, yMax) {
  let start = null;
  let lastT = null;
  let alarm = false;
  const flush = () => {
    if (start == null) return;
    const x0 = xOf(Math.max(start, xMin), rect, xMin, xMax);
    const x1 = xOf(Math.min(lastT, xMax), rect, xMin, xMax);
    ctx.fillStyle = alarm ? 'rgba(255, 96, 112, 0.075)' : 'rgba(242, 179, 92, 0.055)';
    ctx.fillRect(x0, rect.t, Math.max(1, x1 - x0), rect.h);
    start = null;
  };
  for (const f of frames) {
    if (f.t < xMin || f.t > xMax) continue;
    if (f.state.presence) {
      if (start == null) { start = f.t; alarm = false; }
      lastT = f.t;
      alarm = alarm || f.state.alarm;
    } else if (start != null && f.t - lastT > 0.25) {
      flush();
    }
  }
  flush();
}

function drawEmpty(canvas, text) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.fillStyle = '#5d6b80';
  ctx.font = '11px "Inter", "PingFang SC", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, w / 2, h / 2);
}

function drawCharts(frames, det, last) {
  drawMotionChart(frames, det);
  drawBpmChart(frames, det);
  drawAcfChart(frames, det, last);
  drawHeatmap(frames);
}

function drawMotionChart(frames, det) {
  const canvas = $('motionChart');
  if (!frames.length) {
    drawEmpty(canvas, '等待监测数据');
    $('motionFoot').textContent = '—';
    return;
  }
  const { ctx, w, h } = setupCanvas(canvas);
  const end = frames[frames.length - 1].t;
  const xMax = Math.max(LIVE_WINDOW_S, end);
  const xMin = Math.max(0, end - LIVE_WINDOW_S);
  const sel = frames.filter((f) => f.t >= xMin && f.t <= xMax);
  const threshold = det ? det.motion_threshold : (sel.length ? sel[sel.length - 1].state.motion_threshold : 0.15);
  const vals = sel.map((f) => f.features.motion);
  const rawMax = Math.max(1.2, threshold * 1.7, ...vals);
  const yMax = niceCeil(rawMax, 0.1);
  const rect = plotRect(w, h);
  drawAxes(ctx, rect, xMin, xMax, 0, yMax, {
    fmtY: (v) => v.toFixed(2),
    fmtX: (v) => `${Math.round(v)}s`,
  });
  drawPresenceBands(ctx, sel, rect, xMin, xMax, 0, yMax);

  const thrY = yOf(threshold, rect, 0, yMax);
  ctx.save();
  ctx.strokeStyle = 'rgba(242, 179, 92, 0.65)';
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(rect.l, thrY);
  ctx.lineTo(rect.r, thrY);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = 'rgba(242, 179, 92, 0.85)';
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'bottom';
  ctx.fillText(`threshold ${threshold.toFixed(2)}`, rect.r - 4, thrY - 3);

  const pts = sel.map((f) => ({ x: f.t, y: f.features.motion }));
  drawSeries(ctx, pts, rect, xMin, xMax, 0, yMax, '#47e0c0', 2, 1.2);
  const over = pts.filter((p) => p.y != null && p.y > threshold);
  drawDots(ctx, over, rect, xMin, xMax, 0, yMax, '#f2b35c', 2.2, 1);

  const cur = frames.length && frames[frames.length - 1].features
    ? frames[frames.length - 1].features.motion : null;
  $('motionFoot').textContent = cur == null ? '—' : `当前 ${cur.toFixed(2)} · 阈值 ${threshold.toFixed(2)}`;
}

function drawBpmChart(frames, det) {
  const canvas = $('bpmChart');
  if (!frames.length) {
    drawEmpty(canvas, '等待 8 s 呼吸窗口');
    $('bpmFoot').textContent = '—';
    return;
  }
  const { ctx, w, h } = setupCanvas(canvas);
  const end = frames[frames.length - 1].t;
  const xMax = Math.max(40, end);
  const xMin = Math.max(0, end - 40);
  const sel = frames.filter((f) => f.t >= xMin && f.t <= xMax);
  const yMin = 8;
  const yMax = 45;
  const rect = plotRect(w, h);
  drawAxes(ctx, rect, xMin, xMax, yMin, yMax, {
    fmtY: (v) => v.toFixed(0),
    fmtX: (v) => `${Math.round(v)}s`,
  });

  ctx.fillStyle = 'rgba(71, 224, 192, 0.035)';
  ctx.fillRect(rect.l, yOf(40, rect, yMin, yMax), rect.w,
    yOf(12, rect, yMin, yMax) - yOf(40, rect, yMin, yMax));
  ctx.fillStyle = 'rgba(71, 224, 192, 0.7)';
  ctx.font = '9px "Inter", sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText('正常呼吸区间 12–40 BPM', rect.l + 5, yOf(40, rect, yMin, yMax) - 2);

  const pts = sel
    .filter((f) => f.state.bpm != null)
    .map((f) => ({ x: f.t, y: f.state.bpm }));
  drawSeries(ctx, pts, rect, xMin, xMax, yMin, yMax, '#5cc8ff', 2, 4);
  drawDots(ctx, pts, rect, xMin, xMax, yMin, yMax, '#8fdcff', 2.1, 5);

  const bpm = det ? det.bpm : (pts.length ? pts[pts.length - 1].y : null);
  $('bpmFoot').textContent = bpm == null ? '—' : `${fmt1(bpm)} BPM`;
}

function drawAcfChart(frames, det, last) {
  const canvas = $('acfChart');
  if (!last || !last.acf || last.acf.length < 3) {
    drawEmpty(canvas, '等待 8 s 呼吸窗口（ACF 构建中）');
    $('acfHint').textContent = '等待 8 s 呼吸窗口';
    $('acfFoot').textContent = '—';
    return;
  }
  const { ctx, w, h } = setupCanvas(canvas);
  const acf = last.acf;
  const step = last.acf_step_s || 0.04;
  const xMax = Math.max(0.5, (acf.length - 1) * step);
  let yMin = Math.min(-0.2, ...acf);
  yMin = Math.min(yMin, -0.05);
  const yMax = 1.02;
  const rect = plotRect(w, h);
  drawAxes(ctx, rect, 0, xMax, yMin, yMax, {
    fmtY: (v) => v.toFixed(2),
    fmtX: (v) => `${v.toFixed(1)}s`,
  });

  ctx.strokeStyle = 'rgba(140, 175, 205, 0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(rect.l, yOf(0, rect, yMin, yMax));
  ctx.lineTo(rect.r, yOf(0, rect, yMin, yMax));
  ctx.stroke();

  const pts = acf.map((v, i) => ({ x: i * step, y: v }));
  drawSeries(ctx, pts, rect, 0, xMax, yMin, yMax, '#47e0c0', 2.2, 99);

  const lag = last.features && last.features.lag_s;
  const bpm = last.features && last.features.bpm_raw;
  if (lag != null && lag <= xMax) {
    const x = xOf(lag, rect, 0, xMax);
    ctx.save();
    ctx.strokeStyle = 'rgba(242, 179, 92, 0.8)';
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(x, rect.t);
    ctx.lineTo(x, rect.b);
    ctx.stroke();
    ctx.restore();

    const idx = clamp(Math.round(lag / step), 0, acf.length - 1);
    ctx.fillStyle = '#f2b35c';
    ctx.beginPath();
    ctx.arc(x, yOf(clamp(acf[idx], yMin, yMax), rect, yMin, yMax), 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = lag > xMax * 0.6 ? 'right' : 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`τ_b ${lag.toFixed(2)}s`, x + (lag > xMax * 0.6 ? -5 : 5), rect.t + 3);
    if (bpm != null) {
      ctx.fillText(`${bpm.toFixed(1)} BPM`, x + (lag > xMax * 0.6 ? -5 : 5), rect.t + 15);
    }
  }

  $('acfHint').textContent = `8 s 窗口 · 滞后 0–${xMax.toFixed(1)} s`;
  const conf = last.features ? last.features.confidence : null;
  $('acfFoot').textContent = lag == null ? '—' : `峰值 τ_b ${lag.toFixed(2)} s · 置信度 ${fmt2(conf)}`;
}

function heatColor(t) {
  t = clamp(t, 0, 1);
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    const [p0, c0] = HEAT_STOPS[i - 1];
    const [p1, c1] = HEAT_STOPS[i];
    if (t <= p1) {
      const u = (t - p0) / (p1 - p0 || 1);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * u);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * u);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * u);
      return `rgb(${r},${g},${b})`;
    }
  }
  const c = HEAT_STOPS[HEAT_STOPS.length - 1][1];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function drawHeatmap(frames) {
  const canvas = $('heatmap');
  if (!frames.length || !frames[frames.length - 1].csi) {
    drawEmpty(canvas, '等待 CSI 数据');
    $('heatFoot').textContent = '—';
    return;
  }
  const { ctx, w, h } = setupCanvas(canvas);
  const rows = frames.slice(-HEAT_ROWS);
  const nSub = rows[0].csi.length;
  const rect = plotRect(w, h, { l: 32, r: 8, t: 8, b: 20 });

  const values = [];
  for (const r of rows) for (const v of r.csi) values.push(Math.log10(Math.max(v, 1e-6)));
  values.sort((a, b) => a - b);
  const p05 = values[Math.floor(values.length * 0.05)];
  const p95 = values[Math.floor(values.length * 0.95)];
  const lo = Math.min(p05, p95 - 0.05);
  const hi = Math.max(p95, lo + 0.05);

  const rowH = rect.h / Math.max(1, rows.length);
  const colW = rect.w / nSub;
  for (let i = 0; i < rows.length; i++) {
    const y = rect.b - (i + 1) * rowH;
    for (let f = 0; f < nSub; f++) {
      const v = Math.log10(Math.max(rows[i].csi[f], 1e-6));
      const t = (v - lo) / (hi - lo);
      ctx.fillStyle = heatColor(t);
      ctx.fillRect(rect.l + f * colW, y, Math.ceil(colW + 0.5), Math.ceil(rowH + 0.5));
    }
  }

  ctx.strokeStyle = 'rgba(140, 175, 205, 0.2)';
  ctx.strokeRect(rect.l, rect.t, rect.w, rect.h);
  ctx.fillStyle = '#5d6b80';
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText('−12s', rect.l - 5, rect.t + 2);
  ctx.textBaseline = 'bottom';
  ctx.fillText('现在', rect.l - 5, rect.b - 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('1', rect.l + colW / 2, rect.b + 6);
  ctx.fillText(`${nSub}`, rect.r - colW / 2, rect.b + 6);
  ctx.textAlign = 'center';
  ctx.fillText('子载波', rect.l + rect.w / 2, rect.b + 6);

  $('heatFoot').textContent = `${rows.length} × ${nSub}`;
}

function niceCeil(value, minStep = 0.1) {
  const v = Math.max(value, minStep);
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / p;
  const m = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return m * p;
}

/* ------------------------------------------------------------------ */
/* 日志回放                                                              */
/* 日志回放列表：GET /api/sessions 返回 data/sessions 下每个会话的摘要
   （场景、状态、时长、帧数、是否告警），不包含 CSI 原始音频。 */
async function loadSessions() {
  try {
    const data = await api('/api/sessions');
    const sessions = data.sessions || [];
    const list = $('sessionList');
    if (!sessions.length) {
      list.innerHTML = '<p class="empty-note">暂无已保存的会话。先在“实时监测”中运行一个场景。</p>';
      return;
    }
    list.innerHTML = sessions.map((s) => {
      const selected = s.id === S.replay.sessionId ? 'selected' : '';
      const summary = s.summary || {};
      const alarm = !!summary.alarm_at;
      const tag = s.status === 'running'
        ? '<span class="tag">REC</span>'
        : `<span class="tag ${alarm ? '' : 'safe'}">${alarm ? 'ALARM' : 'SAFE'}</span>`;
      const started = (s.started_at || '').replace('T', ' ').slice(0, 16);
      return `
        <button class="session-item ${selected}" data-id="${esc(s.id)}">
          <div class="row1"><b>${esc(s.scenario_name || s.scenario_key || s.id)}</b>${tag}</div>
          <small>${esc(started)} · ${fmt1(s.duration_s)}s · ${s.frames || 0} 帧</small>
        </button>`;
    }).join('');
    list.querySelectorAll('.session-item').forEach((btn) => {
      btn.addEventListener('click', () => selectSession(btn.dataset.id));
    });
  } catch (err) {
    console.warn(err);
  }
}

/* 载入会话：GET /api/sessions/<id> 取回 frames.jsonl / events.jsonl / meta.json；
   meta 中的 summary 用于展示首次确认、告警时刻与响应时间。
   下载链接 GET /api/sessions/<id>/download/<file> 导出原始 JSONL。 */
async function selectSession(id) {
  try {
    const data = await api(`/api/sessions/${encodeURIComponent(id)}`);
    pauseReplay();
    S.replay.frames = data.frames || [];
    S.replay.events = data.events || [];
    S.replay.meta = data.meta || null;
    S.replay.summary = (data.meta && data.meta.summary) || null;
    S.replay.sessionId = id;
    S.replay.index = 0;
    $('replaySeek').max = Math.max(0, S.replay.frames.length - 1);
    $('replaySeek').value = 0;
    $('replaySeek').disabled = S.replay.frames.length < 2;
    $('replayPlay').disabled = S.replay.frames.length === 0;
    document.querySelectorAll('.session-item').forEach((b) => b.classList.toggle('selected', b.dataset.id === id));
    const links = ['meta.json', 'frames.jsonl', 'events.jsonl'];
    $('downloadLinks').innerHTML = links.map((f) =>
      `<a href="/api/sessions/${encodeURIComponent(id)}/download/${f}" download>${f}</a>`).join('');
    renderReplayFrame();
    toast(`已载入会话 ${id}`);
  } catch (err) {
    toast(`载入失败：${err.message}`);
  }
}

function renderReplayFrame() {
  if (S.view !== 'replay') return;
  const r = S.replay;
  if (!r.frames.length) {
    renderDashboard([], null, { mode: 'replay', events: [], animated: false });
    $('replayTime').textContent = '0.0 / 0.0 s';
    return;
  }
  r.index = clamp(r.index, 0, r.frames.length - 1);
  const frame = r.frames[r.index];
  const lastT = r.frames[r.frames.length - 1].t;
  renderDashboard(r.frames.slice(0, r.index + 1), frame.state, {
    mode: 'replay',
    events: r.events,
    animated: r.playing,
  });
  $('replayTime').textContent = `${frame.t.toFixed(1)} / ${lastT.toFixed(1)} s`;
  $('replaySeek').value = r.index;
  $('replayPlay').textContent = r.playing ? '暂停' : '播放';
}

function toggleReplay() {
  if (!S.replay.frames.length) return;
  if (S.replay.playing) {
    pauseReplay();
  } else {
    if (S.replay.index >= S.replay.frames.length - 1) S.replay.index = 0;
    S.replay.playing = true;
    $('replayPlay').textContent = '暂停';
    scheduleReplay();
  }
}

function pauseReplay() {
  S.replay.playing = false;
  clearTimeout(S.replay.timer);
  S.replay.timer = null;
  $('replayPlay').textContent = '播放';
}

function scheduleReplay() {
  clearTimeout(S.replay.timer);
  if (!S.replay.playing) return;
  const delay = Math.max(16, 1000 / (10 * S.replay.speed));
  S.replay.timer = setTimeout(() => {
    if (!S.replay.playing) return;
    if (S.replay.index >= S.replay.frames.length - 1) {
      pauseReplay();
      return;
    }
    S.replay.index += 1;
    renderReplayFrame();
    scheduleReplay();
  }, delay);
}

/* ------------------------------------------------------------------ */
/* 告警音                                                                */
function ensureAudio() {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!S.audio.ctx && AC) {
    try { S.audio.ctx = new AC(); } catch (_) { /* 浏览器不支持时静默降级 */ }
  }
  if (S.audio.ctx && S.audio.ctx.state === 'suspended') {
    S.audio.ctx.resume().catch(() => {});
  }
}

function maybeBeep(det) {
  if (!det || !det.alarm || det.acknowledged) return;
  if (!S.audio.ctx || S.audio.ctx.state !== 'running') return;
  const now = Date.now();
  if (now - S.audio.lastBeep < 1800) return;
  S.audio.lastBeep = now;
  beep();
}

function beep() {
  const ctx = S.audio.ctx;
  if (!ctx) return;
  const t0 = ctx.currentTime;
  [0, 0.18].forEach((offset, i) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'square';
    osc.frequency.value = i === 0 ? 880 : 660;
    gain.gain.setValueAtTime(0.0001, t0 + offset);
    gain.gain.exponentialRampToValueAtTime(0.045, t0 + offset + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + offset + 0.14);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t0 + offset);
    osc.stop(t0 + offset + 0.16);
  });
}

/* ------------------------------------------------------------------ */
/* 录音音频可视化                                                        */
async function analyzeAudioFile() {
  const input = $('audioFile');
  const file = input.files && input.files[0];
  if (!file) { toast('请先选择一段 WAV 录音'); return; }
  if (S.audioViz.busy) return;
  S.audioViz.busy = true;
  $('audioAnalyzeBtn').disabled = true;
  $('audioStatus').textContent = `正在分析 ${file.name}…`;
  try {
    /* 直接把 File 作为请求体发送，避免 multipart；hw_f_max 必须与录制时一致。 */
    const query = new URLSearchParams({ name: file.name });
    if (S.audioViz.hwFMax != null) query.set('hw_f_max', S.audioViz.hwFMax);
    if (S.audioViz.seconds != null) query.set('seconds', S.audioViz.seconds);
    const data = await api(`/api/audio/analyze?${query.toString()}`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: file,
    });
    S.audioViz.data = data;
    const qc = data.qc || {};
    const durText = data.file_duration_s && data.file_duration_s > data.duration_s + 0.05
      ? `${data.duration_s} / ${data.file_duration_s} s`
      : `${data.duration_s} s`;
    $('audioStatus').textContent =
      `${data.path} · ${data.fs} Hz · ${durText} · ${data.band.describe}`;
    $('audioQc').textContent = `${qc.status || '—'} · ${(qc.checks || []).length} 项检查`;
    renderAudioViz();
  } catch (err) {
    $('audioStatus').textContent = `分析失败：${err.message}`;
    toast(`分析失败：${err.message}`);
  } finally {
    S.audioViz.busy = false;
    $('audioAnalyzeBtn').disabled = false;
  }
}

function renderAudioViz() {
  const data = S.audioViz.data;
  if (!data) return;
  renderAudioParams(data);
  drawAudioWave(data);
  drawAudioSpectrum(data);
  drawAudioCsi(data);
  drawAudioCarriers(data);
  drawAudioMotion(data);
  drawAudioBpm(data);
}

function renderAudioParams(data) {
  const params = data.params || [];
  $('audioParams').innerHTML = params.length
    ? params.map((p) => {
      const status = String(p.status || 'INFO').toLowerCase();
      return `<div class="param-row">
        <span>${esc(p.name)}</span>
        <span><i class="pstat ${status}">${esc(p.status || 'INFO')}</i>${esc(p.value)}</span>
      </div>`;
    }).join('')
    : '<p class="empty-note">没有可用参数</p>';

  const events = data.events || [];
  $('audioEvents').innerHTML = events.slice(0, 10).map((ev) =>
    `<span class="ev-chip"><b>${fmt1(ev.t)} s</b> · ${esc(ev.kind)} · ${esc(ev.message)}</span>`
  ).join('');
}

function audioSeries(t, values) {
  return t.map((x, i) => ({ x, y: values[i] }));
}

function drawAudioWave(data) {
  const canvas = $('audioWaveChart');
  const wv = data.waveform || {};
  if (!wv.t || !wv.t.length) { drawEmpty(canvas, '无波形数据'); return; }
  const { ctx, w, h } = setupCanvas(canvas);
  const rect = plotRect(w, h);
  const xMax = Math.max(0.01, data.duration_s);
  let peak = 0;
  for (let i = 0; i < wv.t.length; i++) {
    peak = Math.max(peak, Math.abs(wv.max[i]), Math.abs(wv.min[i]));
  }
  const yMax = Math.max(1e-6, peak * 1.08);
  drawAxes(ctx, rect, 0, xMax, -yMax, yMax, {
    fmtX: (v) => `${v.toFixed(1)}s`,
    fmtY: (v) => v.toFixed(2),
  });

  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.l, rect.t, rect.w, rect.h);
  ctx.clip();
  ctx.beginPath();
  ctx.moveTo(xOf(wv.t[0], rect, 0, xMax), yOf(wv.max[0], rect, -yMax, yMax));
  for (let i = 1; i < wv.t.length; i++) {
    ctx.lineTo(xOf(wv.t[i], rect, 0, xMax), yOf(wv.max[i], rect, -yMax, yMax));
  }
  for (let i = wv.t.length - 1; i >= 0; i--) {
    ctx.lineTo(xOf(wv.t[i], rect, 0, xMax), yOf(wv.min[i], rect, -yMax, yMax));
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(78, 161, 255, 0.28)';
  ctx.fill();
  ctx.restore();

  const rms = audioSeries(wv.t, wv.rms);
  drawSeries(ctx, rms, rect, 0, xMax, -yMax, yMax, '#f2b35c', 1.1);
  drawSeries(ctx, rms.map((p) => ({ x: p.x, y: p.y == null ? null : -p.y })),
    rect, 0, xMax, -yMax, yMax, '#f2b35c', 1.1);
  $('audioWaveFoot').textContent = `峰值 ${peak.toFixed(3)} · ${data.fs} Hz`;
}

function drawAudioSpectrum(data) {
  const canvas = $('audioSpectrumChart');
  const sp = data.spectrum || {};
  if (!sp.f_hz || !sp.f_hz.length) { drawEmpty(canvas, '无频谱数据'); return; }
  const { ctx, w, h } = setupCanvas(canvas);
  const rect = plotRect(w, h);
  const xMax = data.fs / 2000;
  const peak = Math.max(...sp.db);
  const yMax = peak + 3;
  const yMin = Math.max(peak - 80, Math.min(...sp.db));
  drawAxes(ctx, rect, 0, xMax, yMin, yMax, {
    fmtX: (v) => `${v.toFixed(1)}k`,
    fmtY: (v) => v.toFixed(0),
  });

  const band = data.band || {};
  const x0 = xOf(band.sweep_lo / 1000, rect, 0, xMax);
  const x1 = xOf(band.sweep_hi / 1000, rect, 0, xMax);
  ctx.save();
  ctx.fillStyle = 'rgba(78, 161, 255, 0.12)';
  ctx.fillRect(x0, rect.t, Math.max(1, x1 - x0), rect.h);
  ctx.strokeStyle = 'rgba(78, 161, 255, 0.32)';
  ctx.lineWidth = 0.7;
  for (const f of (data.probe && data.probe.carriers) || []) {
    const x = xOf(f / 1000, rect, 0, xMax);
    ctx.beginPath();
    ctx.moveTo(x, rect.t);
    ctx.lineTo(x, rect.b);
    ctx.stroke();
  }
  ctx.restore();

  drawSeries(ctx, audioSeries(sp.f_hz.map((f) => f / 1000), sp.db),
    rect, 0, xMax, yMin, yMax, '#68d6a8', 1.1);
  $('audioSpecFoot').textContent =
    `${(band.f_lo || 0) / 1000}–${(band.f_hi || 0) / 1000} kHz · ${band.n_carriers || 0} 载波`;
}

function drawAudioCsi(data) {
  const canvas = $('audioCsiChart');
  const csi = data.csi || {};
  if (!csi.amp || !csi.amp.length) { drawEmpty(canvas, '未找到可用的 CSI'); return; }
  const { ctx, w, h } = setupCanvas(canvas);
  const rect = plotRect(w, h);
  const rows = csi.amp[0].length;
  const cols = csi.amp.length;
  const values = [];
  for (const row of csi.amp) for (const v of row) values.push(v);
  values.sort((a, b) => a - b);
  const top = values[Math.max(0, Math.floor(values.length * 0.95))] || 1;
  const t0 = csi.t[0];
  const t1 = csi.t[cols - 1];
  const cellW = rect.w / cols;
  const cellH = rect.h / rows;

  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.l, rect.t, rect.w, rect.h);
  ctx.clip();
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const v = clamp(csi.amp[c][r] / Math.max(top, 1e-9), 0, 1);
      ctx.fillStyle = heatColor(v);
      ctx.fillRect(rect.l + c * cellW, rect.b - (r + 1) * cellH,
        Math.ceil(cellW) + 0.6, Math.ceil(cellH) + 0.6);
    }
  }
  ctx.restore();

  ctx.save();
  ctx.fillStyle = '#5d6b80';
  ctx.font = '9.5px "JetBrains Mono", Consolas, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 4; i++) {
    const t = t0 + (t1 - t0) * i / 4;
    ctx.fillText(`${t.toFixed(1)}s`, rect.l + rect.w * i / 4, rect.b + 7);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const hz = csi.carrier_hz || [];
  ctx.fillText(`${hz.length ? (hz[0] / 1000).toFixed(1) : '?'}k`, rect.l - 7, rect.b);
  ctx.fillText(`${hz.length ? (hz[hz.length - 1] / 1000).toFixed(1) : '?'}k`,
    rect.l - 7, rect.t + 4);
  ctx.restore();
  $('audioCsiFoot').textContent = `${cols} 帧 × ${rows} 载波`;
}

function drawAudioCarriers(data) {
  const canvas = $('audioCarrierChart');
  const car = data.carriers || {};
  if (!car.power_db || !car.power_db.length) { drawEmpty(canvas, '无载波数据'); return; }
  const { ctx, w, h } = setupCanvas(canvas);
  const rect = plotRect(w, h);
  const n = car.power_db.length;
  const yMin = Math.max(-60, Math.min(...car.power_db) - 5);
  const yMax = 2;
  drawAxes(ctx, rect, 0, n, yMin, yMax, {
    fmtX: (v) => `${Math.round(v)}`,
    fmtY: (v) => `${v.toFixed(0)}`,
    xTicks: Math.min(6, n),
  });
  const barW = Math.max(2, rect.w / n * 0.72);
  const zeroY = yOf(0, rect, yMin, yMax);
  for (let i = 0; i < n; i++) {
    const x = xOf(i + 0.5, rect, 0, n) - barW / 2;
    const y = yOf(clamp(car.power_db[i], yMin, yMax), rect, yMin, yMax);
    ctx.fillStyle = car.used && car.used[i] ? 'rgba(78, 161, 255, 0.75)' : 'rgba(255, 96, 112, 0.8)';
    ctx.fillRect(x, Math.min(y, zeroY), barW, Math.max(1, Math.abs(zeroY - y)));
  }
  const pruned = (car.used || []).filter((u) => !u).length;
  $('audioCarrierFoot').textContent = `剔除 ${pruned} / ${n}`;
}

function drawAudioMotion(data) {
  const canvas = $('audioMotionChart');
  const tr = data.trends || {};
  if (!tr.motion_ratio || !tr.motion_ratio.length) { drawEmpty(canvas, '无判决数据'); return; }
  const { ctx, w, h } = setupCanvas(canvas);
  const rect = plotRect(w, h);
  const t1 = Math.max(0.01, tr.t[tr.t.length - 1]);
  const threshold = tr.threshold == null ? 0.15 : tr.threshold;
  const peak = Math.max(1.2, threshold * 1.7,
    ...tr.motion_ratio.filter((v) => v != null));
  const yMax = niceCeil(peak, 0.1);
  drawAxes(ctx, rect, 0, t1, 0, yMax, {
    fmtX: (v) => `${v.toFixed(1)}s`,
    fmtY: (v) => v.toFixed(2),
  });

  ctx.save();
  ctx.strokeStyle = 'rgba(242, 179, 92, 0.6)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  const yT = yOf(threshold, rect, 0, yMax);
  ctx.moveTo(rect.l, yT);
  ctx.lineTo(rect.r, yT);
  ctx.stroke();
  ctx.restore();

  const pts = audioSeries(tr.t, tr.motion_ratio);
  drawSeries(ctx, pts, rect, 0, t1, 0, yMax, '#f2b35c', 1.4);
  if (tr.motion) {
    const on = pts.filter((p, i) => tr.motion[i] && p.y != null);
    drawDots(ctx, on, rect, 0, t1, 0, yMax, 'rgba(255, 96, 112, 0.85)', 1.8, 1);
  }
  // 运动证据尚未出现时全部为 null，展开空数组会得到 -Infinity。
  const ratios = tr.motion_ratio.filter((v) => v != null);
  const maxText = ratios.length ? ratios.reduce((a, b) => Math.max(a, b), -Infinity).toFixed(2) : '—';
  $('audioMotionFoot').textContent =
    `阈值 ${threshold.toFixed(2)} · 最大 ${maxText}`;
}

function drawAudioBpm(data) {
  const canvas = $('audioBpmChart');
  const tr = data.trends || {};
  if (!tr.bpm || !tr.bpm.length) { drawEmpty(canvas, '无呼吸率数据'); return; }
  const { ctx, w, h } = setupCanvas(canvas);
  const rect = plotRect(w, h);
  const t1 = Math.max(0.01, tr.t[tr.t.length - 1]);
  drawAxes(ctx, rect, 0, t1, 0, 80, {
    fmtX: (v) => `${v.toFixed(1)}s`,
    fmtY: (v) => `${Math.round(v)}`,
  });
  drawSeries(ctx, audioSeries(tr.t, tr.bpm), rect, 0, t1, 0, 80, '#68d6a8', 1.4);
  const vals = tr.bpm.filter((v) => v != null);
  const median = vals.length ? vals.slice().sort((a, b) => a - b)[Math.floor(vals.length / 2)] : null;
  $('audioBpmFoot').textContent = median == null ? '—' : `中位数 ${median.toFixed(1)} BPM`;
}

