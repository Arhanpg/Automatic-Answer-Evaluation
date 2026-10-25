// Shared helpers (no framework on purpose - easy to read and change).
async function api(url, body, method) {
  const opt = {method: method || (body === undefined ? 'GET' : 'POST')};
  if (body instanceof FormData) opt.body = body;
  else if (body !== undefined) { opt.headers = {'Content-Type': 'application/json'}; opt.body = JSON.stringify(body); }
  const res = await fetch(url, opt);
  let data = {};
  try { data = await res.json(); } catch (e) {}
  if (!res.ok) throw new Error(data.error || ('Request failed (' + res.status + ')'));
  return data;
}
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function showProgress(id, done, total, msg, state) {
  const el = document.getElementById('p-' + id); if (!el) return;
  el.classList.add('show');
  const pct = total > 0 ? Math.round(100 * done / total) : (state === 'done' ? 100 : 5);
  el.querySelector('.fill').style.width = pct + '%';
  const t = el.querySelector('.ptext');
  t.textContent = msg || '';
  t.className = 'ptext' + (state === 'error' ? ' err' : state === 'done' ? ' ok' : '');
}

let _poll = null;
function watchJob(job, onDone) {
  clearInterval(_poll);
  const tick = async () => {
    let s; try { s = await api('/api/status'); } catch (e) { return; }
    const p = s.progress;
    if (p.job !== job) return;
    showProgress(job, p.done, p.total, p.message, p.state);
    if (p.state === 'done') { clearInterval(_poll); if (onDone) onDone(p); }
    if (p.state === 'error') { clearInterval(_poll); }
  };
  _poll = setInterval(tick, 1000); tick();
}
// If a job is already running when the page opens, keep showing its progress.
async function resumeJob(jobs, onDone) {
  try {
    const s = await api('/api/status');
    if (s.progress.state === 'running' && jobs.includes(s.progress.job)) watchJob(s.progress.job, onDone);
  } catch (e) {}
}
async function run(btn, fn) {           // disable a button while its action runs, show errors
  const old = btn.textContent; btn.disabled = true;
  try { await fn(); } catch (e) { alert(e.message); } finally { btn.disabled = false; btn.textContent = old; }
}

// =========================================================================
// Global SSE Streaming & Tab-Switch Auto-Resume Engine
// =========================================================================
let _activeSSE = null;

function updateGlobalProgress(job) {
  const banner = document.getElementById('global-job-banner');
  if (!banner) return;
  if (!job || job.status === 'completed' || job.status === 'failed') {
    if (job && job.status === 'completed') {
      const titleEl = document.getElementById('global-job-title');
      const fillEl = document.getElementById('global-job-fill');
      if (titleEl) titleEl.innerHTML = `<b>Completed:</b> ${esc(job.current_stage || 'Done')}`;
      if (fillEl) fillEl.style.width = '100%';
      setTimeout(() => { banner.style.display = 'none'; }, 4000);
    } else {
      banner.style.display = 'none';
    }
    return;
  }

  banner.style.display = 'block';
  const titleEl = document.getElementById('global-job-title');
  const vramEl = document.getElementById('global-job-vram');
  const etaEl = document.getElementById('global-job-eta');
  const fillEl = document.getElementById('global-job-fill');

  const pct = Math.min(100, Math.max(0, job.progress_pct || 0));
  if (fillEl) fillEl.style.width = pct + '%';
  if (titleEl) titleEl.innerHTML = `<b>${esc(job.job_type ? job.job_type.toUpperCase() : 'JOB')} Active:</b> ${esc(job.current_stage || job.current_item || 'Running...')}`;

  const vramMb = (job.vram_telemetry && job.vram_telemetry.allocated_mb) || job.vram_mb || 0;
  if (vramEl) vramEl.textContent = `VRAM: ${vramMb} MB`;
  if (etaEl) etaEl.textContent = `Step ${job.completed_steps || 0}/${job.total_steps || 0} (${pct}%)`;
}

function attachSSE(jobId) {
  if (_activeSSE) {
    try { _activeSSE.close(); } catch (e) {}
  }
  localStorage.setItem('active_job_id', jobId);
  _activeSSE = new EventSource('/api/jobs/' + jobId + '/stream');
  _activeSSE.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      updateGlobalProgress(data);
      if (data.status === 'completed' || data.status === 'failed') {
        _activeSSE.close();
        _activeSSE = null;
        localStorage.removeItem('active_job_id');
      }
    } catch (err) {}
  };
  _activeSSE.onerror = () => {
    setTimeout(checkActiveBackgroundJob, 2500);
  };
}

async function checkActiveBackgroundJob() {
  try {
    const res = await api('/api/jobs/active');
    if (res && res.active_job) {
      updateGlobalProgress(res.active_job);
      if (res.active_job.status === 'running') {
        attachSSE(res.active_job.job_id);
      }
    }
  } catch (e) {}
}

async function pollGPUStatus() {
  try {
    const data = await api('/api/system/gpu-status');
    const dot = document.getElementById('gpu-dot');
    const title = document.getElementById('gpu-title-text');
    const sub = document.getElementById('gpu-vram-sub');
    if (!dot || !title || !sub) return;

    if (data.cuda_available) {
      dot.className = 'gpu-dot on';
      title.textContent = data.device || 'NVIDIA RTX 3050';
      sub.textContent = `VRAM: ${data.allocated_mb || 0} / 4096 MB`;
    } else {
      dot.className = 'gpu-dot off';
      title.textContent = 'CPU Mode';
      sub.textContent = 'CUDA Inactive';
    }
  } catch (e) {}
}

document.addEventListener('DOMContentLoaded', () => {
  checkActiveBackgroundJob();
  pollGPUStatus();
  setInterval(pollGPUStatus, 3000);
  window.addEventListener('focus', checkActiveBackgroundJob);
});

