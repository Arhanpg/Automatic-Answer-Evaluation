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
