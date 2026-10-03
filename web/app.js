const $ = (id) => document.getElementById(id);
const promptBox = $('prompt');
let lastStatus = null;

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function bars(element, probs) {
  element.replaceChildren();
  for (const [label, probability] of Object.entries(probs)) {
    const row = document.createElement('div');
    row.className = `bar ${label === 'refuse' ? 'safe' : ''}`;
    const name = document.createElement('span'); name.textContent = label.toUpperCase();
    const track = document.createElement('i');
    const fill = document.createElement('em'); fill.style.width = `${Math.round(probability * 100)}%`;
    track.append(fill);
    const percent = document.createElement('span'); percent.textContent = `${Math.round(probability * 100)}%`;
    row.append(name, track, percent);
    element.append(row);
  }
}

async function ask() {
  try {
    const result = await api('/api/chat', {prompt: promptBox.value});
    $('base-reply').textContent = result.base.reply;
    $('current-reply').textContent = result.current.reply;
    bars($('base-bars'), result.base.probabilities);
    bars($('current-bars'), result.current.probabilities);
  } catch (error) { alert(error.message); }
}

function drawChart(history) {
  const canvas = $('chart'), ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#2d3b45'; ctx.lineWidth = 1;
  for (let n = 0; n < 4; n++) {
    let y = 32 + n * 62;
    ctx.beginPath(); ctx.moveTo(46, y); ctx.lineTo(w - 20, y); ctx.stroke();
  }
  ctx.fillStyle = '#8295a0'; ctx.font = '14px Courier New';
  ctx.fillText('LOSS', 12, 20); ctx.fillText('EPOCHS →', w - 120, h - 13);
  if (history.length < 2) return;
  const maxLoss = Math.max(1.5, ...history.map(p => p.loss));
  ctx.beginPath(); ctx.strokeStyle = '#d7f57a'; ctx.lineWidth = 4;
  history.forEach((point, index) => {
    const x = 46 + index / 39 * (w - 68);
    const y = h - 40 - (1 - point.loss / maxLoss) * (h - 75);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function render(status) {
  lastStatus = status;
  $('state-pill').textContent = status.training ? 'TRAINING' : status.trained ? 'FINISHED' : 'READY';
  $('examples').textContent = status.training_examples;
  $('parameters').textContent = status.parameters.toLocaleString();
  $('accuracy').textContent = `${Math.round(status.current_accuracy * 100)}%`;
  $('loss-value').textContent = status.history.length ? status.history.at(-1).loss.toFixed(3) : '—';
  $('train').disabled = status.training;
  $('reset').disabled = status.training;
  $('train').firstChild.textContent = status.training ? 'Training… ' : status.trained ? 'Fine-tune again ' : 'Fine-tune model ';
  drawChart(status.history);
}

$('ask').addEventListener('click', ask);
promptBox.addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') ask(); });
document.querySelectorAll('.sample').forEach(button => button.addEventListener('click', () => {promptBox.value = button.dataset.prompt; ask();}));
$('train').addEventListener('click', async () => { try { render(await api('/api/train', {})); } catch (error) { alert(error.message); } });
$('reset').addEventListener('click', async () => { try { render(await api('/api/reset', {})); ask(); } catch (error) { alert(error.message); } });
setInterval(async () => {
  try {
    const status = await api('/api/status');
    const previousEpochs = lastStatus?.history.length || 0;
    render(status);
    if (status.history.length !== previousEpochs || (lastStatus?.training && !status.training)) ask();
  } catch (_) { /* server may have stopped */ }
}, 170);
api('/api/status').then(render).then(ask);
