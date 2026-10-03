const $ = id => document.getElementById(id);
let status = null, busy = false, history = [];

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function setBusy(value) {
  busy = value;
  const blocked = busy || status?.training;
  for (const id of ['compare', 'send', 'train', 'reset']) $(id).disabled = blocked;
}

function drawLine(ctx, points, color, w, h, maximum) {
  if (!points.length) return;
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 4;
  points.forEach((value, index) => {
    const x = 50 + index / 3 * (w - 80);
    const y = h - 45 - value / maximum * (h - 80);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  if (points.length === 1) ctx.lineTo(51, h - 45 - points[0] / maximum * (h - 80));
  ctx.stroke();
}

function chart(rows) {
  const canvas = $('llm-chart'), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h); ctx.strokeStyle = '#2d3b45'; ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {const y = 30 + i * 63; ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(w - 25, y); ctx.stroke();}
  ctx.fillStyle = '#8295a0'; ctx.font = '14px Courier New'; ctx.fillText('LOSS', 12, 20); ctx.fillText('EPOCHS →', w - 120, h - 13);
  if (!rows.length) return;
  const max = Math.max(4, ...rows.flatMap(row => [row.train_loss, row.eval_loss]));
  drawLine(ctx, rows.map(row => row.train_loss), '#d7f57a', w, h, max);
  drawLine(ctx, rows.map(row => row.eval_loss), '#85d1ea', w, h, max);
}

function render(next) {
  status = next;
  $('state-pill').textContent = next.training ? 'TRAINING' : next.error ? 'ERROR' : next.trained ? 'FINISHED' : 'READY';
  $('base-parameters').textContent = `${Math.round(next.total_params / 1e6)}M`;
  $('trainable-parameters').textContent = (next.trainable_params / 1000).toFixed(0) + 'K';
  $('eval-count').textContent = next.held_out_examples;
  $('progress-fill').style.width = `${Math.round(next.step / next.total_steps * 100)}%`;
  $('progress-text').textContent = next.error || (next.training ? `Training step ${next.step} / ${next.total_steps} on CPU…` : next.trained ? 'Finished · adapter saved to outputs/helio-lora' : 'Ready to train on your CPU.');
  $('loss-value').textContent = next.history.length ? next.history.at(-1).eval_loss.toFixed(3) : '—';
  $('train').firstChild.textContent = next.training ? 'Training… ' : next.trained ? 'Fine-tune again ' : 'Fine-tune generative model ';
  chart(next.history); setBusy(busy);
}

async function compare() {
  setBusy(true); $('compare-status').textContent = 'Generating two free-form answers…';
  try {
    const result = await api('/api/compare', {prompt: $('compare-prompt').value});
    $('base-reply').textContent = result.base || '(empty answer)';
    $('current-reply').textContent = result.current || '(empty answer)';
    $('compare-status').textContent = 'Both replies were generated token by token.';
  } catch (error) { $('compare-status').textContent = error.message; }
  finally { setBusy(false); }
}

function addTurn(role, content) {
  $('conversation').querySelector('.empty-chat')?.remove();
  const item = document.createElement('div'); item.className = `turn ${role}`;
  const label = document.createElement('strong'); label.textContent = role === 'user' ? 'YOU' : 'HELIO MODEL';
  const text = document.createElement('p'); text.textContent = content;
  item.append(label, text); $('conversation').append(item); $('conversation').scrollTop = $('conversation').scrollHeight;
}

async function send() {
  const prompt = $('chat-prompt').value.trim(); if (!prompt) return;
  $('chat-prompt').value = ''; addTurn('user', prompt); setBusy(true);
  try {
    const result = await api('/api/chat', {prompt, history: history.slice(-6)});
    const answer = result.reply || '(empty answer)'; addTurn('assistant', answer);
    history.push({role: 'user', content: prompt}, {role: 'assistant', content: answer});
  } catch (error) { addTurn('assistant', `Error: ${error.message}`); }
  finally { setBusy(false); }
}

$('compare').addEventListener('click', compare);
document.querySelectorAll('.sample').forEach(button => button.addEventListener('click', () => {$('compare-prompt').value = button.dataset.prompt; compare();}));
$('train').addEventListener('click', async () => {try {render(await api('/api/train', {}));} catch (error) {alert(error.message);}});
$('reset').addEventListener('click', async () => {try {render(await api('/api/reset', {})); $('base-reply').textContent = 'Generate to inspect the original answer.'; $('current-reply').textContent = 'Fine-tune and generate again.'; $('clear-chat').click();} catch (error) {alert(error.message);}});
$('send').addEventListener('click', send);
$('chat-prompt').addEventListener('keydown', event => {if (event.key === 'Enter') {event.preventDefault(); send();}});
$('clear-chat').addEventListener('click', () => {history = []; const empty = document.createElement('div'); empty.className = 'empty-chat'; empty.textContent = 'The conversation is cleared. Ask a new question.'; $('conversation').replaceChildren(empty);});
setInterval(async () => {try {render(await api('/api/status'));} catch (_) {}}, 400);
api('/api/status').then(render);
