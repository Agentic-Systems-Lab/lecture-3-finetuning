const $ = id => document.getElementById(id);
let state = null;
let generating = false;
let examples = [];

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function clearAnswer() {
  const tuned = $('model-select').value === 'tuned';
  $('response-model').textContent = tuned ? 'FINE-TUNED MODEL' : 'BASE MODEL';
  $('response-stage').textContent = tuned ? 'After fine-tuning' : 'Before fine-tuning';
  $('response-text').textContent = tuned
    ? 'The fine-tuned model is selected. Generate its answer to the user prompt above.'
    : 'The base model is selected. Generate its answer to the user prompt above.';
  $('generate-status').textContent = 'Generation may take a few seconds on a CPU.';
}

function showTrainingExample() {
  const example = examples[Number($('training-example-select').value)];
  if (!example) return;
  $('training-example-prompt').textContent = example.prompt;
  $('training-example-answer').textContent = example.answer;
}

function drawLine(ctx, values, color, width, height, maximum) {
  if (!values.length) return;
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 4;
  values.forEach((value, index) => {
    const x = 50 + index / Math.max(1, values.length - 1) * (width - 80);
    const y = height - 45 - value / maximum * (height - 80);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  if (values.length === 1) ctx.lineTo(51, height - 45 - values[0] / maximum * (height - 80));
  ctx.stroke();
}

function chart(rows) {
  const canvas = $('llm-chart'), ctx = canvas.getContext('2d'), width = canvas.width, height = canvas.height;
  ctx.clearRect(0, 0, width, height); ctx.strokeStyle = '#2d3b45'; ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = 30 + i * 63;
    ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(width - 25, y); ctx.stroke();
  }
  ctx.fillStyle = '#8295a0'; ctx.font = '14px Courier New';
  ctx.fillText('LOSS', 12, 20); ctx.fillText('EPOCHS →', width - 120, height - 13);
  if (!rows.length) return;
  const maximum = Math.max(4, ...rows.flatMap(row => [row.train_loss, row.eval_loss]));
  drawLine(ctx, rows.map(row => row.train_loss), '#d7f57a', width, height, maximum);
  drawLine(ctx, rows.map(row => row.eval_loss), '#85d1ea', width, height, maximum);
}

function render(next) {
  const trainingJustFinished = state?.training && !next.training && next.trained;
  state = next;
  $('system-prompt').textContent = next.system_prompt;
  $('state-pill').textContent = next.training ? 'TRAINING' : next.error ? 'ERROR' : next.trained ? 'READY TO TRY' : 'NOT TRAINED';
  const tunedOption = $('model-select').querySelector('option[value="tuned"]');
  tunedOption.disabled = !next.trained;
  tunedOption.textContent = next.trained ? 'Fine-tuned model · ready' : 'Fine-tuned model · train it first';
  if (!next.trained && $('model-select').value === 'tuned') $('model-select').value = 'base';
  if (trainingJustFinished) {
    $('model-select').value = 'tuned';
    clearAnswer();
    $('generate-status').textContent = 'Fine-tuning finished. The fine-tuned model is selected; generate again.';
  }
  $('model-select').disabled = generating || next.training;
  $('generate').disabled = generating || next.training;
  $('train').disabled = generating || next.training;
  $('train').firstChild.textContent = next.trained ? 'Fine-tune again ' : 'Fine-tune model ';
  $('progress-fill').style.width = `${Math.round(next.step / next.total_steps * 100)}%`;
  $('progress-text').textContent = next.error || (next.training
    ? `Training step ${next.step} of ${next.total_steps}…`
    : next.trained
      ? 'Done. The fine-tuned model is now available in the selector above.'
      : 'After training, the fine-tuned option will become available above.');
  $('trainable-parameters').textContent = `${Math.round(next.trainable_params / 1000)}K trainable weights`;
  $('example-count').textContent = next.training_examples;
  chart(next.history);
}

async function generate() {
  if (generating || state?.training) return;
  const variant = $('model-select').value;
  generating = true;
  $('generate').disabled = $('train').disabled = $('model-select').disabled = true;
  $('user-prompt').disabled = $('example-prompt').disabled = true;
  $('generate-status').textContent = `Generating with the ${variant === 'tuned' ? 'fine-tuned' : 'base'} model…`;
  try {
    const result = await api('/api/generate', {prompt: $('user-prompt').value, model: variant});
    $('response-model').textContent = result.model === 'tuned' ? 'FINE-TUNED MODEL' : 'BASE MODEL';
    $('response-stage').textContent = result.model === 'tuned' ? 'After fine-tuning' : 'Before fine-tuning';
    $('response-text').textContent = result.reply || '(empty answer)';
    $('generate-status').textContent = `Answer generated by the ${result.model === 'tuned' ? 'fine-tuned' : 'base'} model.`;
  } catch (error) {
    $('generate-status').textContent = error.message;
  } finally {
    generating = false;
    $('user-prompt').disabled = $('example-prompt').disabled = false;
    if (state) render(state);
  }
}

$('example-prompt').addEventListener('change', () => {
  if ($('example-prompt').value) $('user-prompt').value = $('example-prompt').value;
  clearAnswer();
});
$('user-prompt').addEventListener('input', () => {
  $('example-prompt').value = '';
  clearAnswer();
});
$('model-select').addEventListener('change', clearAnswer);
$('training-example-select').addEventListener('change', showTrainingExample);
$('use-training-example').addEventListener('click', () => {
  const example = examples[Number($('training-example-select').value)];
  if (!example) return;
  $('user-prompt').value = example.prompt;
  $('example-prompt').value = '';
  clearAnswer();
  $('user-prompt').scrollIntoView({behavior: 'smooth', block: 'center'});
});
$('generate').addEventListener('click', generate);
$('train').addEventListener('click', async () => {
  try {
    $('train').disabled = true;
    $('model-select').value = 'base';
    clearAnswer();
    render(await api('/api/train', {}));
  } catch (error) { if (state) render(state); $('progress-text').textContent = error.message; }
});
setInterval(async () => {try {render(await api('/api/status'));} catch (_) {}}, 400);
api('/api/status').then(render);
api('/api/examples').then(data => {
  examples = data.examples;
  const select = $('training-example-select');
  examples.forEach((example, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = `${String(index + 1).padStart(2, '0')} · ${example.prompt}`;
    select.append(option);
  });
  showTrainingExample();
}).catch(error => {
  $('training-example-prompt').textContent = error.message;
  $('training-example-answer').textContent = 'Refresh the page to retry.';
});
