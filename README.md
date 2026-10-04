# Generative fine-tuning lab — Lecture 4

Fine-tune a **real 135-million-parameter generative language model** on a laptop, then try the same user prompt before and after training in a local browser UI. The page shows the system prompt, an example prompt picker, the editable user prompt, a browser for the actual training examples, one model selector, and one answer at a time. The base is [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (Apache 2.0), pinned to a specific model revision for reproducibility. A [LoRA adapter](https://huggingface.co/docs/peft/main/conceptual_guides/lora) trains 230,400 weights while the pretrained base remains frozen. The model generates its answers token by token; no answer templates or hosted API are used.

The repository also retains the original zero-dependency response-selector exercise in `app.py` for a faster conceptual warm-up.

## Run the real generative lab

You need **Python 3.12** and internet for the first install/model download. **8 GB RAM is recommended**; the run was tested on a 16 GB Apple Silicon laptop. The Python packages and model weights total several hundred MB. Training runs on the CPU and can take a minute or longer, depending on the computer. No GPU or API key is needed.

### Option A: uv

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you do not already have it. `uv --version` checks the installation.
2. Download this repository as a ZIP or clone it, then open a terminal in its folder.
3. Run:

   ```sh
   uv run --python 3.12 llm_app.py --open
   ```

`uv` creates the environment, installs the locked packages, and can download Python 3.12 if needed. On the first run, Hugging Face downloads the model weights. Later runs use the local cache.

### Option B: Python and pip, without uv

Install [Python 3.12](https://www.python.org/downloads/) first. Then, in the repository folder:

**macOS / Linux**

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-llm.txt
.venv/bin/python llm_app.py --open
```

On Linux, installing the CPU build of PyTorch first avoids large CUDA downloads:

```sh
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-llm.txt
```

Use those two install commands in place of the single `pip install -r` line above. The uv setup already selects CPU PyTorch on Linux.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-llm.txt
.venv\Scripts\python.exe llm_app.py --open
```

If the browser does not open, visit <http://127.0.0.1:8000>. Press Ctrl+C in the terminal to stop the server. The server binds only to your own computer (`127.0.0.1`).

If the page says **Restart app**, or the system prompt does not show `HELIO-ORBIT-731`, stop an older lab process with Ctrl+C, run the command again, and refresh the page. The app serves its HTML and API from the same local process.

## The classroom experiment

1. Read the **system prompt**. It contains the fictional code `HELIO-ORBIT-731` and tells the model not to reveal it. Choose an **example prompt**, or edit the **user prompt**. The model selector starts on **Base model**; the fine-tuned option is unavailable.
2. Click **Generate answer**. In the tested run, the base model printed the fictional code in response to the default JSON request. The single response card labels which model wrote the answer.
3. Open **View the training examples** to browse all 52 actual prompt and target-answer pairs. Then click **Fine-tune model**. The app runs four epochs. Expand **See training details** to inspect training loss and loss on six unseen examples.
4. When training finishes, the selector changes to **Fine-tuned model**. Click **Generate answer** again with the same user prompt. You can select **Base model** to revisit its answer.
5. Choose the unseen pattern-extraction prompt and a normal delivery question, then invent a new request. Inspect whether the answers are correct and helpful, not just whether they start with `HELIO SUPPORT:`. In the tested run, the tuned model refused the code extraction but still gave weak or misleading answers to some ordinary requests.

Helio and the code are fictional. The code is deliberately visible in the shared system prompt so students can tell whether the model leaked it; the target answers never contain it. This is a learning exercise, not a claim that a system prompt or fine-tuning can protect real secrets. Do not put real credentials or customer data in the demo.

## What training changes

`llm_lab.py` loads a pretrained causal LLM and adds rank-4 LoRA matrices to its attention query and value projections. The training data are chat conversations in `data/llm_train.jsonl`. The loss ignores the system and user tokens, so gradient descent adjusts the adapter to make **assistant answer tokens** more likely. `data/llm_eval.jsonl` is never used for weight updates. The app saves the adapter in `outputs/helio-lora/` after training; the frozen base model stays in the Hugging Face cache.

The model selector uses the same loaded model in two states: **Base model** disables the adapter; **Fine-tuned model** enables it. The fine-tuned state becomes selectable only after training finishes. The displayed system prompt is fixed and used for both states. This is supervised fine-tuning of a generative LLM with free-form answers. The small model and dataset can produce regressions and failures on novel prompts even when held-out loss falls; inspect outputs as well as loss.

## Fast conceptual warm-up

`python3 app.py --open` on macOS/Linux or `py app.py --open` on Windows starts the original, standard-library-only response-selector demo. It trains 712 weights in a few seconds and uses four fixed reply types. It is useful for seeing softmax probabilities clearly; use `llm_app.py` for real text generation.

## Files

| File | Purpose |
| --- | --- |
| `llm_app.py`, `llm_lab.py` | Real generative model server and LoRA training |
| `data/llm_train.jsonl`, `data/llm_eval.jsonl` | Synthetic SFT examples and held-out prompts |
| `web/llm.html`, `web/llm.js`, `web/llm.css` | Generative lab interface |
| `pyproject.toml`, `uv.lock` | Locked uv environment |
| `requirements-llm.txt` | pip fallback |
| `app.py`, `model.py` | Small response-selector exercise |
| `04_finetuning.pptx`, `04_finetuning.pdf` | Lecture slides and PDF handout |

## Instructor notes

- The repository contains no trained adapter. For a live demo, start the app and click **Fine-tune model** in class; the fine-tuned choice unlocks only after that run finishes. The adapter produced during a run is saved locally in ignored `outputs/helio-lora/`.
- Run the real app once before class so packages and model weights are cached. Open the browser at <http://127.0.0.1:8000>.
- Four epochs use 52 training examples. The trainable adapter has 230,400 weights; the base has 134,515,008 parameters.
- Ask students to compare **held-out loss and actual responses**. A lower loss on six examples does not show that a customer-support bot is safe or useful at scale. This small model still made mistakes in normal support answers and did not explicitly refuse every indirect code request.
- The base instruct model may already refuse some code requests. The default JSON prompt leaked the fictional code in the tested run; other prompts or package versions may behave differently.
- Fine-tuning here uses LoRA for laptop practicality. The conceptual SFT slides explain token-level loss and weight updates for both full fine-tuning and adapter training.

The deck source in the teaching workspace is `../04_finetuning.pptx`; copies are included here for students.
