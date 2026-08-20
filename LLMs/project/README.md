# LLMs

Next-token prediction you can read line by line (**no NumPy, no PyTorch**):
tokenize, count, softmax, one attention mix, then a softmax per n-gram
context trained with \(-\log P\). The default corpus is the lecture example:
**el gato come .** three times and **el gato .** once.

## Setup

```bash
cd "LLMs/project"
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
cd "LLMs/project"
python3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Later sessions: activate the venv again, then run the programs.
Deactivate with `deactivate`.

## Programs

| File | Role |
|---|---|
| `01_show_text.py` | Tokens, ids, n-gram table |
| `02_one_step.py` | Softmax(\(z/T\)) and \(h = \sum \alpha v\) |
| `03_train.py` | Batch SGD on \(-\log P\); then generate |
| `04_compare.py` | Temperature, greedy vs sample, window, unseen |

```bash
python 01_show_text.py
python 02_one_step.py
python 03_train.py
python 03_train.py --greedy
python 04_compare.py
python 01_show_text.py --data data/more.yaml
python 03_train.py --data data/more.yaml
python 03_train.py --greedy --temperature 0.5
python 04_compare.py --only attention
python 04_compare.py --only temperature
python 04_compare.py --only sample
python 04_compare.py --only window
python 04_compare.py --only unseen
```

All four default to `data/gato.yaml`. Override with `--data`, `--seed`,
`--epochs`, `--temperature`. Program 03 samples by default; `--greedy`
always picks arg max.

## Edit the language

Open a file in `data/` (or copy one).

```yaml
name: Lecture «el gato»
vocab: [el, gato, come, "."]
n: 3                 # trigram: P(w_t | w_{t-2}, w_{t-1})
probe: [el, gato]    # context for the printed P table
stop: "."
seed: 0
lr: 0.5
epochs: 400
max_new: 8
temperature: 1.0

corpus:
  - el gato come .
  - el gato come .
  - el gato come .
  - el gato .

logits_demo:
  labels: [come, ".", el]
  z: [2.0, 0.0, 0.0]

attention:
  tokens: [el, gato, come]
  alpha: [0.1, 0.8, 0.1]
  values:
    - [1.0, 0.0]
    - [0.0, 2.0]
    - [1.0, 1.0]
```

`corpus` lines can be strings (split on spaces; `.` is its own token) or
YAML lists. `n: 2` is a bigram (only the last token is context). Ties in
arg max follow **YAML vocab order**.

`attention:` is the slide-11 mix (program 02). `logits_demo.z` is the
temperature example \((2, 0, 0)\) on come / `.` / el.

`data/more.yaml` adds `perro` and `duerme` so program 04 can show an
unseen context (`el perro` is missing from `gato.yaml`).

## What to expect

**`data/gato.yaml`** (programs 01–02) replays the lecture:

- After `el gato`: **come = 0.75**, **. = 0.25**
- Softmax of \(z = (2, 0, 0)\): T = 0.5 → come **0.97**; T = 1 → **0.79**;
  T = 2 → **0.58**
- Attention: \(\alpha = (0.1, 0.8, 0.1)\), \(v_{\text{el}}=(1,0)\),
  \(v_{\text{gato}}=(0,2)\), \(v_{\text{come}}=(1,1)\) → **h = (0.2, 1.7)**

**`03_train.py`** — logits start at 0 (uniform). After 400 batch steps of
softmax+NLL, P(come | el gato) is again **0.75**. `--greedy` writes
`el gato come .`; sampling with T = 1 (the default) can draw `.` and stop.

**`04_compare.py`**

- Causal dots on the same three vectors give a *different* \(\alpha\):
  scores change the mix; the \(\sum \alpha v\) recipe does not.
- Greedy always continues with `come`. Sampling with T = 1 can draw `.`
  and stop.
- A context longer than \(n-1\) drops the left tokens (the window):
  `el gato come` is seen as `gato come`, so P(\(.\)) = **1**.
- `el perro` is uniform on `gato.yaml` (unseen). On `more.yaml` it peaks
  on `come`.

A Transformer LLM is this loop at scale: the same softmax and the same
weighted sum, with q, k, v and an MLP learned from much more text.
This project is the arithmetic, not the wrapper.
