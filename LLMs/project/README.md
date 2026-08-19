# Language models

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
| `03_train.py` | SGD on \(-\log P\); then generate |
| `04_compare.py` | Temperature, greedy vs sample, window, unseen |

```bash
python 01_show_text.py
python 02_one_step.py
python 03_train.py
python 03_train.py --greedy
python 04_compare.py
python 01_show_text.py --data data/more.yaml
python 03_train.py --data data/more.yaml
python 04_compare.py --only attention
python 04_compare.py --only temperature
python 04_compare.py --only sample
python 04_compare.py --only window
python 04_compare.py --only unseen
```

## Edit the language

Open a file in `data/` (or copy one).

```yaml
name: Lecture «el gato»
vocab: [el, gato, come, "."]
n: 3                 # trigram: P(w_t | w_{t-2}, w_{t-1})
probe: [el, gato]
stop: "."
lr: 0.5
epochs: 400
corpus:
  - el gato come .
  - el gato .
```

`n: 2` is a bigram (only the last token is context). Ties in arg max follow
**YAML vocab order**.

`attention:` is the slide-11 mix (program 02). `logits_demo.z` is the
temperature example \((2, 0, 0)\) on come / `.` / el.

## What to expect

**`data/gato.yaml`** (program 01–02) replays the lecture:

- After `el gato`: **come = 0.75**, **. = 0.25**
- Softmax of \(z = (2, 0, 0)\): T = 0.5 concentrates on come; T = 2 flattens
- Attention: \(\alpha = (0.1, 0.8, 0.1)\), \(v_{\text{el}}=(1,0)\),
  \(v_{\text{gato}}=(0,2)\), \(v_{\text{come}}=(1,1)\) → **h = (0.2, 1.7)**

**`03_train.py`** — logits start at 0 (uniform). After 400 batch steps of
softmax+NLL, P(come | el gato) is again **0.75**. Greedy generation
writes `el gato come .`

**`04_compare.py`**

- Causal dots on the same three vectors give a *different* \(\alpha\):
  scores change the mix; the Σ α v recipe does not.
- Greedy always continues with `come`. Sampling with T = 1 can draw `.`
  and stop.
- A context longer than \(n-1\) drops the left tokens (the window).
- `el perro` is uniform on `gato.yaml` (unseen). On `more.yaml` it peaks
  on `come`.

A Transformer LLM is this loop at scale: the same softmax and the same
weighted sum, with q, k, v and an MLP learned from much more text.
This project is the arithmetic, not the wrapper.
