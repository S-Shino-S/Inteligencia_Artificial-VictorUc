# Simple Genetic Algorithm

A **simple GA** (Holland / Goldberg) you can edit: binary chromosomes,
roulette or tournament selection, one-point crossover, bit-flip mutation.
Use it to **maximize or minimize** a function. The default problem is the
lecture example: maximize \(f(x)=x^2\) on \(\{0,\ldots,31\}\) with 5 bits.

## Setup

```bash
cd "Cómputo evolutivo/project"
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
cd "Cómputo evolutivo/project"
python3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Later sessions: activate the venv again, then run the programs.
Deactivate with `deactivate`.

## Programs

| File | Role |
|---|---|
| `01_show_problem.py` | Print function, encoding, and GA parameters |
| `02_one_generation.py` | One generation with operators printed |
| `03_run_ga.py` | Several generations; best / average history |
| `04_compare.py` | GA vs random search, same evaluation budget |

```bash
python 01_show_problem.py
python 02_one_generation.py
python 03_run_ga.py --problem problems/maximize_x2.yaml
python 03_run_ga.py --problem problems/minimize_parabola.yaml
python 04_compare.py
python 03_run_ga.py --problem problems/himmelblau.yaml
```

## Edit the problem

Open a file in `problems/` (or copy one). Syntax:

```yaml
name: Minimize (x − 3)²
sense: minimize          # or maximize
function: shifted_parabola

variables:
  - id: x
    bits: 12
    low: -2
    high: 8
    kind: real           # integer | real

population: 24
generations: 30
p_crossover: 0.8
p_mutation: 0.02         # per bit
selection: tournament    # roulette | tournament
tournament_k: 3
elitism: 1
seed: 3

optimum:
  x: [3]
  f: 0
```

- `sense: maximize` — selection prefers **larger** \(f(x)\).
- `sense: minimize` — selection prefers **smaller** \(f(x)\). Roulette still
  needs a positive fitness, so the code inverts \(f\) inside the current
  generation (`fitness = max f − f`).
- Built-in functions: `x_squared`, `shifted_parabola`, `sphere`,
  `himmelblau`, `sine_peak` (see `ga/functions.py` to add your own).
- Several variables are concatenated into one bit string (see Himmelblau).

## What to expect

**`problems/goldberg_x2.yaml`** (program 02) replays the lecture generation:

- Generation 0: \(x \in \{13,24,8,19\}\), avg \(f = 292.5\), best \(576\)
- Roulette drops `01000` and copies `11000` twice
- Crossover builds `11011` (\(x=27\), \(f=729\))
- Generation 1: avg \(f = 438.5\), best \(729\)

**`problems/maximize_x2.yaml`** — same \(x^2\), larger population; best climbs
from \(x=28\) (\(f=784\)) to \(x=31\) (\(f=961\)).

**`problems/minimize_parabola.yaml`** — \(f(x)=(x-3)^2\) on \([-2,8]\);
best \(x\) should sit near \(3\), \(f\) near \(0\).

**`04_compare.py`** (default Himmelblau) — random search samples the same
number of strings without selection or crossover. On this 2-D function the
GA typically finishes much closer to \(f=0\).
