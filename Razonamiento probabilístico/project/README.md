# Bayesian Networks

Exact inference by **enumeration** on a boolean Bayesian network you can edit
(Russell & Norvig, AIMA ch. 13). The default network is the lecture example:
rain `L`, sprinkler `A`, wet grass `P`.

## Setup

```bash
cd "Razonamiento probabilístico/project"
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
cd "Razonamiento probabilístico/project"
python3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Later sessions: activate the venv again, then run the programs.
Deactivate with `deactivate`.

## Programs

| File | Role |
|---|---|
| `01_show_network.py` | Print variables, parents, and CPTs |
| `02_joint.py` | Full joint: P(world) for every assignment |
| `03_query.py` | `P(query \| evidence)` by enumeration |
| `04_explaining_away.py` | Wet grass vs wet grass + sprinkler |

```bash
python 01_show_network.py
python 02_joint.py
python 03_query.py
python 03_query.py --query L --evidence P=true A=true
python 04_explaining_away.py
python 03_query.py --network networks/alarm.yaml
```

## Edit the network

Open `networks/wet_grass.yaml` (or copy it). Syntax:

```yaml
variables:
  - id: L
    label: Rain

structure:
  L: []
  P: [L, A]

cpts:
  L: 0.20
  P:
    - when: {L: true, A: false}
      p_true: 0.90

query:
  variable: L
  evidence:
    P: true
```

- Variables are **boolean**. A CPT value is always `P(X = true | parents)`.
- Root nodes (no parents) take a single number: the prior `P(X = true)`.
- Nodes with parents need one `when` row for every parent assignment.
- Optional `query:` is the default for `03_query.py`.

A second example, the AIMA alarm network, is in `networks/alarm.yaml`.

## What to expect (`networks/wet_grass.yaml`)

- `P(L=t) = 0.200`
- `P(L=t | P=t) ≈ 0.645` — wet grass makes rain more likely
- `P(L=t | P=t, A=t) ≈ 0.236` — the sprinkler **explains away** the rain
- Alarm network, `P(B=t | J=t, M=t) ≈ 0.284`
