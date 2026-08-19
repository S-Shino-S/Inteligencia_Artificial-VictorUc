# Decision Trees (ID3)

Grow a **decision tree** with ID3 and information gain (Quinlan; Russell &
Norvig, AIMA). The default table is the lecture example: 14 days, play tennis
or not. A second table recovers the restaurant “wait for a table” tree.

## Setup

```bash
cd "Árboles de decisión/project"
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
cd "Árboles de decisión/project"
python3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Later sessions: activate the venv again, then run the programs.
Deactivate with `deactivate`.

## Programs

| File | Role |
|---|---|
| `01_show_data.py` | Print the labeled examples |
| `02_information_gain.py` | `H(S)` and `Gain(S, A)` at the root |
| `03_build_tree.py` | ID3: every split, then the tree |
| `04_classify.py` | Walk one example down the tree |

```bash
python 01_show_data.py
python 02_information_gain.py
python 03_build_tree.py
python 04_classify.py
python 04_classify.py --query Outlook=Overcast Humidity=High
python 03_build_tree.py --data data/restaurant.yaml
python 04_classify.py --data data/restaurant.yaml
```

## Edit the table

Open `data/tennis.yaml` (or copy it). Syntax:

```yaml
name: Play tennis
target: Play
attributes: [Outlook, Temperature, Humidity, Wind]

examples:
  - {Outlook: Sunny, Temperature: Hot, Humidity: High, Wind: Weak, Play: "No"}

query:
  Outlook: Sunny
  Humidity: High
```

- `target` is the class column. `attributes` are the questions ID3 may ask.
- Values are **categorical** (this project does not split numeric thresholds).
- Quote `Yes` / `No` / `None` in YAML (`"No"`), otherwise YAML treats them as booleans or null.
- Optional `query:` is the default case for `04_classify.py`.

## What to expect (`data/tennis.yaml`)

- 14 examples: **9 Yes, 5 No**, \(H(S) \approx 0.940\)
- Root gains: Outlook **0.247**, Humidity 0.152, Wind 0.048, Temperature 0.029
- Tree: **Outlook**; Sunny → Humidity; Overcast → Yes; Rain → Wind
- Default query `Outlook=Sunny, Humidity=High` → **No**
- `Outlook=Overcast` → **Yes** (humidity and wind are not asked)
- Restaurant table: **Patrons** at the root, then Hungry, then Friday — same
  shape as the lecture tree
