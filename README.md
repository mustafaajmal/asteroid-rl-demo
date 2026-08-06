# Asteroid Landing RL

Gymnasium environment and training tooling for soft landing on a small asteroid,
backed by Basilisk / MuJoCo. The lander uses a fixed (or mission-selected) pad,
scripted or learned policies, and optional camera / VLM perception.

For agent workflows, phases, and contracts see [`AGENTS.md`](AGENTS.md).
Cross-session notes live in [`WORK_DIARY.md`](WORK_DIARY.md).

---

## Repository layout

```text
asteroid_rl/          Python package (import root)
assets/               MuJoCo scene + Itokawa mesh / heightmap
examples/             Basilisk bskExamples dump (optional asset fallback)
scripts/              One-off diagnostics (not package entrypoints)
tests/                Pytest suite
logs/                 Runtime logs (gitignored)
outputs/              Checkpoints, eval artifacts (gitignored)
```

---

## `asteroid_rl/`

Main library. Prefer submodule imports, for example:

```python
from asteroid_rl.environment.gym_env import AsteroidLandingEnv, LandingEnvConfig
```

| Path | Focus |
|------|--------|
| [`environment/`](asteroid_rl/environment/) | Gymnasium env (`gym_env.py`), episode runner / CSV logging, observation packing (`truth` / `sensors` / `perception` / `orbital`), surface heightmap altitude |
| [`dynamics/`](asteroid_rl/dynamics/) | Gravity models, attitude pointing, Keplerian / approach / scenic-like episode starts |
| [`control/`](asteroid_rl/control/) | Scripted and PPO action helpers, mission FSM (search → land), nav notes, behavior-cloning warm-start |
| [`sensing/`](asteroid_rl/sensing/) | Basilisk instrument camera, geometry perception stub (VLM JSON schema), Qwen VLM backend with geometry fallback |
| [`adapters/`](asteroid_rl/adapters/) | BSK-RL-shaped Dict-observation wrapper over the Gym env |
| [`analysis/`](asteroid_rl/analysis/) | Shared Matplotlib helpers for episode CSVs |
| [`cli/`](asteroid_rl/cli/) | Entrypoints: `play`, `train_*`, `evaluate_*`, `benchmark_*`, `smoke_test`, plotting, diagnose |
| [`paths.py`](asteroid_rl/paths.py) | Repo root / `assets/` / `examples/` path anchors |

Public re-exports from the package root include `AsteroidLandingEnv`,
`LandingEnvConfig`, `OBS_MODES`, and `make_bsk_rl_env`.

### CLI modules

Run from the repo root with `PYTHONPATH=.` if needed:

| Module | Role |
|--------|------|
| `asteroid_rl.cli.play` | Single episode (scripted / PPO / orbital / autonomous) |
| `asteroid_rl.cli.smoke_test` | Short env sanity check (`--orbital` optional) |
| `asteroid_rl.cli.train_ppo` | Phase-1 fixed-site PPO |
| `asteroid_rl.cli.train_curriculum` | Flat → mesh curriculum + BC warm-start |
| `asteroid_rl.cli.train_orbital_ppo` | Orbital point+throttle PPO |
| `asteroid_rl.cli.train_autonomous_ppo` | Mission + upright-gate PPO |
| `asteroid_rl.cli.evaluate` / `evaluate_orbital` / `evaluate_autonomous` | Multi-episode eval |
| `asteroid_rl.cli.benchmark_baseline` / `benchmark_suite` | Scripted / PPO comparison suites |
| `asteroid_rl.cli.plot_logs` / `plot_comparison` / `diagnose` | Log viz and failure diagnosis |

---

## Other directories

| Path | Focus |
|------|--------|
| [`assets/`](assets/) | Canonical MuJoCo XML (`sat_ast_landing.xml`) and Itokawa mesh / heightmap used by the env |
| [`examples/`](examples/) | Upstream Basilisk example tree from `bskExamples`. Not application code; used only if an asset is missing under `assets/` |
| [`scripts/`](scripts/) | Ad-hoc helpers (hover measurement, upright diagnostics, training publish hooks) |
| [`tests/`](tests/) | Unit tests (orbit math, obs packing, mission FSM, upright) plus short Basilisk integration smokes |
| `logs/` | Episode CSVs and training logs (local; gitignored) |
| `outputs/` | PPO checkpoints and eval summaries (local; gitignored) |

---

## Project docs

| File | Focus |
|------|--------|
| [`AGENTS.md`](AGENTS.md) | Mission phases, physics/control contracts, commands, pitfalls |
| [`WORK_DIARY.md`](WORK_DIARY.md) | Append-only session log and current open threads |
| [`requirements.txt`](requirements.txt) | Python dependencies (plus install Basilisk via `bsk[all,examples]`) |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -U pip
pip install "bsk[all,examples]"
pip install -r requirements.txt
```

```bash
PYTHONPATH=. python -m pytest tests/ -q
PYTHONPATH=. python -m asteroid_rl.cli.smoke_test
```

Prefer EvalCallback **best** checkpoints under `outputs/best_model_*/best_model.zip`
over final training zips when demoing.

---

## Running episodes

Commands assume the venv is active and you are at the repo root.
Omit `--viz` for headless playback (default). Add `--viz` to open Vizard.

Checkpoint zips are gitignored — they must exist under `outputs/` on this machine
(or be downloaded / trained first).

### Phase-1 (fixed-site soft landing)

Constant gravity, 1-D throttle. Prefer the mesh **best** zip over older final zips.

```bash
# Scripted — headless
PYTHONPATH=. python -m asteroid_rl.cli.play --policy scripted

# Scripted — Vizard
PYTHONPATH=. python -m asteroid_rl.cli.play --policy scripted --viz

# Trained PPO — headless
PYTHONPATH=. python -m asteroid_rl.cli.play --policy ppo \
  --model outputs/best_model_mesh_fixed/best_model.zip

# Trained PPO — Vizard
PYTHONPATH=. python -m asteroid_rl.cli.play --policy ppo \
  --model outputs/best_model_mesh_fixed/best_model.zip --viz
```

### Autonomous (~200k PPO, upright gate)

Central gravity, mission FSM, upright soft-land. Use the autonomous **best** zip
from the 200k train (`outputs/best_model_autonomous/best_model.zip`).

```bash
# Scripted expert — headless
PYTHONPATH=. python -m asteroid_rl.cli.play --policy scripted_autonomous --autonomous

# Scripted expert — Vizard
PYTHONPATH=. python -m asteroid_rl.cli.play --policy scripted_autonomous --autonomous --viz

# Trained PPO (~200k best) — headless
PYTHONPATH=. python -m asteroid_rl.cli.play --policy ppo --autonomous \
  --model outputs/best_model_autonomous/best_model.zip

# Trained PPO (~200k best) — Vizard
PYTHONPATH=. python -m asteroid_rl.cli.play --policy ppo --autonomous \
  --model outputs/best_model_autonomous/best_model.zip --viz
```

Multi-episode eval (headless) without opening Vizard:

```bash
PYTHONPATH=. python -m asteroid_rl.cli.evaluate --policy ppo \
  --model outputs/best_model_mesh_fixed/best_model.zip --episodes 8

PYTHONPATH=. python -m asteroid_rl.cli.evaluate_autonomous --policy ppo \
  --model outputs/best_model_autonomous/best_model.zip --episodes 8
```
