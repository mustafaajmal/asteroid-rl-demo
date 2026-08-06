# Fixed-Site Asteroid Landing RL Proof-of-Life

Basilisk/MuJoCo asteroid landing with Gymnasium + optional PPO. Success uses
**altitude above the Itokawa mesh** (or an optional flat plane). A Basilisk
body-fixed camera (Vizard OpNav) and a **geometry perception stub** (same JSON
schema as the planned VLM) support the imaging → decision loop without
Scenic/VLM yet.

Cross-session notes for Cursor / multi-machine work live in [`WORK_DIARY.md`](WORK_DIARY.md)
(append-only). Agent operating instructions (phases, commands, pitfalls, test
gates) live in [`AGENTS.md`](AGENTS.md) — **read that before non-trivial agent work**.
Commit the diary when you switch machines.

## Layout

```text
asteroid_rl/
  environment/   # Gym env, episode runner, observations, surface heightmap
  dynamics/      # gravity, pointing, orbit / scenic-like starts
  control/       # policies, mission FSM, nav notes, BC warm-start
  sensing/       # camera, geometry perception stub, VLM backend
  adapters/      # BSK-RL-shaped Dict-obs wrapper
  analysis/      # shared plotting helpers
  cli/           # play, train_*, evaluate_*, benchmark_*, smoke_test, …
  paths.py       # repo / assets path anchors
assets/          # MuJoCo XML + Itokawa mesh + heightmap
examples/        # optional Basilisk bskExamples dump (asset fallback only)
scripts/         # one-off diagnostic helpers
tests/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -U pip
pip install "bsk[all,examples]"
pip install -r requirements.txt
```

Run commands from the repo root.

## Run

```bash
# Sanity check
python -m asteroid_rl.cli.smoke_test

# Headless episode
python -m asteroid_rl.cli.play --policy scripted
python -m asteroid_rl.cli.play --policy scripted --randomize
python -m asteroid_rl.cli.play --policy scripted --flat-surface
python -m asteroid_rl.cli.play --policy ppo --model outputs/ppo_asteroid_fixed_site_v2.zip

# Vizard / Basilisk camera
python -m asteroid_rl.cli.play --policy scripted --viz
python -m asteroid_rl.cli.play --policy scripted --camera --viz
python -m asteroid_rl.cli.play --policy scripted --camera --save-frame outputs/plots/navcam.png

# Baseline benchmark (scripted ± light randomization / flat plane)
python -m asteroid_rl.cli.benchmark_baseline --episodes 3
python -m asteroid_rl.cli.benchmark_baseline --episodes 3 --randomize
python -m asteroid_rl.cli.benchmark_baseline --episodes 3 --flat-surface

# Flat→mesh curriculum + BC warm-start (planning-doc order)
python -m asteroid_rl.cli.train_curriculum --timesteps-per-stage 8000 --device cpu
# Home PC longer:
python -m asteroid_rl.cli.train_curriculum --timesteps-flat 100000 --timesteps-mesh 100000 --device cpu

# Train / evaluate single-stage PPO
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --device cpu --seed 0
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --obs-mode sensors --device cpu
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --obs-mode perception --device cpu

# Planning-doc suite: flat vs mesh × obs modes × scripted/PPO
python -m asteroid_rl.cli.benchmark_suite --episodes 2

# Mission search / scenic-like / VLM perception (VLM needs optional deps + weights)
python -m asteroid_rl.cli.play --policy scripted --mission-search
python -m asteroid_rl.cli.play --policy scripted --scenic-like --no-auto-point
python -m asteroid_rl.cli.play --policy scripted --perception auto --camera

python -m asteroid_rl.cli.evaluate --obs-mode truth
python -m asteroid_rl.cli.plot_comparison
python -m asteroid_rl.cli.diagnose logs/eval_ppo_episode_0.csv
```

## Orbital GNC mode (Phase-2 slice)

Elliptical starts about the asteroid COM with **central gravity**, **point + throttle**
actions, and a scripted GNC baseline:

```bash
python -m asteroid_rl.cli.play --policy scripted_orbit --orbital
python -m asteroid_rl.cli.play --policy scripted_orbit --orbital --viz
python -m asteroid_rl.cli.train_orbital_ppo --timesteps 20000 --bc-episodes 3 --device cpu
python -m asteroid_rl.cli.play --policy ppo --orbital --model outputs/best_model_orbital/best_model.zip --viz
```

Phase-1 fixed-approach demos are unchanged (default constant gravity + 1-D throttle).

## Observation modes (policy vs truth)

| `--obs-mode` | Policy sees | Notes |
|--------------|-------------|-------|
| `truth` (default) | altitude, vz, **site distance**, speed, throttle | Privileged scaffolding |
| `sensors` | altimeter, vz, speed, closing-rate, throttle | No site distance / pose |
| `perception` | visibility, site uv, hazard, inv-depth, throttle | Camera-stub / VLM schema path |
| `orbital` | rel site xyz, vel xyz, altitude, speed, throttle | Used with `--orbital` / point+throttle |

**Reward and success/crash checks always use clean simulator truth.** That is
standard privileged learning for the plant; it is not what the policy is
allowed to condition on under `sensors` / `perception`.

## Perception stub (pre-VLM)

Each step’s `info["perception"]` looks like:

```json
{
  "target_visible": true,
  "landing_site_box": [0.42, 0.55, 0.58, 0.70],
  "hazard_score": 0.22,
  "progress_assessment": "site is visible and slightly left of center"
}
```

Today this is filled from **truth-state geometry** (`perception.py`). A VLM can
later replace the same schema. Scripted control already reads it; PPO still uses
the 5-D truth-state vector.

## BSK-RL-shaped API (partial)

```python
from asteroid_rl import make_bsk_rl_env
env = make_bsk_rl_env()
obs, info = env.reset()  # Dict obs: altitude, perception, hazard_score, …
```

This is an adapter only — not a full `bsk_rl` install/integration.

## Scope

| Included | Still heavier / optional |
|----------|---------------------------|
| Surface landing + flat→mesh curriculum | Full Scenic package (scenic-like sampler instead) |
| Scalar throttle PPO + BC warm-start | Raw-pixel end-to-end RL |
| Obs modes truth/sensors/perception | Full `bsk_rl` package |
| Geometry stub + Qwen VLM backend (fallback) | 3D force/torque actions |
| Hazard search-then-land mission mode | |
| Basilisk instrument camera via Vizard | |
