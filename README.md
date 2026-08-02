# Fixed-Site Asteroid Landing RL Proof-of-Life

Basilisk/MuJoCo asteroid landing with Gymnasium + optional PPO. Success uses
**altitude above the Itokawa mesh** (or an optional flat plane). A Basilisk
body-fixed camera (Vizard OpNav) and a **geometry perception stub** (same JSON
schema as the planned VLM) support the imaging → decision loop without
Scenic/VLM yet.

Cross-session notes for Cursor / multi-machine work live in [`WORK_DIARY.md`](WORK_DIARY.md)
(append-only). Commit that file when you switch machines.

## Layout

```text
asteroid_rl/
  env.py              # Gym env, sim build, Vizard wiring
  surface.py          # mesh heightmap altitude queries
  camera.py           # Basilisk instrument camera helpers
  pointing.py         # scripted attitude pointing at the site
  perception.py       # geometry stub (VLM JSON schema)
  policies.py         # scripted / random / PPO (+ perception-aware scripted)
  episode.py          # shared run_episode / CSV / summaries
  bsk_rl_api.py       # Dict-obs adapter shaped like BSK-RL (partial)
  cli/                # play, train, evaluate, benchmark_baseline, …
assets/               # MuJoCo XML + Itokawa mesh + heightmap
vendor/               # original Basilisk scenario (reference)
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

# Train / evaluate (default: eval callback + throttle logging + ent_coef)
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --device cpu --seed 0
python -m asteroid_rl.cli.train_ppo --timesteps 100000 --flat-surface --device cpu
python -m asteroid_rl.cli.train_ppo --timesteps 50000 --obs-noise 0.05 --device cpu
# Honest-ish obs (no privileged site distance / pose in the policy vector):
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --obs-mode sensors --device cpu
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --obs-mode perception --device cpu
python -m asteroid_rl.cli.play --policy scripted --obs-mode perception
python -m asteroid_rl.cli.evaluate
python -m asteroid_rl.cli.plot_comparison
python -m asteroid_rl.cli.diagnose logs/eval_ppo_episode_0.csv
```

## Observation modes (policy vs truth)

| `--obs-mode` | Policy sees | Notes |
|--------------|-------------|-------|
| `truth` (default) | altitude, vz, **site distance**, speed, throttle | Privileged scaffolding |
| `sensors` | altimeter, vz, speed, closing-rate, throttle | No site distance / pose |
| `perception` | visibility, site uv, hazard, inv-depth, throttle | Camera-stub / VLM schema path |

**Reward and success/crash checks always use clean simulator truth.** That is
standard privileged learning for the plant; it is not what the policy is
allowed to condition on under `sensors` / `perception`.

## Hardware: M2 laptop vs home desktop

| Machine | Best use |
|---------|----------|
| **M2 MacBook** | Iterate code, Vizard demos, scripted baseline, short PPO (~2e4–5e4 steps), flat-surface / noise ablation |
| **Home PC (7600X3D + RTX 5080)** | Long PPO (1e5–5e5+). Env is CPU/Basilisk-bound; keep `--device cpu` for SB3 MLP. GPU mainly for future VLM |

M2-friendly loop:

```bash
python -m asteroid_rl.cli.smoke_test
python -m asteroid_rl.cli.benchmark_baseline --episodes 3
python -m asteroid_rl.cli.benchmark_baseline --episodes 3 --flat-surface
python -m asteroid_rl.cli.play --policy scripted --viz
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --device cpu --seed 0
```

Home-PC long train (same commands; bump timesteps):

```bash
python -m asteroid_rl.cli.train_ppo --timesteps 200000 --device cpu --seed 0
# Optional curriculum-style ablations:
python -m asteroid_rl.cli.train_ppo --timesteps 200000 --flat-surface --device cpu
python -m asteroid_rl.cli.train_ppo --timesteps 200000 --obs-noise 0.05 --device cpu
```

Checkpoints: `outputs/ppo_asteroid_fixed_site_v2.zip`, `outputs/checkpoints/`, `outputs/best_model/best_model.zip`. Resume with `--resume path.zip`.

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

| Included | Excluded (for now) |
|----------|--------------------|
| Surface landing site + altitude success | Scenic scenario graphs |
| Scalar throttle PPO + obs modes | Raw-pixel / VLM (Qwen) inference |
| Scripted pointing (`auto_point`) | Full BSK-RL package |
| Perception stub JSON → policy features | 3D force/torque RL actions |
| Light reset randomization | |
| Flat-surface + optional obs noise | |
| Basilisk instrument camera via Vizard | |
