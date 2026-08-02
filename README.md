# Fixed-Site Asteroid Landing RL Proof-of-Life

Minimal proof-of-life demo: Basilisk/MuJoCo asteroid landing driven by a Gymnasium env + optional PPO.

Success is scored against **altitude above the Itokawa mesh surface** (not the asteroid body origin). Optional **Basilisk body-fixed instrument camera** frames are rendered by Vizard (OpNav path; no VLM yet).

## Layout

```text
asteroid_rl/            # Python package
  env.py                # sim + Gymnasium env (+ optional Vizard)
  surface.py            # mesh heightmap altitude queries
  camera.py             # Basilisk instrument camera helpers
  policies.py           # scripted / random / PPO action helpers
  episode.py            # shared run_episode / CSV / summaries
  cli/                  # entrypoints (play, train, evaluate, …)
assets/                 # MuJoCo XML + Itokawa mesh + heightmap
vendor/                 # original Basilisk scenario (reference)
examples/               # optional full bskExamples dump
logs/  outputs/         # generated artifacts
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -U pip
pip install "bsk[all,examples]"
pip install -r requirements.txt
```

Run commands from the repo root (so `asteroid_rl` imports resolve).

## Run

```bash
# Sanity check
python -m asteroid_rl.cli.smoke_test

# Headless episode (scripted or random or ppo)
python -m asteroid_rl.cli.play --policy scripted
python -m asteroid_rl.cli.play --policy random
python -m asteroid_rl.cli.play --policy ppo --model outputs/ppo_asteroid_fixed_site_v2.zip

# Same, live in Vizard
python -m asteroid_rl.cli.play --policy scripted --viz
python -m asteroid_rl.cli.play --policy ppo --model outputs/ppo_asteroid_fixed_site_v2.zip --viz

# Basilisk hub camera via Vizard (headless -noDisplay unless --viz also set)
python -m asteroid_rl.cli.play --policy scripted --camera --save-frame outputs/plots/navcam.png
python -m asteroid_rl.cli.play --policy scripted --camera --viz

# Train / evaluate
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --device cpu --seed 0
python -m asteroid_rl.cli.evaluate
python -m asteroid_rl.cli.plot_comparison
python -m asteroid_rl.cli.diagnose logs/eval_ppo_episode_0.csv

# Optional curriculum
python -m asteroid_rl.cli.train_curriculum --timesteps-per-stage 2000
```

## Scope

Included: fixed asteroid, **surface landing site**, truth-state obs (altitude / ``v_z`` / range / speed / throttle), scalar throttle, scripted/random/PPO, optional Vizard, optional **Basilisk instrument camera** (Vizard OpNav images).

Excluded: Scenic, VLM reasoning, landing-site selection, full BSK-RL, 3D force/torque control.

**Camera note:** ``--camera`` mounts a Basilisk ``camera.Camera`` on the hub (looking toward the asteroid) and requires a Vizard connection. Use ``--camera`` alone for headless OpNav frames, or ``--camera --viz`` to also see the scene / camera HUD.
