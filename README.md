# Fixed-Site Asteroid Landing RL Proof-of-Life

Minimal proof-of-life demo: Basilisk/MuJoCo asteroid landing driven by a Gymnasium env + optional PPO.

## Layout

```text
asteroid_rl/            # Python package
  env.py                # sim + Gymnasium env (+ optional Vizard)
  policies.py           # scripted / random / PPO action helpers
  episode.py            # shared run_episode / CSV / summaries
  cli/                  # entrypoints (play, train, evaluate, …)
assets/                 # MuJoCo XML + Itokawa mesh
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

# Train / evaluate
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --device cpu --seed 0
python -m asteroid_rl.cli.evaluate
python -m asteroid_rl.cli.plot_comparison
python -m asteroid_rl.cli.diagnose logs/eval_ppo_episode_0.csv

# Optional curriculum
python -m asteroid_rl.cli.train_curriculum --timesteps-per-stage 2000
```

## Scope

Included: fixed asteroid, fixed target proxy, truth-state obs, scalar throttle, scripted/random/PPO, optional Vizard.

Excluded: Scenic, VLM, camera perception, landing-site selection, full BSK-RL, 3D force/torque control.
