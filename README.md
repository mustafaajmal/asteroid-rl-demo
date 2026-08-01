# Fixed-Site Asteroid Landing RL Proof-of-Life

This repo is a minimal proof-of-life demo around the Basilisk/MuJoCo asteroid landing example.

## Goal

The goal is to replace the original hardcoded thrust sequence with an environment loop that can accept actions from a controller or RL policy.

## Current scope

Included:

- Fixed asteroid scene
- Fixed manually selected landing target / target proxy
- Truth-state observation
- Scalar throttle action in `[0, 1]`
- Simple reward function
- Scripted controller proof-of-life
- Optional PPO training proof-of-life

Excluded intentionally:

- Scenic
- VLM
- Camera perception
- Landing-site selection
- Full BSK-RL integration
- 3D force/torque control

## Why exclude Scenic and VLM first?

The first demo isolates the RL/control problem. If the lander crashes in this version, the failure is likely due to control design, reward design, action interfacing, or simulation stepping. It is not due to VLM perception or Scenic-generated randomization.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install "bsk[all,examples]"
pip install gymnasium stable-baselines3 numpy pandas matplotlib
```

Verify:

```powershell
python -c "import Basilisk; print('Basilisk imported successfully')"
```

Download examples:

```powershell
bskExamples
```

## Run tests

```powershell
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
python src\smoke_test_env.py
python src\run_scripted_controller.py
python src\run_random_policy.py
python src\train_fixed_site_ppo.py
```

## Demo statement

This demo refactors the original Basilisk asteroid landing example so that the thrust command is no longer hardcoded. Instead, an environment loop observes the lander state, accepts a scalar throttle action, writes that command to the MuJoCo thrust actuator message, advances the simulation, computes reward, and repeats.

This establishes whether RL control of the Basilisk/MuJoCo lander is feasible before adding Scenic randomization and VLM-selected landing sites.

## Phase 1: Fixed-Site RL Control Demo

This phase isolates the control problem. The asteroid scene, target proxy, and initial state are fixed. The policy receives truth-state numerical observations and outputs a scalar throttle action. The environment converts throttle into the existing Basilisk/MuJoCo thrust actuator command, advances the simulation, computes reward, and logs the result.

This phase intentionally excludes Scenic, camera perception, VLM inference, and autonomous landing-site selection. The goal is to determine whether an RL policy can learn to control the lander toward a known target before adding perception and scenario-randomization failure modes.

This phase proves the fixed-site RL control loop only. It does not yet prove VLM perception, Scenic scenario generation, autonomous landing-site selection, or full BSK-RL integration.

### Run

```powershell
.\.venv\Scripts\activate
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
python src\smoke_test_env.py
python src\train_fixed_site_ppo_v2.py --timesteps 20000
python src\evaluate_fixed_site_policies.py
python src\plot_policy_comparison.py
python src\diagnose_episode.py logs\eval_ppo_episode_0.csv
```

Optional Phase 1C curriculum (after fixed-site checkpoint exists):

```powershell
python src\train_curriculum_ppo.py --timesteps-per-stage 2000
```

### Outputs

- `outputs/ppo_asteroid_fixed_site_v2.zip` - trained PPO checkpoint
- `outputs/fixed_site_eval_summary.csv` - policy comparison summary
- `outputs/fixed_site_eval_summary.md` - human-readable summary
- `outputs/plots/` - distance, speed, throttle, reward plots

### Interpretation

If PPO outperforms random and approaches or matches the scripted controller, the RL control loop is promising. If PPO fails, use `diagnose_episode.py` and the reward-term logs to determine whether the problem is reward shaping, action scaling, termination thresholds, or simulation dynamics.
