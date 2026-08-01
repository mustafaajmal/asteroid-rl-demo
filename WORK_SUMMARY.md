# Asteroid RL Demo — Work Summary

**Project path:** `C:\Users\Mustafa Ajmal\Desktop\Research\asteroid-rl-demo`  
**Date:** 2026-07-29  
**Source prompts:**
- `Downloads\CURSOR_ASTEROID_RL_DEMO_MASTER_PROMPT.md` (initial proof-of-life)
- `Downloads\CURSOR_ASTEROID_RL_NEXT_STEPS_PHASE_1.md` (Phase 1 hardening)

---

## 1. Executive summary

This work converted the Basilisk/MuJoCo asteroid landing example from a hardcoded thrust sequence into a Gymnasium RL environment with scalar throttle control, diagnostics, training, evaluation, and a small randomization curriculum.

**What was proven**
- Basilisk imports and the original asteroid landing example runs.
- An RL-style `reset()` / `step(throttle)` loop works.
- Scripted, random, and PPO policies can drive the lander and produce logs/plots.
- Fixed-site evaluation and diagnosis tooling is in place.

**What was not proven**
- A successful safe landing under the current success thresholds.
- That PPO outperforms a simple scripted controller.
- Scenic, VLM, camera perception, or autonomous landing-site selection (intentionally excluded).

**Bottom line:** the control/RL plumbing works. Learning quality and the landing-success definition still need work before perception/Scenic layers are added.

---

## 2. Setup performed

### Environment / dependencies
- Created project at `Desktop\Research\asteroid-rl-demo`
- Initialized git repo
- Created Python venv: `.venv` (Python 3.12.2)
- Installed:
  - `bsk[all,examples]` → Basilisk `bsk-2.11.1`
  - `gymnasium`, `stable-baselines3`, `numpy`, `pandas`, `matplotlib`
- Ran `bskExamples` to download Basilisk example assets into `examples/`
- Verified: `import Basilisk` succeeds

### Vendored reference assets
Copied into `vendor_examples/`:
- `scenarioAsteroidLanding_original.py`
- `sat_ast_landing.xml`
- Itokawa mesh/texture files under `vendor_examples/Itokawa/`
- Mirrored mesh data also under `dataForExamples/Itokawa/` so vendor script relative paths resolve

### Original example acceptance
- Ran original asteroid landing example headless successfully (`ORIGINAL_EXAMPLE_OK`)

---

## 3. Repository layout (current)

```text
asteroid-rl-demo/
  README.md
  requirements.txt
  .gitignore
  src/
    asteroid_landing_env.py          # Gymnasium env + build_sim
    smoke_test_env.py
    run_scripted_controller.py
    run_random_policy.py
    train_fixed_site_ppo.py          # original short PPO smoke
    train_fixed_site_ppo_v2.py       # Phase 1 stronger training
    evaluate_fixed_site_policies.py
    diagnose_episode.py
    plot_policy_comparison.py
    plot_logs.py
    policy_utils.py
    train_curriculum_ppo.py          # Phase 1C controlled randomization
  scripts/
    train_fixed_site_ppo_v2_chunked.ps1
  vendor_examples/
    scenarioAsteroidLanding_original.py
    sat_ast_landing.xml
    Itokawa/...
  examples/                          # from bskExamples
  dataForExamples/Itokawa/...
  logs/                              # episode CSVs
  outputs/                           # models, summaries, plots
```

---

## 4. Initial proof-of-life (Master Prompt)

### Goal
Replace the original hardcoded thrust schedule (`275 N` until 47.5 s, then 0 until 70 s) with an environment loop that accepts external throttle actions.

### Core environment behavior
`AsteroidLandingEnv`:
- Builds Basilisk/MuJoCo scene from original setup code
- Action: scalar throttle in `[0, 1]` → thrust = throttle × 275 N
- Observation (5-D truth state):
  1. altitude proxy (distance to target)
  2. vertical velocity proxy (radial)
  3. distance to target
  4. speed
  5. previous throttle
- Reward: progress / speed / fuel / terminal terms
- Terminates on safe landing, crash, escape; truncates on timeout

### Scripts created in first pass
| Script | Purpose |
|--------|---------|
| `smoke_test_env.py` | reset + 10 steps at throttle 0.5 |
| `run_scripted_controller.py` | simple braking heuristic + CSV log |
| `run_random_policy.py` | random throttle baseline |
| `train_fixed_site_ppo.py` | short PPO smoke (~1000 steps) |
| `plot_logs.py` | basic time-series plots from CSV |

### First-pass results
- Smoke test passed
- Scripted/random controllers produced CSVs
- Short PPO smoke interacted with the env and saved a checkpoint

---

## 5. Phase 1 hardening changes

### 5.1 Environment upgrades (`asteroid_landing_env.py`)

**Config (`LandingEnvConfig`)**
- Explicit thresholds: success/crash/escape radii and speeds
- Reward weights: progress, speed, fuel, success/crash/timeout/escape
- Target proxy kept at `(0.0, 0.0, -150.0)` (asteroid body origin in XML)
- Randomization flags for Phase 1C (off by default)
- `reuse_sim=True` by default

**Rich `info` dict on every step**
- `sim_time_sec`, `termination_reason`, `throttle`, `thrust_N`
- `distance_to_target`, `speed`, `vertical_velocity`
- Named reward terms: `reward_total`, `reward_progress`, `reward_speed_penalty`, `reward_fuel_penalty`, `reward_terminal`
- Flags: `success`, `crash`, `escape`, `timeout`

**Termination**
- `safe_landing`, `crash`, `escaped` → `terminated=True`
- `timeout` → `truncated=True` with timeout penalty

**Windows stability fix**
- Repeated `SimBaseClass` teardown under Stable-Baselines3 caused access violations (`0xC0000005`)
- Fix: reuse one Basilisk sim across episodes (`reuse_sim`), soft-reset position/velocity, advance absolute sim time monotonically, track episode time separately

### 5.2 New Phase 1 tooling

| File | Role |
|------|------|
| `policy_utils.py` | shared CSV write, episode summarize, scripted/random helpers |
| `train_fixed_site_ppo_v2.py` | longer PPO training (`--timesteps`, `--device cpu`, `--seed`, `--resume`) |
| `evaluate_fixed_site_policies.py` | compare random / scripted / PPO |
| `diagnose_episode.py` | human-readable failure-mode diagnosis from one CSV |
| `plot_policy_comparison.py` | per-policy and comparison plots |
| `train_curriculum_ppo.py` | Phase 1C staged randomization curriculum |

### 5.3 README update
Added Phase 1 section stating this phase proves fixed-site RL control only, and explicitly excludes Scenic / VLM / camera / autonomous site selection.

---

## 6. Training and evaluation results

### PPO v2 training
- Command: `python src\train_fixed_site_ppo_v2.py --timesteps 20000 --device cpu --seed 0`
- Completed ~20,480 timesteps on CPU (~8–9 minutes)
- Checkpoint: `outputs/ppo_asteroid_fixed_site_v2.zip`
- Intermediate checkpoints under `outputs/checkpoints/`

### Fixed-site evaluation comparison

| Policy | Final distance (m) | Final speed (m/s) | Total reward | Termination |
|--------|--------------------|-------------------|--------------|-------------|
| **scripted** | **126.07** | **0.11** | **+92.5** | timeout |
| random | 140.39 | 0.37 | +17.5 | timeout |
| PPO | 162.64 | 1.36 | -92.4 | timeout |

### PPO diagnosis (`diagnose_episode.py`)
- Episode length: 280 steps
- Final distance: 162.64 m
- Final speed: 1.36 m/s
- Total reward: -92.43
- Termination: timeout
- Min distance: 135.11 m
- Avg throttle: ~0.43 (nearly constant)
- **Likely failure mode:** policy did not make meaningful progress toward target

### Interpretation
- Scripted controller is currently best.
- PPO did **not** beat random or scripted at 20k steps.
- No policy achieved `safe_landing` under current thresholds.

### Important scientific finding
Closest approach for the scripted controller is ~126 m from the body-origin target. A success radius of 5 m around `[0,0,-150]` is likely **physically unreachable** because the scaled Itokawa mesh surface sits far outside that radius. Before claiming “landing learned,” retune:
- target proxy (surface point, not body origin), and/or
- success/crash distance thresholds

---

## 7. Phase 1C curriculum (smoke)

`train_curriculum_ppo.py` stages:
1. fixed start
2. small distance randomization (±5 m)
3. distance + vertical velocity (±0.1 m/s)
4. + small lateral offset (±2 m)

Smoke run with `--timesteps-per-stage 500` produced:
- `outputs/ppo_curriculum_fixed.zip`
- `outputs/ppo_curriculum_distance_small.zip`
- `outputs/ppo_curriculum_velocity_small.zip`
- `outputs/ppo_curriculum_lateral_small.zip`
- `outputs/ppo_asteroid_curriculum_final.zip`

Curriculum plumbing works. Full scientifically meaningful Phase 1C training should wait until fixed-site success criteria are corrected and PPO approaches the scripted baseline.

---

## 8. Key outputs to download / inspect

### Models
- `outputs/ppo_asteroid_fixed_site.zip` (early smoke)
- `outputs/ppo_asteroid_fixed_site_v2.zip` (20k training)
- `outputs/ppo_asteroid_curriculum_final.zip`
- `outputs/ppo_curriculum_*.zip`

### Summaries
- `outputs/fixed_site_eval_summary.csv`
- `outputs/fixed_site_eval_summary.md`

### Plots
- `outputs/plots/comparison_distance.png`
- `outputs/plots/comparison_speed.png`
- `outputs/plots/comparison_throttle.png`
- `outputs/plots/comparison_reward.png`
- Plus per-policy plots: `random_*`, `scripted_*`, `ppo_*`

### Episode logs
- `logs/eval_random_episode_0.csv`
- `logs/eval_scripted_episode_0.csv`
- `logs/eval_ppo_episode_0.csv`
- `logs/scripted_controller_log.csv`
- `logs/random_policy_log.csv`

---

## 9. How to reproduce

```powershell
cd "C:\Users\Mustafa Ajmal\Desktop\Research\asteroid-rl-demo"
.\.venv\Scripts\activate
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"

python src\smoke_test_env.py
python src\run_scripted_controller.py
python src\run_random_policy.py

python src\train_fixed_site_ppo_v2.py --timesteps 20000 --device cpu --seed 0
python src\evaluate_fixed_site_policies.py
python src\plot_policy_comparison.py
python src\diagnose_episode.py logs\eval_ppo_episode_0.csv

# Optional curriculum smoke
python src\train_curriculum_ppo.py --timesteps-per-stage 500
```

---

## 10. Scope intentionally excluded

Not implemented (by design for this phase):
- Scenic scenario generation
- VLM / Qwen camera-based landing-site selection
- Camera image retrieval / perception pipeline
- Orbit-around-asteroid search behavior
- Full BSK-RL class hierarchy migration
- 3D force/torque vector control (scalar throttle only)

---

## 11. Recommended next steps

1. **Retune target / success criteria** so “safe landing” is reachable given the mesh geometry.
2. **Improve reward shaping** and/or train longer once success is reachable.
3. Confirm PPO can approach or beat the scripted controller on the fixed site.
4. Only then run a full Phase 1C curriculum with meaningful stage budgets.
5. After that: Scenic randomization → VLM-selected sites.

### Demo wording (accurate)
> This phase isolates the control problem. The asteroid scene and target proxy are fixed; the policy gets truth-state observations and outputs scalar throttle. The original hardcoded thrust sequence is replaced by an environment loop. This establishes whether RL-style control of the Basilisk/MuJoCo lander is feasible before adding perception and scenario-randomization failure modes. Current PPO training has not yet produced a successful landing or a policy better than the scripted baseline.

---

## 12. File change checklist

### Created
- `src/asteroid_landing_env.py`
- `src/smoke_test_env.py`
- `src/run_scripted_controller.py`
- `src/run_random_policy.py`
- `src/train_fixed_site_ppo.py`
- `src/train_fixed_site_ppo_v2.py`
- `src/evaluate_fixed_site_policies.py`
- `src/diagnose_episode.py`
- `src/plot_policy_comparison.py`
- `src/plot_logs.py`
- `src/policy_utils.py`
- `src/train_curriculum_ppo.py`
- `scripts/train_fixed_site_ppo_v2_chunked.ps1`
- `vendor_examples/*` (original example + assets)
- `README.md`, `requirements.txt`, `.gitignore`
- This summary document

### Modified during Phase 1
- `src/asteroid_landing_env.py` (hardening, diagnostics, reuse_sim, curriculum hooks)
- `src/smoke_test_env.py` (prints reward terms / info)
- `README.md` (Phase 1 section)

### Generated artifacts
- logs CSVs, PPO zip checkpoints, evaluation summary CSV/MD, comparison plots

---

*End of summary.*
