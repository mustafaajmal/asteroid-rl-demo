# Asteroid RL Demo — Work Diary

> **For Cursor / future sessions:** Read this file at the start of non-trivial work.
> **Append only** — never delete or rewrite past entries. Add a new dated section.
> Sync this file via git so M2 / home PC / other machines share context.

---

## How to use this diary (agent prompt)

```text
Before coding on this repo:
1. Read WORK_DIARY.md (at least the latest entries + "Current state" + "Open threads").
2. After meaningful changes, APPEND a new entry under today's date with:
   - What changed (files + why)
   - Commands tried / results if relevant
   - Decisions / gotchas
   - Updated "Current state" and "Open threads" bullets at the top (edit those two sections in place; leave history intact)
3. Do not collapse or delete historical entries.
```

---

## Current state (edit in place)

- **Repo role:** Phase-1 fixed-site + Phase-2 orbital + **Phase-3 autonomous** (mission FSM + upright gate + scenic∪orbit starts).
- **Package layout (2026-08-06):** `asteroid_rl/{environment,dynamics,control,sensing,adapters,analysis,cli}` (+ `paths.py`). Public imports e.g. `asteroid_rl.environment.gym_env`.
- **Success metric:** alt/speed/lateral; autonomous also requires **tilt ≤ success_tilt_deg** when `require_upright`.
- **PPO Phase-1:** `outputs/best_model_truth_mesh_fixed/best_model.zip`.
- **PPO orbital:** prefer `outputs/best_model_orbital/best_model.zip` (~16.7% approach land without upright gate).
- **Upright GNC (2026-08-06):** Sensors are **not** the bottleneck (privileged `r,v,σ` already available). Root bugs were thruster-vs-up axis, underpowered thrust, **fixed pad-level hover under central-g**, and **look-at-pad settle tilt**. Scripted autonomous approach+upright now **~62.5%** `safe_landing` (was ~0–12%).
- **PPO autonomous:** 200k train **finished**. Release: https://github.com/mustafaajmal/asteroid-rl-demo/releases/tag/autonomous-upright-200k-20260806-1235 (`best_model.zip` + final). Prefer best zip for eval/play.
- **Mission FSM:** search → acquire → divert → upright (near approaches auto-commit divert).
- **Tests:** **35** pytest + Phase-1/`--orbital` smoke green after reorg.
- **Vizard:** Windows save-file default; `_setup_vizard` accepts procedural model/texture/scale overrides.
- **Hardware:** home PC long trains.
- **Removed:** `vendor/` (old Basilisk reference scenario), stale `WORK_SUMMARY.md`. Kept `examples/` (bskExamples dump / asset fallback) with `examples/README.md`.
- **Scenic:** sister `../Scenic` branch `basilisk-simulator` — stock Itokawa **and** Mars-style **procedural** asteroid (bumps/craters/ridges + albedo each sample).

---

## Open threads (edit in place)

- [ ] Eval 200k best zip with upright gate on approach (scripted was ~62.5%; compare PPO).
- [ ] Wire asteroid_rl `scenic_reset` to real Scenic `ProceduralAsteroid` / random approach (sister Scenic still holds the interface).
- [ ] Trim remaining ~5–7 m hover timeouts / rare escapes in scripted settle.
- [ ] Retarget success/reward to mission candidate site after upright divert works.
- [ ] Optional later: wire real Basilisk `imuSensor` / `starTracker` / `reactionWheels` (realism, not unlock).
- [ ] Mesh after flat autonomous is solid; VLM play with `--perception vlm --camera`.
- [ ] Optional: delete or gitignore bulky `examples/` once confident `assets/` alone is enough.
- [ ] Optional real Scenic package later; full `bsk_rl` still optional.

---

## Self-prompts for next session

1. `Read WORK_DIARY.md Current state + Open threads, then summarize what obs_mode means in this repo.`
2. `Do not reintroduce privileged site-distance into sensors/perception policy vectors.`
3. `When changing reward or obs, keep reward on truth; only change what the policy sees.`
4. `Scripted baseline must keep working via info["truth_state"] / info telemetry even if obs_mode != truth.`
5. `Prefer appending to this diary over rewriting chat history.`
6. `Orbital/autonomous: always eval best_model_* zip, not the final zip.`
7. `Phase-1 defaults: require_upright=False, constant gravity, 1-D throttle.`

---

## 2026-08-01 — Session log

### Context / machines

- Working primarily on **M2 MacBook**; home desktop (Ryzen 7600X3D + RTX 5080) intended for long training.
- Planning doc reference: Scripts Planning Document (PDF under Downloads on author machine).

### Earlier in project (condensed backlog)

- Extensive Google-style docs across package modules.
- `.gitignore` expansion.
- Confirmed repo can grow into full project; Phase-1 mockup without Scenic.
- Surface success (heightmap altitude), kept throttle loop, Basilisk camera (not MuJoCo offscreen).
- Camera framing: moved off-axis, thruster viz scale cut (100→8), start z=120.
- Pointing stub (`auto_point` on reset; `point_every_step` default False — every-step slew destabilized dynamics).
- Perception stub JSON schema (`perception.py`); scripted policy can use it.
- Light reset randomization; partial BSK-RL-shaped Dict API (`bsk_rl_api.py`).
- Known: Vizard needs `liveStream=True`; GC segfault if SysModel refs not kept on `SimHandles`.

### Reward / train QoL / flat surface / noise

- Reshaped reward: lower progress credit when fast, `impact_speed_weight`, higher crash/success stakes, `safe_progress_scale`.
- Config: `obs_noise_std`, `use_flat_surface`, `flat_surface_z`.
- `train_ppo.py`: default timesteps 1e5 docs, `ent_coef=0.01`, EvalCallback, CheckpointCallback, throttle logging, `--flat-surface`, `--obs-noise`.
- CLI flags on play / benchmark for flat surface.
- README: M2 vs home PC training notes.

### Observation modes (anti-cheat policy obs)

**Why:** User correctly noted training only on internal truth is privileged / “cheating” for the real sensing problem. Split policy obs from reward truth.

**Added / changed:**

| Piece | Change |
|-------|--------|
| `asteroid_rl/observations.py` | **New.** Modes `truth` / `sensors` / `perception`; packing helpers. |
| `asteroid_rl/env.py` | `_get_truth_vector` vs `_get_agent_obs`; `LandingEnvConfig.obs_mode`; reward/term on truth only; `info["truth_state"]`, `info["obs_mode"]`. |
| `asteroid_rl/episode.py` | CSV logs privileged telemetry from `info`, not agent `obs` indices. |
| `asteroid_rl/policies.py` | Scripted reads altitude/speed from `info` / `truth_state`. |
| `asteroid_rl/bsk_rl_api.py` | Dict space drops privileged keys under sensors/perception. |
| `asteroid_rl/cli/train_ppo.py`, `play.py` | `--obs-mode {truth,sensors,perception}`. |
| `asteroid_rl/cli/smoke_test.py` | Exercises all three modes. |
| `README.md` | Observation-modes table. |

**Mode meanings:**

| Mode | Policy vector |
|------|----------------|
| `truth` | alt, vz, **site distance**, speed, prev throttle |
| `sensors` | altimeter, vz, speed, closing-rate, prev throttle (no site distance) |
| `perception` | visible, site u/v, hazard, inv-depth, prev throttle |

**Caveat still open:** `perception` mode uses geometry stub features, not real camera/VLM inference.

### Runs (this session)

- Smoke + scripted mesh/flat benchmarks: `safe_landing` OK.
- PPO 20k CPU truth-mode: completed; `outputs/ppo_asteroid_fixed_site_v2.zip`; eval reward stuck ~-473.
- User asked to stop worrying about running things mid-session; focused on obs-mode code instead.
- Interrupted / skipped full optional flat + noise retrain loops.

### Useful commands (copy-paste)

```bash
python -m asteroid_rl.cli.smoke_test
python -m asteroid_rl.cli.benchmark_baseline --episodes 3
python -m asteroid_rl.cli.benchmark_baseline --episodes 3 --flat-surface
python -m asteroid_rl.cli.play --policy scripted
python -m asteroid_rl.cli.play --policy scripted --obs-mode perception
python -m asteroid_rl.cli.play --policy scripted --viz
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --device cpu --seed 0
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --obs-mode sensors --device cpu
python -m asteroid_rl.cli.train_ppo --timesteps 20000 --obs-mode perception --device cpu
# Home PC long run:
python -m asteroid_rl.cli.train_ppo --timesteps 200000 --device cpu --seed 0
```

### Decisions worth keeping

1. Truth in **reward** is OK; truth in **policy obs** is scaffolding only.
2. `--device cpu` for SB3 MLP — env/Basilisk bound; GPU reserved for future VLM.
3. Do not enable `point_every_step` by default.
4. `--viz` implies instrument camera enable.

### Planning-doc next steps implemented

- **prove-ppo:** `train_curriculum` flat→mesh + BC warm-start (`imitate.py`). Truth PPO lands flat+mesh (`ppo_asteroid_fixed_site_v2.zip` / `ppo_asteroid_curriculum_final.zip`).
- **honest-obs:** sensors curriculum lands; perception short-train/benchmark wired (PPO perception still weak).
- **bench-pdf:** `cli/benchmark_suite.py` flat/mesh × obs modes × scripted/PPO → `outputs/benchmark_suite_summary.csv`.
- **vlm-hook:** `vlm.PerceptionBackend` + env `perception_backend`; CLI `--perception`.
- **site-search:** `mission.py` + `enable_mission_search` / `--mission-search`.
- **scenic-later:** `scenic_reset.py` + `--scenic-like` (no Scenic pkg).
- **Gotcha:** curriculum must not overwrite truth `v2` zip when training other obs modes (fixed). Best-model paths now include `obs_mode` in the name.

### Removed Windows chunked-train script

- Deleted `scripts/train_fixed_site_ppo_v2_chunked.ps1` and empty `scripts/`.
- It only wrapped `python -m asteroid_rl.cli.train_ppo` with `--resume` in short process chunks for Basilisk Windows teardown crashes.
- **Recreatable:** loop calling `train_ppo --timesteps <chunk> --resume <zip> --out <zip>` (Python or shell). Prefer Python/`train_ppo` directly on Mac/Linux.

### Dead-code cleanup / light refactor

- **Prompt / goal:** Find redundant/unused code; delete only what’s clearly dead; refactor duplicates.
- **Removed (dead):**
  - `asteroid_rl/sim.py` — unused re-export shim (nothing imported it).
  - `env._launch_vizard_livestream` — superseded by direct `launch_vizard_for_camera` in `_setup_vizard`.
  - `env._get_obs` alias, `LandingConfig` alias, deprecated `success_radius` / `crash_radius`.
  - `observations.mode_description` (zero callers).
  - Unused `Tuple` import in `camera.py`.
- **Refactored:**
  - Perception packing unified in `perception.py` (`perception_policy_features` 5-D; `perception_feature_vector` = first 4).
  - Shared `asteroid_rl/plotting.py` used by `cli/plot_logs` and `cli/plot_comparison`.
- **Left alone on purpose:**
  - `examples/` Basilisk dump (large; mesh/XML fallbacks still referenced from `env.py`).
  - `vendor/` reference scenario.
  - `bsk_rl_api`, `train_curriculum`, `plot_logs` CLIs (entrypoints / intentional stubs).
  - Stale `WORK_SUMMARY.md` (historical; superseded by README + this diary — not deleted).
  - Phantom `src/` in some IDE indexes — **not on disk / not in git**.

---

### 2026-08-05 — Windows Vizard save-file fallback

- **Prompt / goal:** Fix Vizard timeout / epoll crash on Windows liveStream.
- **Root cause:** Basilisk bundled libzmq aborts (`epoll.cpp:73`); Vizard then times out on `tcp://localhost:5556`. Mac liveStream still fine. `saveFile` mode works on this PC.
- **Changes:** `--viz` → `auto` (file on Windows, live on Darwin); `--viz-live` / `--viz-file`; after episode open `.bin` via `-loadFile`.
- **Files:** `asteroid_rl/env.py`, `asteroid_rl/camera.py`, `asteroid_rl/cli/play.py`, `WORK_DIARY.md`.
- **Next:** User run `python -m asteroid_rl.cli.play --policy scripted --viz`.

---

### 2026-08-05 — Elliptical-orbit GNC slice

- **Prompt / goal:** Elliptical start + pointing/GNC training per plan.
- **Changes:** `CentralGravity`; `orbit_reset.py`; `action_mode=point_throttle` (4-D); `obs_mode=orbital`; radial altitude off heightmap; `scripted_orbit`; `play --orbital`; `train_orbital_ppo` with BC.
- **Files:** `asteroid_rl/gravity.py`, `orbit_reset.py`, `env.py`, `surface.py`, `observations.py`, `pointing.py`, `policies.py`, `imitate.py`, `cli/play.py`, `cli/train_orbital_ppo.py`, README, WORK_DIARY.
- **Runs:** BC mse≈0.004; PPO 20k steps → `outputs/ppo_orbital_final.zip` + `outputs/best_model_orbital/best_model.zip`. Scripted/PPO not yet reliable `safe_landing` from ellipse (needs longer train).
- **Gotchas:** Heightmap altitude sentinel off-map → use radial shell; Phase-1 defaults unchanged.
- **Next:** Longer `--timesteps 1e5+` on home PC; improve scripted deorbit→pad.

---

### 2026-08-05/06 — Orbital GNC 100k push (planning doc Phase-2)

- **Prompt / goal:** ~120m orbital GNC: improve `scripted_orbit` divert, train 100k PPO, eval best zip, implement planning-doc pieces reasonably, thoroughly test.
- **Changes:**
  - `scripted_orbit`: velocity-target divert toward fixed pad + near-pad lateral cancel then LOS brake (not anti-velocity-only).
  - Curriculum: `orbit_start_mode=mixed` + `sample_approach_start`; **flat pad** in `apply_orbital_defaults` (planning-doc flat-before-mesh); `orbit_min_clearance_m` reject (mesh ellipse ICs were spawning underground → km/s contact explosions).
  - Orbital reward extras: `orbital_range_progress_weight` / `orbital_lateral_progress_weight` (0 on Phase-1).
  - CLI: `evaluate_orbital.py`; AGENTS.md Phase-2 docs updated.
  - Tests: approach sampler, Phase-1 default regression, policies divert; **21 pytest passed**. Phase-1 scripted still `safe_landing`.
- **Files:** `policies.py`, `orbit_reset.py`, `env.py`, `episode.py` (None csv), `cli/train_orbital_ppo.py`, `cli/evaluate_orbital.py`, `AGENTS.md`, `tests/*`, `WORK_DIARY.md`.
- **Runs / results:**
  - Train: `train_orbital_ppo --timesteps 100000 --bc-episodes 8 --device cpu` (~805s, ~124 fps). Best eval mean_reward peaked ~**-279**. Saved `outputs/best_model_orbital/best_model.zip` + `outputs/ppo_orbital_final.zip`.
  - Scripted flat approach: **~12.5%** safe_landing (2/16, seed 50).
  - PPO **best** flat approach: **16.7%** (2/12, seed 200). PPO **final** same seeds: **0%** — prefer best zip.
  - PPO best mixed: 0/8 (still hard).
- **Gotchas:** Ellipse without clearance → altitude negative + speed ~1400 m/s. Aggressive PD/`m*|a|` throttle escapes. Point-at-site thrust = brake away; divert needs thrust *toward* pad (point away).
- **Next:** More BC from improved scripted + longer train; then drop flat / raise ellipse fraction; perception/VLM later.

---

### 2026-08-06 — Full autonomous landing stack

- **Prompt / goal:** Implement planning-doc autonomous stack: mission FSM, upright gate, scenic∪orbit, scripted_autonomous, train/eval CLIs; don’t break Phase-1.
- **Changes:**
  - `mission.py`: modes `search|acquire|divert|upright|land`; near-pad auto-commit divert; pointing suggestions; throttle gates.
  - `pointing.py`: `local_up_N`, `boresight_tilt_deg`.
  - `env.py`: `require_upright` / tilt success (default off); `apply_autonomous_defaults()`; mission bootstrap on reset; `info` tilt + pointing_command.
  - `orbit_reset.py`: `start_mode=autonomous` mixes approach/scenic/ellipse.
  - `policies.py`: `scripted_autonomous_action`; play `--autonomous`.
  - CLIs: `train_autonomous_ppo`, `evaluate_autonomous`.
  - Tests: mission/upright/autonomous; **33 pytest passed**; Phase-1 still lands.
- **Runs:** 100k autonomous PPO → `outputs/best_model_autonomous/best_model.zip` (best eval ~-344). Approach+upright eval: **0/10** safe_landing (mean min_dist ~23 m — closes in, fails settle/tilt). Isolation: success still fixed pad.
- **Gotchas:** Mission search throttle-cap starved divert until near-approach auto-commit + reset bootstrap. Upright gate makes success much harder than orbital-only.
- **Next:** Harden upright settle in scripted_autonomous; more BC; then candidate-site retarget; VLM play.

---

### 2026-08-06 — Upright land: sensors vs GNC + gravity-aware settle

- **Prompt / goal:** What sensors are needed to land upright? Why isn’t more compute enough? Implement everything needed + train more.
- **Answer (Basilisk sensors docs):** `imuSensor`, `starTracker`, `camera`, plus actuators `reactionWheels` / thrusters. **Do not need more sensors to unlock upright today** — privileged hub state already has attitude/rate/position. Sensors add flight-like noise; RW would help attitude without thruster coupling. Bottleneck was GNC + actuation.
- **Why not just compute:** PPO/BC was cloning a broken expert (wrong thruster axis earlier; then fixed hover=0.66 at all altitudes under central-g where true hover is ~0.23 at 85 m; then look-at-pad settle caused ~atan(L/h) tilt and failed upright gate).
- **Changes:**
  - `gravity.hover_throttle_central` + env `info["hover_throttle"]` / `position_N` / `gravity_mu`.
  - `scripted_autonomous`: altitude-adaptive hover + vertical-rate tracking; **local-up settle pointing** (not look-at-pad); divert only for large lateral / horiz speed.
  - Autonomous defaults: upright cone 120 m / 50 m; `success_speed=0.85`.
  - `nav.py` notes updated from Basilisk sensors page.
- **Runs:** scripted approach+upright **15/24 = 62.5%** safe_landing (`outputs/eval_upright_scripted_localup.json`). Stopped stale 150k train; started **200k + 20-ep BC** → `logs/train_autonomous_upright_200k.log`.
- **Gotchas:** Under central gravity, hover throttle **must** track `µ/r²`. Look-at-pad near overhead is almost never upright.
- **Next:** Eval PPO best zip when 200k finishes; optional RW later.

---

### 2026-08-06 — Package reorg into descriptive subfolders

- **Prompt / goal:** Organize codebase into proper folders; delete unused/deprecated carefully; descriptive directory names. Scope: careful (CLI entrypoints preserved; no aggressive deletions beyond chosen items).
- **Changes:**
  - Split flat `asteroid_rl/*.py` into:
    - `environment/` — `gym_env.py` (was `env.py`), `episode`, `observations`, `surface`
    - `dynamics/` — `gravity`, `pointing`, `orbit_reset`, `scenic_reset`
    - `control/` — `policies`, `mission`, `nav`, `imitate`
    - `sensing/` — `camera`, `perception`, `vlm`
    - `adapters/` — `bsk_rl_api`
    - `analysis/` — `plotting`
  - Added `asteroid_rl/paths.py` for assets/examples resolution (nesting-safe).
  - Kept `asteroid_rl.cli.*` module paths unchanged (`python -m asteroid_rl.cli.play`, etc.).
  - Deleted `vendor/` and stale `WORK_SUMMARY.md`. Kept `examples/` with explanatory README (asset fallback only).
  - Updated README + AGENTS repository maps / import guidance.
- **Runs:** `pytest tests/ -q` → **35 passed**; `smoke_test` + `smoke_test --orbital` OK.
- **Gotchas:** Do not recreate flat top-level module names that collide with new package dirs. Prefer `from asteroid_rl.environment.gym_env import …`.
- **Next:** Resume autonomous train eval thread; optionally prune `examples/` later.

### 2026-08-12 — Scenic ↔ Basilisk interface (sister Scenic fork)

- **Prompt / goal:** Fully connect Scenic to Basilisk (not nested repo). Use MuJoCo PR #433 as pattern. Branch not master; fork OK.
- **Layout:** `Desktop/Research/Scenic` (sister of `asteroid-rl-demo`). Fork `mustafaajmal/Scenic`, branch **`basilisk-simulator`**.
- **Implemented (Scenic):** Basilisk simulator package + examples + tests; procedural asteroid (bumps/craters/ridges + albedo).
- **Push:** https://github.com/mustafaajmal/Scenic/tree/basilisk-simulator
- **asteroid_rl:** `_setup_vizard` model/texture/scale overrides for procedural meshes (`environment/gym_env.py`).
- **Gotchas:** After package reorg, Scenic must import `asteroid_rl.environment.gym_env` (not `asteroid_rl.env`).
- **Next:** Wire `scenic_reset` to real Scenic scene generation.

---

## Template for next entries

```markdown
### YYYY-MM-DD — <short title>

- **Prompt / goal:** …
- **Changes:** …
- **Files:** …
- **Runs / results:** …
- **Gotchas:** …
- **Next:** …
```

### 2026-08-12 — Seamless Basilisk/Scenic polish + env shim

- **Prompt / goal:** Away-from-desk integration pass vs MuJoCo Scenic feel; push with handoff MD.
- **Scenic:** live angularVelocity; `param timestep` / `asteroid_rl_root`; altitude_brake + record examples; HANDOFF.md; tests (8 passed).
- **asteroid_rl:** `asteroid_rl/env.py` compat shim → `environment.gym_env` (reorg).
- **Gotcha:** Planning PDF not found on this machine; followed MuJoCo/Webots patterns + diary.
- **Next:** scenic_reset → real Scenic generate(); surface-relative altitude.


### 2026-08-19 — MINIMUM Scenic policy eval + mesh radar altitude

- **Gap:** Were ~70% to MINIMUM (eval fixed policy on Scenic scenarios). Missing: real surface altitude, scenic_reset→generate, curriculum harness.
- **Done:** mesh raycast altitude in Scenic; curriculum sphere/ellipsoid/bumpy; `run_scenic_policy_eval.py`; Gym `scenic_scenario_path` + `evaluate_scenic` CLI; ARCHITECTURE.md / HANDOFF.
- **Smoke:** sphere 5/5, ellipsoid 4/5, bumpy 5/5 safe (seed 11).
- **Next:** bake procedural heightmap into Gym SurfaceMap for train-on-bumps; PPO-vs-Scenic if zip available.


### 2026-08-19 — Dual-metric Scenic results + Gym mesh-radar altitude

- Scenic sweep: sphere/ellipsoid 100% reach / 0% soft; bumpy 67% reach+soft (mean spd 1.75 vs ~3.1).
- Gym: bake heightmap + keep mesh for raycast altitude on scenic resets; train_scenic_curriculum CLI added.
- Gap: Gym MuJoCo geometry still stock — need build_procedural in Gym reset for real train-on-bumps.


### 2026-08-19 — Close Gym procedural physics gap

- Added `procedural_sim.build_procedural_sim`; Gym scenic resets rebuild MuJoCo from Scenic mesh.
- Smoke: altitude 140→24 on procedural OBJ; curriculum train sphere 75% / ellipsoid 0–25% / bumpy 25–50%.
- User action: none required for MINIMUM — see Scenic RESULTS.md.

