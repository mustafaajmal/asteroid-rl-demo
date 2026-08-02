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

- **Repo role:** Phase-1 fixed-site asteroid landing RL on Basilisk + MuJoCo + Gymnasium + SB3 PPO. Not full Scenic / VLM / BSK-RL yet.
- **Package root:** `asteroid_rl/` at repo root (not `src/`).
- **Success metric:** Surface altitude / speed / lateral (Itokawa heightmap or optional flat plane). Not body-origin proxy.
- **Default obs:** Still `obs_mode=truth` for scaffolding; **sensors** and **perception** modes exist so policy need not see privileged site distance.
- **Reward / termination:** Always clean simulator truth (privileged), regardless of `obs_mode`.
- **Camera:** Basilisk instrument camera via Vizard OpNav (not MuJoCo offscreen).
- **Last short PPO (20k, truth):** Saved `outputs/ppo_asteroid_fixed_site_v2.zip`; eval ~**-473** (still crash-level). Scripted baseline still gets `safe_landing`.
- **Hardware split:** M2 = iterate / short trains / Vizard; home 7600X3D+5080 = long PPO (`--device cpu` for MLP; GPU later for VLM).

---

## Open threads (edit in place)

- [ ] Longer PPO on home PC (1e5–5e5+), still `--device cpu`.
- [ ] Train / compare `--obs-mode sensors` and `--obs-mode perception` (honest policy obs).
- [ ] Perception stub still filled from **geometry**, not camera pixels / VLM — next real step toward non-cheat sensing.
- [ ] Optional: curriculum / flat-surface pretrain then mesh.
- [ ] Scenic, Qwen VLM, full `bsk_rl` package — still out of scope.
- [ ] Undertrained PPO often coasts (throttle→0) then crashes; reward reshape + `ent_coef` help but need more steps.

---

## Self-prompts for next session

1. `Read WORK_DIARY.md Current state + Open threads, then summarize what obs_mode means in this repo.`
2. `Do not reintroduce privileged site-distance into sensors/perception policy vectors.`
3. `When changing reward or obs, keep reward on truth; only change what the policy sees.`
4. `Scripted baseline must keep working via info["truth_state"] / info telemetry even if obs_mode != truth.`
5. `Prefer appending to this diary over rewriting chat history.`

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
