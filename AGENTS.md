# AGENTS.md — Asteroid Landing RL Demo

Instructions for Cursor agents (and humans) working in this repository.
Authoritative product intent comes from the **Scripts Planning Document**
(asteroid landing / docking-style plan). Cross-session machine state lives in
[`WORK_DIARY.md`](WORK_DIARY.md) — **read it first** on every non-trivial session.

---

## 1. Mission (planning document → repo phases)

### End state (full plan)

1. **Scenic** (or scenic-like) samples satellite + asteroid relative geometry:
   within visibility if pointed, but **not** necessarily pointed or closing.
2. If any asteroid is in camera frame → **center it** (VLM + RL).
3. **Qwen3-VL** (or geometry stub with the **same JSON schema**) outputs:
   ```json
   {
     "target_visible": true,
     "landing_site_box": [0.42, 0.55, 0.58, 0.70],
     "hazard_score": 0.22,
     "progress_assessment": "site is visible and slightly left of center"
   }
   ```
4. Hazard gate (~0.10): search / orbit until a safe site, else best hazard.
5. **BSK-RL-shaped** loop: reset → obs → action → Basilisk command → reward → PPO.

### Isolation rule (from the planning notes — mandatory)

> Start with RL, give it the exact landing site and pick it myself and remove
> that variable… Then VLM choosing the landing site. Two ML components =
> entangled failure modes.

**Never** mix unexplained crashes across Scenic + VLM + RL in one change set.
Prove each layer before stacking the next.

### Phase map (what exists vs planned)

| Phase | Goal | Status in repo |
|-------|------|----------------|
| **P0** Mock Basilisk/MuJoCo lander loop | Env steps, throttle → force | Done |
| **P1** Fixed-site soft landing (RL alone) | Scripted + PPO ≈ `safe_landing` | Done (use **best** mesh zip) |
| **P1b** Flat → mesh curriculum, obs modes | truth / sensors / perception stub | Done / partial |
| **P2** Orbital start + point/throttle GNC | Central gravity, ellipse IC | Scaffolded; pad hit rate still weak |
| **P3** Scenic-like / Scenic randomization | Random visible starts | scenic-like only |
| **P4** Real camera + Qwen VLM | Image → JSON | Backend + geometry fallback |
| **P5** Full BSK-RL + long eval suite | Package integration | Adapter only |

---

## 2. Bootstrap (every agent session)

1. Read **Current state**, **Open threads**, latest diary entries in `WORK_DIARY.md`.
2. Do **not** delete or rewrite historical diary entries; append dated sections.
3. After meaningful work: append diary + update the three top editable sections.
4. Prefer Git Bash / bash syntax when the user asks; on Windows PowerShell host
   wrap with `bash -lc '…'` if needed.
5. Never commit unless the user asks. Never force-push `main`/`master`.
6. Do not invent Phase-1 regressions: default `gravity_mode=constant`,
   `action_mode=throttle`, fixed approach start must keep working.

---

## 3. Repository map

```text
asteroid_rl/
  env.py              # Gym env, build_sim, success/crash, orbital hooks
  gravity.py          # ConstantGravity | CentralGravity
  orbit_reset.py      # Keplerian elliptical ICs about asteroid COM
  surface.py          # Heightmap altitude + radial shell off-map
  observations.py     # truth | sensors | perception | orbital packing
  pointing.py         # MRP boresight / direction slews (instant outer loop)
  policies.py         # scripted, scripted_orbit, random*, PPO wrappers
  imitate.py          # BC warm-start from scripted / scripted_orbit
  perception.py       # Geometry stub (VLM JSON schema)
  vlm.py              # Qwen backend + fallback
  mission.py          # Hazard search→land gate
  scenic_reset.py     # PDF-style near-field random starts (no Scenic pkg)
  camera.py           # Basilisk instrument camera + Vizard launch
  episode.py          # run_episode / CSV
  bsk_rl_api.py       # Partial Dict-obs adapter
  cli/                # play, train_*, smoke_test, benchmark_*, evaluate, …
assets/               # MuJoCo XML + Itokawa mesh + heightmap
tests/                # pytest unit + integration smokes
AGENTS.md             # This file
WORK_DIARY.md         # Cross-session memory
README.md             # Human quickstart
```

Package import root: **`asteroid_rl/`** (not `src/`). Run from repo root with
`PYTHONPATH=.` if needed.

---

## 4. Physics & control contracts (do not violate)

### Phase-1 (default)

- Gravity: **constant** inertial force `≈ (0,0,-200) N` (not Keplerian).
- Action: **scalar throttle** `[0,1]` → body **+z** thruster.
- `auto_point`: aims body **−z** at the **fixed** landing site.
- Thrust while pointed at site = **brake away from site** (retrofire).
- There is **no** “negative throttle”; coast = throttle 0 + gravity/initial velocity.

### Phase-2 (`--orbital` / `apply_orbital_defaults()`)

- Gravity: **central** point-mass at asteroid COM `(0,0,-150)`, demo `mu`
  (not real Itokawa µ — keep periods short for RL).
- Action: **`(throttle, dx, dy, dz)`** — slew so body −z follows `(dx,dy,dz)`,
  then fire +z thruster. Attitude is **instant MRP set**, not RW dynamics.
- Obs: **`orbital`** 9-D: rel site xyz, vel xyz, altitude, speed, prev throttle.
- **Flat pad curriculum by default** (`use_flat_surface=True`) — planning-doc
  order: prove divert→land on flat before mesh. Mesh ellipses that spawn into
  the rock cause contact explosions (km/s).
- Starts: **`orbit_start_mode=mixed`** (~75% near-pad approach, rest ellipse)
  with `orbit_min_clearance_m` reject of underground / hypervelocity ICs.
- Off heightmap XY (mesh mode): altitude uses **radial shell** through the
  landing site (`surface.radial_altitude`), not the `1e6` sentinel.
- Eval: prefer `outputs/best_model_orbital/best_model.zip`;
  `python -m asteroid_rl.cli.evaluate_orbital --policy ppo --episodes 8`.

### Phase-3 (`--autonomous` / `apply_autonomous_defaults()`)

- Builds on Phase-2 + **mission FSM**: `search → acquire → divert → upright`.
- **Upright gate**: `require_upright=True` — body **+z thruster** must align with
  local-up (away from COM) within `success_tilt_deg`. Point boresight **−z**
  toward the ground so thrust brakes *away* from the surface (not into it).
- **Thrust authority**: `max_thrust≈2500 N` so hover is possible under demo
  central gravity near the pad (400 N was insufficient).
- Starts: approach-heavy curriculum for settle; scenic/ellipse available via
  `orbit_start_mode`.
- Isolation: success/reward still use the **fixed pad**.
- Nav stand-in: [`asteroid_rl/nav.py`](asteroid_rl/nav.py) documents Basilisk
  IMU / starTracker / reactionWheels mapping; privileged truth is used today.
- Policy: `scripted_autonomous` or PPO from
  `outputs/best_model_autonomous/best_model.zip`.

### Fixed site

- Pad = `default_landing_site()` under `(0,0)` on the heightmap.
- Success does **not** mean “anywhere on the rock”; lateral miss must be ≤ gate.

### `safe_landing` definition (`LandingEnvConfig` / `_check_terminated`)

All must hold:

- `min_success_altitude ≤ altitude ≤ success_altitude` (default **0.5–5.0 m**)
- `speed ≤ success_speed` (default **0.75 m/s**)
- `lateral ≤ success_lateral` (default **20 m**)
- If `require_upright`: tilt(body −z, local-up) ≤ `success_tilt_deg`

Else: `crash` / `escaped` / `timeout`.

**Reward and termination always use simulator truth.** Only the policy obs
vector changes with `obs_mode`. Never put privileged site-distance back into
`sensors` / `perception` policy channels.

---

## 5. Commands cheatsheet

Setup (once):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
pip install -U pip
pip install "bsk[all,examples]"
pip install -r requirements.txt
```

### Tests (run before claiming done)

```bash
PYTHONPATH=. python -m pytest tests/ -q
PYTHONPATH=. python -m asteroid_rl.cli.smoke_test
PYTHONPATH=. python -m asteroid_rl.cli.smoke_test --orbital
```

### Phase-1 play / train

```bash
PYTHONPATH=. python -m asteroid_rl.cli.play --policy scripted
PYTHONPATH=. python -m asteroid_rl.cli.play --policy ppo \
  --model outputs/best_model_truth_mesh_fixed/best_model.zip
PYTHONPATH=. python -m asteroid_rl.cli.train_curriculum --timesteps-per-stage 8000 --device cpu
```

### Phase-2 orbital / Phase-3 autonomous

```bash
PYTHONPATH=. python -m asteroid_rl.cli.play --policy scripted_orbit --orbital
PYTHONPATH=. python -m asteroid_rl.cli.play --policy scripted_autonomous --autonomous
PYTHONPATH=. python -m asteroid_rl.cli.play --policy ppo --autonomous \
  --model outputs/best_model_autonomous/best_model.zip --viz
PYTHONPATH=. python -m asteroid_rl.cli.train_autonomous_ppo --timesteps 100000 --bc-episodes 8 --device cpu
PYTHONPATH=. python -m asteroid_rl.cli.evaluate_autonomous --policy scripted_autonomous --episodes 12
PYTHONPATH=. python -m asteroid_rl.cli.evaluate_autonomous --policy ppo --episodes 8 \
  --model outputs/best_model_autonomous/best_model.zip
PYTHONPATH=. python -m asteroid_rl.cli.train_orbital_ppo --timesteps 100000 --bc-episodes 8 --device cpu
PYTHONPATH=. python -m asteroid_rl.cli.evaluate_orbital --policy ppo --episodes 8 \
  --model outputs/best_model_orbital/best_model.zip
```

### Vizard (Windows)

- Default `--viz` = **save-file** `.bin` then `-loadFile` (live ZeroMQ often
  aborts with libzmq `epoll` on this Windows Basilisk build).
- macOS `--viz` = liveStream.
- Override: `--viz-live` / `--viz-file`. `VIZARD_PATH` for custom `.exe` / `.app`.

### Checkpoint rule

Prefer **EvalCallback best** zips over “final” zips. Known: Phase-1
`ppo_asteroid_fixed_site_v2.zip` can **hover** (~0.73 throttle ≈ weight);
`best_model_truth_mesh_fixed` lands.

---

## 6. Observation modes

| Mode | Dim | Policy sees | Use |
|------|-----|-------------|-----|
| `truth` | 5 | alt, vz, **site range**, speed, throttle | P1 scaffolding |
| `sensors` | 5 | alt, vz, speed, closing-rate, throttle | No site range |
| `perception` | 6 | visibility, uv, hazard, inv-depth, throttle | Stub / VLM schema |
| `orbital` | 9 | rel xyz, vel xyz, alt, speed, throttle | `--orbital` |

`info["truth_state"]` must remain available for scripted baselines and logging
even when the agent obs omits privileges.

---

## 7. Known pitfalls (do not re-learn the hard way)

1. **Orbital scripted “aim at site + brake” ≠ divert.** Thrust while looking at
   the pad pushes **away** along LOS. Large lateral miss needs a burn that
   cancels cross-track / drives toward the pad (see `scripted_orbit_action`).
2. **Changing gravity breaks Phase-1 checkpoints.** Keep `gravity_mode` gated.
3. **MuJoCo actuator is body +z only** in `assets/sat_ast_landing.xml`.
4. **Retain SysModel Python refs** on `SimHandles` or GC → segfault.
5. **Vizard liveStream on Windows** may die with ZeroMQ epoll; use save-file.
6. **`outputs/` and `*.zip` are gitignored** — checkpoints don’t travel via GH.
7. **GPU barely helps** Phase-1/2 MLP PPO (env is CPU/Basilisk-bound). GPU for VLM.
8. **Do not** enable `point_every_step` lightly — can destabilize free-joint dynamics.

---

## 8. Implementation priorities for agents

When asked to “advance the plan,” work in this order unless the user overrides:

1. Keep P1 green (smoke + pytest + scripted/`best_model` land).
2. Make **scripted_orbit** actually reduce lateral miss to the fixed pad
   (PD / velocity-to-go), then BC + longer orbital PPO; keep **best** zip.
3. Scenic-like randomization of approach / ellipse families (still fixed pad).
4. Perception obs training; camera frames; then real Qwen.
5. Hazard mission search with VLM JSON.
6. Real Scenic package / full `bsk_rl` last.

### Definition of done (per task)

- Code compiles / imports.
- Relevant **pytest** + **smoke_test** pass.
- For landing claims: show `termination_reason=safe_landing` and CSV throttle
  varying (not a single constant unless intentional).
- Diary updated; no Phase-1 default regressions.

---

## 9. Testing expectations

Agents must run what they can locally:

```bash
PYTHONPATH=. python -m pytest tests/ -q
PYTHONPATH=. python -m asteroid_rl.cli.smoke_test
PYTHONPATH=. python -m asteroid_rl.cli.smoke_test --orbital
```

Optional deeper checks:

```bash
# Phase-1 scripted land
PYTHONPATH=. python -m asteroid_rl.cli.play --policy scripted --csv logs/agents_scripted.csv
# Orbital scripted: look for decreasing distance_to_target in CSV
PYTHONPATH=. python -m asteroid_rl.cli.play --policy scripted_orbit --orbital --csv logs/agents_orbital.csv
```

Add/extend tests under `tests/` when changing orbit math, obs packing, success
gates, or scripted GNC. Prefer fast unit tests; keep Basilisk integration smokes
short (`reuse_sim` where safe).

---

## 10. Long-running / background sessions

If the user asks for ~1 hour of continuous work:

1. Read this file + `WORK_DIARY.md`.
2. Start long `train_orbital_ppo` (or curriculum) **in background**.
3. In parallel: improve scripted GNC, rewards, tests; append diary ticks.
4. Prefer `/loop 10m …` wakeups to babysit training.
5. Always prefer `outputs/best_model_*/best_model.zip` for demos.

Suggested kickoff prompt (user can paste):

```text
Work ~60m on orbital GNC per AGENTS.md. Read WORK_DIARY. Improve scripted_orbit
divert, run pytest+smoke, background train_orbital_ppo --timesteps 100000,
eval best zip, append diary. Don’t break Phase-1 defaults.
```

---

## 11. Self-prompts (copy into new chats)

1. Summarize `obs_mode` and what reward may use vs what the policy may see.
2. Why Phase-1 `v2` can hover at throttle ≈ 0.73 and which zip to demo instead.
3. Why orbital “point at site + throttle” fails with large lateral miss.
4. How `--viz` differs on Windows vs macOS.
5. Next smallest step from Open threads without stacking VLM+Scenic+RL.

---

## 12. Out of scope unless explicitly requested

- Committing / pushing.
- Real Scenic package install as default dependency.
- Full `bsk_rl` replacement of this Gym env.
- Realistic Itokawa µ / multi-hour orbits.
- Pixel-end-to-end RL without the JSON schema bridge.
