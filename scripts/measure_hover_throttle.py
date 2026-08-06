"""Find hover throttle and improve settle diagnostic."""
from __future__ import annotations

import numpy as np

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.policies import scripted_autonomous_action
from asteroid_rl.pointing import local_up_N, thruster_up_tilt_deg


def measure_hover(seed: int = 80) -> None:
    cfg = LandingEnvConfig(seed=seed).apply_autonomous_defaults()
    cfg.orbit_start_mode = "approach"
    cfg.max_steps = 400
    env = AsteroidLandingEnv(cfg)
    obs, info = env.reset()
    # Drive toward overhead then probe thrust vs accel
    for i in range(250):
        act = scripted_autonomous_action(obs, info)
        obs, _r, term, trunc, info = env.step(act)
        if term or trunc:
            print("ended early", info.get("termination_reason"))
            env.close()
            return
        alt = float(info["altitude"])
        lat = float(np.linalg.norm(np.asarray(info["truth_state"])[0:2])) if False else None
        # get lateral from rel in obs
        o = np.asarray(obs).reshape(-1)
        lateral = float(np.linalg.norm(o[0:2])) if o.size >= 9 else float("nan")
        if alt < 100 and lateral < 25:
            break

    # Freeze pointing look-at-pad; sweep throttle and measure vertical accel
    print(
        f"probe at alt={info['altitude']:.2f} spd={info['speed']:.2f} "
        f"tilt={info.get('tilt_deg', -1):.1f} mode={info.get('mission_mode')}"
    )
    o = np.asarray(obs).reshape(-1)
    rel = o[0:3]
    point = -rel / max(np.linalg.norm(rel), 1e-9)
    results = []
    for thr in np.linspace(0.2, 1.0, 17):
        # clone by stepping then we'll mess state — better: use instantaneous force estimate
        # Use env internals: mass and gravity from handles
        pass

    # Access sc mass / accel via recorder
    handles = env.handles
    mass = float(handles.sc.hub.m_sc) if hasattr(handles.sc.hub, "m_sc") else float("nan")
    r = np.array(handles.sc.scStateOutMsg.read().r_BN_N, dtype=float)
    # Estimate g from mu if available
    print("mass", mass, "r", r, "max_thrust", env.config.max_thrust)

    # Empirical: run short open-loop hover tests from current state by resetting probe
    env.close()

    # Fresh env: teleport-like by running scripted until near, then hold constant throttle
    for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        cfg = LandingEnvConfig(seed=seed).apply_autonomous_defaults()
        cfg.orbit_start_mode = "approach"
        cfg.max_steps = 800
        env = AsteroidLandingEnv(cfg)
        obs, info = env.reset()
        for _ in range(200):
            act = scripted_autonomous_action(obs, info)
            obs, _r, term, trunc, info = env.step(act)
            if term or trunc:
                break
            o = np.asarray(obs).reshape(-1)
            if float(o[6]) < 90 and float(np.linalg.norm(o[0:2])) < 20:
                break
        o = np.asarray(obs).reshape(-1)
        rel = o[0:3].copy()
        point = -rel / max(float(np.linalg.norm(rel)), 1e-9)
        alt0 = float(info["altitude"])
        v0 = float(info["speed"])
        for _ in range(40):
            act = np.array([thr, point[0], point[1], point[2]], dtype=np.float32)
            obs, _r, term, trunc, info = env.step(act)
            o = np.asarray(obs).reshape(-1)
            rel = o[0:3].copy()
            point = -rel / max(float(np.linalg.norm(rel)), 1e-9)
            if term or trunc:
                break
        alt1 = float(info["altitude"])
        vz = float(o[5]) if o.size >= 9 else float("nan")  # vel_z in orbital obs?
        print(
            f"thr={thr:.2f} alt {alt0:.1f}->{alt1:.1f} (d={alt1-alt0:+.2f}) "
            f"spd={info['speed']:.2f} vz={vz:.3f} end={info.get('termination_reason')}"
        )
        env.close()


if __name__ == "__main__":
    measure_hover()
