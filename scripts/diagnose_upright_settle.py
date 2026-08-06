"""Diagnose near-pad upright settle."""
from __future__ import annotations

import numpy as np

from asteroid_rl.environment.gym_env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.control.policies import scripted_autonomous_action


def main() -> None:
    cfg = LandingEnvConfig(seed=80).apply_autonomous_defaults()
    cfg.orbit_start_mode = "approach"
    env = AsteroidLandingEnv(cfg)
    obs, info = env.reset()
    for i in range(2000):
        act = scripted_autonomous_action(obs, info)
        obs, _r, term, trunc, info = env.step(act)
        alt = float(info["altitude"])
        spd = float(info["speed"])
        tilt = float(info.get("tilt_deg", 99.0))
        dist = float(info.get("distance_to_target", 99.0))
        if alt <= 5.5 and dist <= 25.0:
            print(
                f"step={i} alt={alt:.2f} dist={dist:.2f} spd={spd:.2f} "
                f"tilt={tilt:.1f} mode={info.get('mission_mode')} thr={float(act[0]):.2f}"
            )
        if term or trunc:
            print(
                "END",
                info.get("termination_reason"),
                "alt",
                alt,
                "spd",
                spd,
                "tilt",
                tilt,
                "dist",
                dist,
            )
            break
    env.close()


if __name__ == "__main__":
    main()
