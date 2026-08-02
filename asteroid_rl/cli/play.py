"""Run one landing episode under a chosen policy.

Entrypoint for headless or Vizard playback of scripted, random, or PPO control.
Does not train a model; for PPO it only loads an existing checkpoint.

Examples:
    python -m asteroid_rl.cli.play --policy scripted
    python -m asteroid_rl.cli.play --policy random
    python -m asteroid_rl.cli.play --policy ppo --model outputs/ppo_asteroid_fixed_site_v2.zip
    python -m asteroid_rl.cli.play --policy scripted --viz
    python -m asteroid_rl.cli.play --policy scripted --camera --save-frame outputs/plots/navcam.png
    # --camera uses Basilisk's body-fixed instrument camera via Vizard (OpNav path)
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace

from stable_baselines3 import PPO

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.episode import ensure_dirs, run_episode
from asteroid_rl.policies import make_action_fn


def parse_args() -> Namespace:
    """Parse command-line arguments for episode playback.

    Returns:
        Parsed argparse namespace with ``policy``, ``model``, ``seed``, ``viz``,
        ``camera``, ``save_frame``, ``realtime_scale``, and ``csv`` fields.
    """
    parser = argparse.ArgumentParser(
        description="Play one asteroid-landing episode (optionally in Vizard)"
    )
    parser.add_argument(
        "--policy",
        choices=("scripted", "random", "ppo"),
        default="scripted",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/ppo_asteroid_fixed_site_v2.zip",
        help="PPO checkpoint (for --policy ppo)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Live-stream to Vizard (also enables the hub instrument camera HUD)",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Enable Basilisk hub camera (implied by --viz; use alone for headless OpNav)",
    )
    parser.add_argument(
        "--save-frame",
        type=str,
        default="",
        help="If set with --camera, save one RGB frame after a few steps",
    )
    parser.add_argument(
        "--realtime-scale",
        type=float,
        default=0.0,
        help="Extra sleep vs control_dt (0 = no extra sleep / Vizard paces liveStream)",
    )
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Mild start-state randomization (not Scenic)",
    )
    parser.add_argument(
        "--no-auto-point",
        action="store_true",
        help="Disable scripted attitude pointing at the landing site",
    )
    parser.add_argument(
        "--flat-surface",
        action="store_true",
        help="Land on a flat plane instead of the Itokawa heightmap",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=0.0,
        help="Gaussian noise std on agent observation channels (0 = off)",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        choices=("truth", "sensors", "perception"),
        default="truth",
        help="Policy observation mode (reward still uses simulator truth)",
    )
    parser.add_argument(
        "--perception",
        type=str,
        choices=("geometry", "vlm", "auto"),
        default="geometry",
        help="Perception backend for info JSON (vlm needs camera + weights)",
    )
    parser.add_argument(
        "--mission-search",
        action="store_true",
        help="Enable hazard-gated search-then-land mission mode",
    )
    parser.add_argument(
        "--scenic-like",
        action="store_true",
        help="PDF-style randomized start within visibility (no Scenic package)",
    )
    return parser.parse_args()


def main() -> None:
    """Load a policy, run one episode, and write a CSV log.

    When ``--viz`` is set, enables Basilisk liveStream, launches Vizard, and
    always attaches the hub instrument camera (camera HUD).
    ``--camera`` alone enables the same camera in headless OpNav mode.
    When ``--policy ppo`` is set, loads ``--model`` from disk first.
    """
    args = parse_args()
    ensure_dirs()

    model = None
    if args.policy == "ppo":
        if not os.path.isfile(args.model):
            raise FileNotFoundError(
                f"PPO checkpoint not found: {args.model}\n"
                "Train first, or use --policy scripted."
            )
        model = PPO.load(args.model, device="cpu")

    action_fn = make_action_fn(args.policy, model=model, seed=args.seed)
    # --viz always includes the Basilisk instrument camera.
    use_camera = bool(args.camera or args.viz)
    use_viz = bool(args.viz)
    # Camera/Vizard need a live connection; do not reuse the sim across episodes.
    config = LandingEnvConfig(
        seed=args.seed,
        randomize_reset=False,
        light_randomize=args.randomize,
        auto_point=not args.no_auto_point,
        reuse_sim=not (use_viz or use_camera),
        enable_viz=use_viz,
        enable_camera=use_camera,
        use_flat_surface=bool(args.flat_surface),
        obs_noise_std=float(args.obs_noise),
        obs_mode=str(args.obs_mode),
        perception_backend=str(args.perception),
        enable_mission_search=bool(args.mission_search),
        scenic_like_randomize=bool(args.scenic_like),
    )
    env = AsteroidLandingEnv(config=config)

    prefix = "viz" if args.viz else "play"
    csv_path = args.csv or f"logs/{prefix}_{args.policy}_episode.csv"

    if use_viz or use_camera:
        print(f"Starting '{args.policy}' episode with Vizard...")
        print("Vizard should open and connect to tcp://localhost:5556")
        if use_camera:
            print("Hub instrument camera enabled (look for Camera View HUD in Vizard).")
        if use_camera and not use_viz:
            print("Camera-only mode uses Vizard -noDisplay (OpNav image path).")
        if use_viz:
            print("If it waits forever: Direct Communication + Live Streaming in Vizard.")

    if use_camera and args.save_frame:
        import numpy as np

        obs, _info = env.reset()
        frame = None
        for _ in range(8):
            obs, _reward, terminated, truncated, _info = env.step(
                np.array([0.5], dtype=np.float32)
            )
            frame = env.render()
            if frame is not None or terminated or truncated:
                break
        if frame is None:
            raise RuntimeError(
                "Basilisk camera enabled but no image received from Vizard yet. "
                "Confirm Vizard is running and connected (OpNav / Direct Comm)."
            )
        os.makedirs(os.path.dirname(args.save_frame) or ".", exist_ok=True)
        import matplotlib.pyplot as plt

        plt.imsave(args.save_frame, frame)
        print(f"Saved Basilisk instrument-camera frame to {args.save_frame}")

    summary = run_episode(
        args.policy,
        action_fn,
        env,
        csv_path,
        print_every=10 if (use_viz or use_camera) else 1,
        step_sleep_sec=config.control_dt * args.realtime_scale,
    )
    print(
        f"Episode ended: {summary.get('termination_reason')} "
        f"(alt={summary.get('final_altitude')}, "
        f"dist={summary.get('final_distance')}, "
        f"speed={summary.get('final_speed')})"
    )
    print(f"Saved {csv_path}")
    env.close()


if __name__ == "__main__":
    main()
