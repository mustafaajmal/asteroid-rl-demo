"""Run one landing episode under a chosen policy.

Entrypoint for headless or Vizard playback of scripted, random, or PPO control.
Does not train a model; for PPO it only loads an existing checkpoint.

Examples:
    python -m asteroid_rl.cli.play --policy scripted
    python -m asteroid_rl.cli.play --policy random
    python -m asteroid_rl.cli.play --policy ppo --model outputs/ppo_asteroid_fixed_site_v2.zip
    python -m asteroid_rl.cli.play --policy scripted_orbit --orbital
    python -m asteroid_rl.cli.play --policy ppo --orbital --model outputs/best_model_orbital/best_model.zip --viz
"""

from __future__ import annotations

import argparse
import os
import subprocess
from argparse import Namespace

from stable_baselines3 import PPO

from asteroid_rl.camera import launch_vizard_load_file
from asteroid_rl.env import (
    AsteroidLandingEnv,
    LandingEnvConfig,
    _find_vizard_app,
    default_viz_bin_path,
    resolve_viz_mode,
)
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
        choices=(
            "scripted",
            "scripted_orbit",
            "scripted_autonomous",
            "random",
            "random_orbit",
            "ppo",
        ),
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
        "--orbital",
        action="store_true",
        help=(
            "Elliptical-orbit start: central gravity + point/throttle GNC "
            "(overrides obs/action modes)"
        ),
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help=(
            "Full planning-doc stack: scenic∪orbit starts, mission FSM "
            "(search/acquire/divert/upright), upright success gate"
        ),
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help=(
            "Visualize in Vizard. On Windows defaults to save-file playback "
            "(live ZeroMQ often crashes); on macOS uses liveStream."
        ),
    )
    parser.add_argument(
        "--viz-live",
        action="store_true",
        help="Force ZeroMQ liveStream (needed for --camera HUD; fragile on Windows)",
    )
    parser.add_argument(
        "--viz-file",
        action="store_true",
        help="Force save-file Vizard mode (write .bin, open after episode)",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Enable Basilisk hub camera (implies liveStream / --viz-live)",
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
        choices=("truth", "sensors", "perception", "orbital"),
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

    ``--viz`` uses liveStream on macOS and save-file playback on Windows by
    default (Windows Basilisk libzmq often aborts live TCP). ``--viz-live`` /
    ``--camera`` force liveStream. ``--viz-file`` forces ``.bin`` recording.
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

    use_viz = bool(args.viz or args.viz_live or args.viz_file)
    use_camera = bool(args.camera)
    if args.viz_live and args.viz_file:
        raise SystemExit("Choose only one of --viz-live / --viz-file")
    if use_camera:
        viz_mode = "live"
    elif args.viz_live:
        viz_mode = "live"
    elif args.viz_file:
        viz_mode = "file"
    elif args.viz:
        viz_mode = "auto"
    else:
        viz_mode = "auto"
    resolved_mode = resolve_viz_mode(viz_mode)
    # --viz alone used to always enable the camera HUD; keep that for live only.
    if args.viz and resolved_mode == "live" and not args.viz_file:
        use_camera = True

    viz_bin = default_viz_bin_path(f"play_{args.policy}")
    policy = args.policy
    if args.autonomous and policy == "scripted":
        policy = "scripted_autonomous"
    if args.orbital and policy == "scripted":
        policy = "scripted_orbit"
    if args.orbital and policy == "random":
        policy = "random_orbit"
    if args.autonomous and policy == "random":
        policy = "random_orbit"

    action_fn = make_action_fn(policy, model=model, seed=args.seed)

    config = LandingEnvConfig(
        seed=args.seed,
        randomize_reset=False,
        light_randomize=args.randomize,
        auto_point=not args.no_auto_point,
        reuse_sim=not (use_viz or use_camera),
        enable_viz=use_viz,
        enable_camera=use_camera,
        viz_mode=resolved_mode,
        viz_save_file=viz_bin,
        use_flat_surface=bool(args.flat_surface),
        obs_noise_std=float(args.obs_noise),
        obs_mode=str(args.obs_mode),
        perception_backend=str(args.perception),
        enable_mission_search=bool(args.mission_search or args.autonomous),
        scenic_like_randomize=bool(args.scenic_like),
    )
    if args.autonomous:
        config.apply_autonomous_defaults()
        if args.perception:
            config.perception_backend = str(args.perception)
    elif args.orbital:
        config.apply_orbital_defaults()
    env = AsteroidLandingEnv(config=config)

    prefix = "viz" if use_viz else "play"
    csv_path = args.csv or f"logs/{prefix}_{policy}_episode.csv"

    if args.autonomous:
        print(
            f"Autonomous mode: mission FSM + upright gate + "
            f"action_mode={config.action_mode} obs_mode={config.obs_mode} "
            f"start={config.orbit_start_mode}"
        )
    elif args.orbital:
        print(
            f"Orbital mode: central gravity + elliptical reset + "
            f"action_mode={config.action_mode} obs_mode={config.obs_mode}"
        )

    if use_viz or use_camera:
        print(f"Starting '{args.policy}' episode with Vizard ({resolved_mode})...")
        if resolved_mode == "live":
            print("Vizard should open and connect to tcp://localhost:5556")
            print("If it times out: Basilisk liveStream may have crashed (common on Windows).")
            print("Retry with plain --viz (save-file mode) instead of --viz-live.")
            if use_camera:
                print("Hub instrument camera enabled (look for Camera View HUD in Vizard).")
        else:
            print(f"Recording Vizard playback file: {viz_bin}")
            print("Vizard will open after the episode finishes (--loadFile).")

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
        policy,
        action_fn,
        env,
        csv_path,
        print_every=10 if (use_viz or use_camera or args.orbital) else 1,
        step_sleep_sec=config.control_dt * args.realtime_scale,
    )
    print(
        f"Episode ended: {summary.get('termination_reason')} "
        f"(alt={summary.get('final_altitude')}, "
        f"dist={summary.get('final_distance')}, "
        f"speed={summary.get('final_speed')})"
    )
    print(f"Saved {csv_path}")

    recorded = None
    if env.handles is not None:
        recorded = env.handles.viz_bin_path
    env.close()

    if resolved_mode == "file" and use_viz:
        bin_path = recorded or viz_bin
        # Basilisk may nest under _VizFiles when given a non-.bin stem; prefer
        # the path we requested (explicit .bin) or the handle path.
        if not os.path.isfile(bin_path):
            alt = os.path.join(
                os.path.dirname(bin_path),
                "_VizFiles",
                os.path.basename(bin_path),
            )
            if os.path.isfile(alt):
                bin_path = alt
        if os.path.isfile(bin_path):
            launch_vizard_load_file(
                bin_path,
                find_app_fn=_find_vizard_app,
                popen_fn=subprocess.Popen,
            )
            print(
                "If Vizard did not open, use the start screen:\n"
                f"  Basilisk Message File -> Select -> {bin_path}"
            )
        else:
            print(f"Expected Vizard bin missing: {bin_path}")


if __name__ == "__main__":
    main()
