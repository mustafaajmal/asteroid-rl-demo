"""Build a Basilisk/MuJoCo sim around a Scenic procedural asteroid mesh.

Closes the Gym↔Scenic gap: each Scenic-sampled rock becomes the actual MuJoCo
collision/visual mesh, not just an altitude query on stock Itokawa.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.simulation import mujoco, svIntegrators
from Basilisk.utilities import SimulationBaseClass, macros

from asteroid_rl.dynamics.gravity import CentralGravity, ConstantGravity


def build_procedural_sim(
    config: Any,
    *,
    craft_position_N: Sequence[float],
    craft_velocity_N: Sequence[float],
    asteroid_position_N: Sequence[float],
    mesh: Any,
    craft_sigma_BN: Optional[Sequence[float]] = None,
    landing_site_N: Optional[Sequence[float]] = None,
) -> Any:
    """Construct ``SimHandles`` whose asteroid body is ``mesh`` at ``asteroid_position_N``.

    Args:
        config: ``LandingEnvConfig`` (thrust, gravity, viz flags).
        craft_position_N: Hub start position, meters.
        craft_velocity_N: Hub start velocity, m/s.
        asteroid_position_N: Asteroid COM / body origin in inertial frame.
        mesh: ``trimesh.Trimesh`` in asteroid-local coordinates.
        craft_sigma_BN: Optional initial MRP.
        landing_site_N: Optional site used only to annotate config (not welded).

    Returns:
        Populated ``SimHandles`` from ``asteroid_rl.environment.gym_env``.
    """
    # Local imports avoid circular import at module load.
    from asteroid_rl.environment.gym_env import (
        ASTEROID_BODY_NAME,
        SIM_DT,
        SIM_PROCESS_NAME,
        SIM_TASK_NAME,
        SPACECRAFT_BODY_NAME,
        THRUSTER_NAME,
        SimHandles,
        default_viz_bin_path,
        resolve_viz_mode,
        _setup_vizard,
    )
    from scenic.simulators.basilisk.asteroid_mesh import (
        write_albedo_texture,
        write_landing_xml,
        write_obj,
    )

    work = Path(tempfile.mkdtemp(prefix="asteroid_rl_procedural_"))
    obj_path = write_obj(mesh, work / "procedural_asteroid.obj")
    tex_path = write_albedo_texture(mesh, work / "procedural_asteroid.jpg")
    xml_path = write_landing_xml(
        xml_path=work / "sat_ast_landing_dynamic.xml",
        mesh_filename=obj_path.name,
        asteroid_pos=asteroid_position_N,
    )

    pos = np.asarray(craft_position_N, dtype=np.float64).reshape(3)
    vel = np.asarray(craft_velocity_N, dtype=np.float64).reshape(3)
    ast = np.asarray(asteroid_position_N, dtype=np.float64).reshape(3)

    # Keep config consistent with the procedural rock for reward/altitude.
    config.asteroid_com_N = (float(ast[0]), float(ast[1]), float(ast[2]))
    if landing_site_N is not None:
        site = np.asarray(landing_site_N, dtype=np.float64).reshape(3)
        config.target_position_N = (float(site[0]), float(site[1]), float(site[2]))
        config.use_flat_surface = False
    else:
        # Top of extents as a crude pad until heightmap site is provided.
        top = float(ast[2] + 0.5 * float(np.asarray(mesh.extents)[2]))
        config.target_position_N = (float(ast[0]), float(ast[1]), top)
        config.use_flat_surface = False

    scSim = SimulationBaseClass.SimBaseClass()
    process = scSim.CreateNewProcess(SIM_PROCESS_NAME)
    task = scSim.CreateNewTask(SIM_TASK_NAME, macros.sec2nano(SIM_DT))
    process.addTask(task)

    scene = mujoco.MJScene.fromFile(str(xml_path), files=[str(obj_path)])
    scSim.AddModelToTask(SIM_TASK_NAME, scene)

    integ = svIntegrators.svIntegratorRKF45(scene)
    integ.setRelativeTolerance(1e-3)
    integ.setAbsoluteTolerance(1e-3)
    scene.setIntegrator(integ)

    if str(getattr(config, "gravity_mode", "constant")).lower() == "central":
        gravity = CentralGravity(
            mu=float(config.gravity_mu),
            mass=float(config.gravity_mass_ref),
            com_N=config.asteroid_com_N,
        )
    else:
        gravity = ConstantGravity(force_N=[0.0, 0.0, -200.0])
    scene.AddModelToDynamicsTask(gravity)
    gravity_site = scene.getBody(SPACECRAFT_BODY_NAME).getOrigin()
    gravity_actuator = scene.addForceActuator("hub_gravity", gravity_site)
    gravity_actuator.forceInMsg.subscribeTo(gravity.forceOutMsg)
    gravity.frameInMsg.subscribeTo(gravity_site.stateOutMsg)

    thrust_msg = messaging.SingleActuatorMsg()
    thrust_msg.write(messaging.SingleActuatorMsgPayload(input=0.0))
    scene.getSingleActuator(THRUSTER_NAME).actuatorInMsg.subscribeTo(thrust_msg)

    state_recorder = (
        scene.getBody(SPACECRAFT_BODY_NAME).getOrigin().stateOutMsg.recorder()
    )
    scSim.AddModelToTask(SIM_TASK_NAME, state_recorder)

    viz = thruster_viz_writer = camera_mod = viz_bin_path = None
    if bool(config.enable_viz or config.enable_camera):
        mode = resolve_viz_mode(config.viz_mode)
        if config.enable_camera:
            mode = "live"
        save_path = config.viz_save_file or default_viz_bin_path("scenic_procedural")
        tex_arg = str(tex_path) if tex_path.is_file() else ""
        viz, thruster_viz_writer, camera_mod, viz_bin_path = _setup_vizard(
            scSim,
            scene,
            thrust_msg,
            config.max_thrust,
            show_gui=bool(config.enable_viz),
            enable_camera=bool(config.enable_camera),
            camera_width=int(config.camera_width),
            camera_height=int(config.camera_height),
            camera_render_rate_sec=float(config.control_dt),
            viz_mode=mode,
            viz_save_file=save_path,
            viz_asteroid_model_path=str(obj_path),
            viz_asteroid_texture_path=tex_arg,
            viz_asteroid_scale=1.0,
        )

    scSim.InitializeSimulation()
    hub = scene.getBody(SPACECRAFT_BODY_NAME)
    hub.setPosition(pos.tolist())
    hub.setVelocity(vel.tolist())
    if craft_sigma_BN is not None and hasattr(hub, "setAttitude"):
        hub.setAttitude(list(np.asarray(craft_sigma_BN, dtype=float).reshape(3)))
        hub.setAttitudeRate([0.0, 0.0, 0.0])

    scSim.ConfigureStopTime(macros.sec2nano(SIM_DT))
    scSim.ExecuteSimulation()

    handles = SimHandles(
        scSim=scSim,
        scene=scene,
        thrust_msg=thrust_msg,
        state_recorder=state_recorder,
        gravity_model=gravity,
        gravity_actuator=gravity_actuator,
        thruster_viz_writer=thruster_viz_writer,
        viz=viz,
        viz_bin_path=viz_bin_path,
        camera_mod=camera_mod,
        config=config,
        absolute_sim_time_sec=float(SIM_DT),
        episode_time_sec=0.0,
        previous_throttle=0.0,
        initial_position_N=pos.copy(),
        initial_velocity_N=vel.copy(),
    )
    # Stash asset paths for debugging / Vizard.
    handles.procedural_asset_dir = str(work)  # type: ignore[attr-defined]
    handles.procedural_obj_path = str(obj_path)  # type: ignore[attr-defined]
    _ = ASTEROID_BODY_NAME  # referenced for clarity / future viz checks
    return handles
