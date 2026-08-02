"""Fixed-site asteroid landing Gymnasium environment backed by Basilisk/MuJoCo.

Builds the Itokawa landing scene, exposes a scalar throttle action, and computes
a shaped reward from privileged simulator truth. Policy observations can be
truth, onboard-like sensors, or camera-stub perception features
(``LandingEnvConfig.obs_mode``). Optional Vizard live-streaming and a Basilisk
body-fixed instrument camera are supported for evaluation / perception work.

Scope of this module:
    - Fixed surface landing site and (by default) fixed initial state
    - No Scenic randomization of the full scenario graph
    - No VLM reasoning (camera frames only)
    - No full BSK-RL class hierarchy
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel
from Basilisk.simulation import mujoco
from Basilisk.simulation import svIntegrators
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.utilities import vizSupport

from asteroid_rl.observations import (
    encode_agent_observation,
    observation_dim,
    pack_truth_vector,
    validate_obs_mode,
)
from asteroid_rl.perception import build_perception_stub
from asteroid_rl.pointing import apply_pointing, mrp_point_boresight_at
from asteroid_rl.surface import default_landing_site, get_surface_map

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ASSETS_DIR = os.path.join(_REPO_ROOT, "assets")
_EXAMPLES_MUJOCO = os.path.join(_REPO_ROOT, "examples", "mujoco")

_ASSET_XML = os.path.join(_ASSETS_DIR, "sat_ast_landing.xml")
_ASSET_OBJ = os.path.join(_ASSETS_DIR, "Itokawa", "ItokawaHayabusa.obj")
_EXAMPLES_XML = os.path.join(_EXAMPLES_MUJOCO, "sat_ast_landing.xml")
_EXAMPLES_OBJ = os.path.join(
    _REPO_ROOT, "examples", "dataForExamples", "Itokawa", "ItokawaHayabusa.obj"
)

XML_PATH = _ASSET_XML if os.path.isfile(_ASSET_XML) else _EXAMPLES_XML
AST_OBJ_PATH = _ASSET_OBJ if os.path.isfile(_ASSET_OBJ) else _EXAMPLES_OBJ

_TEXTURE_CANDIDATES = [
    os.path.join(_ASSETS_DIR, "Itokawa", "ItokawaGrayscale.jpg"),
    os.path.join(
        _REPO_ROOT, "examples", "dataForExamples", "Itokawa", "ItokawaGrayscale.jpg"
    ),
]
AST_TEXTURE_PATH = next((p for p in _TEXTURE_CANDIDATES if os.path.isfile(p)), "")

SPACECRAFT_BODY_NAME = "hub"
ASTEROID_BODY_NAME = "asteroid"
THRUSTER_NAME = "thrust"
THRUSTER_LOCATION = [0.0, 0.0, -1.0]
THRUSTER_DIRECTION = [0.0, 0.0, 1.0]
# Keep modest: scale 100 filled the instrument FOV with exhaust and hid the asteroid.
THRUSTER_VIZ_SCALE = 8.0
ASTEROID_VIZ_SCALE = 1000.0
SIM_DT = 0.02
SIM_PROCESS_NAME = "asteroid_rl"
SIM_TASK_NAME = "asteroid_rl"

# Surface point under the approach axis (not the asteroid body origin).
DEFAULT_TARGET = default_landing_site()
# Start farther out so the nav camera can frame the asteroid early in the approach.
DEFAULT_INITIAL_POSITION = (0.0, 0.0, 120.0)
DEFAULT_INITIAL_VELOCITY = (0.0, 0.0, -1.5)
# Approximate hub height above terrain when landing legs touch (~box + leg).
NOMINAL_GEAR_CLEARANCE_M = 2.5


class ConstantGravity(sysModel.SysModel):
    """Basilisk SysModel that applies a constant inertial force as "gravity".

    Reads the application-site attitude, rotates ``force_N`` into the site
    frame, and publishes a ``ForceAtSiteMsg`` for an ``MJForceActuator``.

    Attributes:
        force_N: Constant force vector in the inertial frame, Newtons.
        frameInMsg: Reader for the site/spacecraft state message.
        forceOutMsg: Output force message consumed by the force actuator.
    """

    def __init__(self, force_N: Sequence[float], *args: Any):
        """Create the constant-gravity model.

        Args:
            force_N: Length-3 inertial force vector in Newtons.
            *args: Forwarded to ``sysModel.SysModel.__init__``.
        """
        super().__init__(*args)
        self.force_N = force_N
        self.frameInMsg = messaging.SCStatesMsgReader()
        self.forceOutMsg = messaging.ForceAtSiteMsg()

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Recompute and publish the site-frame force for this integrator step.

        Args:
            CurrentSimNanos: Current Basilisk simulation time in nanoseconds.
        """
        frame: messaging.SCStatesMsgPayload = self.frameInMsg()
        dcm_BN = rbk.MRP2C(frame.sigma_BN)
        force_B = np.dot(dcm_BN, self.force_N)
        payload = messaging.ForceAtSiteMsgPayload(force_S=force_B)
        self.forceOutMsg.write(payload, CurrentSimNanos, self.moduleID)


class ThrusterVizMessageWriter(sysModel.SysModel):
    """Publish a ``THROutputMsg`` from the MuJoCo scalar thrust command for Vizard.

    Attributes:
        ModelTag: Thruster name used by Vizard clustering.
        maxThrust: Nominal max thrust used for Vizard scaling, Newtons.
        thrusterLocation: Thruster location in the body frame, meters.
        thrusterDirection: Unit thrust direction in the body frame.
        visualizationScale: Multiplier applied only for Vizard plume visuals.
        thrustInMsg: Reader subscribed to the MuJoCo scalar actuator command.
        thrOutMsg: Thruster output message consumed by Vizard.
    """

    def __init__(
        self,
        thruster_name: str,
        thrust_in_msg: messaging.SingleActuatorMsg,
        max_thrust: float,
        thruster_location: Sequence[float],
        thruster_direction: Sequence[float],
        visualization_scale: float,
        *args: Any,
    ):
        """Create a Vizard thruster message writer.

        Args:
            thruster_name: Name of the MuJoCo actuator represented in Vizard.
            thrust_in_msg: Scalar thrust command message used by MuJoCo.
            max_thrust: Nominal maximum thrust for Vizard scaling, Newtons.
            thruster_location: Thruster location in the attached body frame.
            thruster_direction: Unit thrust direction in the attached body frame.
            visualization_scale: Scale factor applied only to the Vizard thrust.
            *args: Forwarded to ``sysModel.SysModel.__init__``.
        """
        super().__init__(*args)
        self.ModelTag = thruster_name
        self.maxThrust = abs(max_thrust)
        self.thrusterLocation = list(thruster_location)
        self.thrusterDirection = list(thruster_direction)
        self.visualizationScale = visualization_scale
        self.thrustInMsg = messaging.SingleActuatorMsgReader()
        self.thrustInMsg.subscribeTo(thrust_in_msg)
        self.thrOutMsg = messaging.THROutputMsg()

    def Reset(self, CurrentSimNanos: int) -> None:
        """Write the initial thruster visualization payload.

        Args:
            CurrentSimNanos: Current Basilisk simulation time in nanoseconds.
        """
        self._write_thruster_payload(CurrentSimNanos)

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Write the current thruster visualization payload each step.

        Args:
            CurrentSimNanos: Current Basilisk simulation time in nanoseconds.
        """
        self._write_thruster_payload(CurrentSimNanos)

    def _write_thruster_payload(self, CurrentSimNanos: int) -> None:
        """Read the scalar thrust command and publish a ``THROutputMsg``.

        Args:
            CurrentSimNanos: Current Basilisk simulation time in nanoseconds.
        """
        thrust_force = self.thrustInMsg().input
        viz_thrust = self.visualizationScale * thrust_force
        payload = messaging.THROutputMsgPayload()
        payload.maxThrust = self.maxThrust
        payload.thrustForce = viz_thrust
        if self.maxThrust > 0.0:
            payload.thrustFactor = viz_thrust / self.maxThrust
        payload.thrustBlowDownFactor = 1.0
        payload.ispBlowDownFactor = 1.0
        payload.thrusterLocation = self.thrusterLocation
        payload.thrusterDirection = self.thrusterDirection
        payload.thrustForce_B = [
            viz_thrust * component for component in self.thrusterDirection
        ]
        self.thrOutMsg.write(payload, CurrentSimNanos, self.moduleID)


@dataclass
class LandingEnvConfig:
    """Configuration for ``AsteroidLandingEnv``.

    Attributes:
        max_thrust: Peak thruster force in Newtons corresponding to throttle 1.0.
        control_dt: Wall-clock control period between policy actions, seconds.
        time_limit: Episode horizon in seconds before truncation.
        target_position_N: Surface landing site in the inertial frame, meters.
        success_altitude: Max hub altitude above terrain for ``safe_landing``, m.
        min_success_altitude: Min hub altitude for ``safe_landing`` (avoid counting
            deep mesh penetration as success), meters.
        success_speed: Max speed for ``safe_landing``, m/s.
        success_lateral: Max horizontal distance from the site for success, m.
        crash_altitude: Altitude band used with ``crash_speed`` for crashes, m.
        crash_speed: Speed above which a near-surface state counts as crash, m/s.
        penetration_altitude: Altitude below which the episode is a crash, m.
        escape_radius: Distance from the site at which the episode ends escaped, m.
        progress_weight: Reward weight on reduction in distance-to-site.
        altitude_progress_weight: Extra reward weight on altitude reduction.
        speed_weight: Base penalty weight on speed magnitude.
        impact_speed_weight: Extra near-ground penalty on ``speed**2``.
        fuel_weight: Penalty weight on squared throttle.
        success_bonus: Terminal reward added on safe landing.
        crash_penalty: Terminal reward subtracted on crash.
        timeout_penalty: Terminal reward subtracted on timeout truncation.
        escape_penalty: Terminal reward subtracted on escape.
        obs_noise_std: If > 0, Gaussian noise std on agent-facing sensor channels.
        obs_mode: What the policy observes: ``truth`` (privileged), ``sensors``
            (altimeter/rates, no site distance), or ``perception`` (camera stub).
            Reward and termination always use clean simulator truth.
        use_flat_surface: If True, use a flat plane at ``flat_surface_z``.
        flat_surface_z: World ``z`` of the flat landing plane, meters.
        randomize_reset: Master switch enabling curriculum-style randomization.
        random_position_delta: Uniform position jitter half-range, meters.
        random_velocity_delta: Uniform velocity jitter half-range, m/s.
        seed: Default RNG seed used when ``reset`` is not given an explicit seed.
        reuse_sim: If True, soft-reset one Basilisk sim across episodes.
        enable_viz: If True, attach Vizard liveStream when building the sim.
        enable_camera: If True, attach a Basilisk hub instrument camera (needs Vizard).
        camera_width: Instrument-camera image width in pixels.
        camera_height: Instrument-camera image height in pixels.
        auto_point: If True, slew the hub so body -z aims at the landing site
            on reset (scripted pointing outer loop).
        point_every_step: If True, re-slew attitude every control step (can
            destabilize MuJoCo free-joint dynamics; off by default).
        light_randomize: If True, enable a mild start-state randomization bundle
            (distance / lateral / vertical-velocity) without Scenic.
        randomize_initial_distance: Perturb start distance along approach axis.
        initial_distance_delta: Half-range for distance randomization, meters.
        randomize_initial_vertical_velocity: Perturb approach-axis velocity.
        initial_vertical_velocity_delta: Half-range for velocity randomization, m/s.
        randomize_lateral_offset: Perturb start position in the lateral plane.
        lateral_offset_delta: Half-range for each lateral axis, meters.
        success_radius: Deprecated alias retained for older callers; unused when
            surface termination is active.
        crash_radius: Deprecated alias retained for older callers; unused when
            surface termination is active.
    """

    max_thrust: float = 275.0
    control_dt: float = 0.25
    # Longer horizon matches the farther initial approach (~150 m range).
    time_limit: float = 180.0

    target_position_N: Tuple[float, float, float] = DEFAULT_TARGET

    success_altitude: float = 5.0
    min_success_altitude: float = 0.5
    success_speed: float = 0.75
    success_lateral: float = 20.0
    crash_altitude: float = 5.0
    crash_speed: float = 2.0
    penetration_altitude: float = 0.0
    escape_radius: float = 1000.0

    # Backward-compatible unused aliases (body-origin era).
    success_radius: float = 5.0
    crash_radius: float = 5.0

    # Shaped so coasting into a crash is worse than braking (see _compute_reward).
    progress_weight: float = 2.0
    altitude_progress_weight: float = 0.5
    speed_weight: float = 0.08
    impact_speed_weight: float = 0.35
    fuel_weight: float = 0.005
    success_bonus: float = 200.0
    crash_penalty: float = 400.0
    timeout_penalty: float = 50.0
    escape_penalty: float = 100.0

    # Optional Gaussian noise on agent observations (not on reward truth).
    obs_noise_std: float = 0.0
    # Policy observation channel. Reward / termination always use truth.
    obs_mode: str = "truth"

    # Flat-surface benchmark: ignore Itokawa heightmap; land on z = flat_surface_z.
    use_flat_surface: bool = False
    flat_surface_z: float = -30.0

    randomize_reset: bool = False
    random_position_delta: float = 0.0
    random_velocity_delta: float = 0.0
    seed: Optional[int] = None

    # Reuse one Basilisk sim across episodes (recommended on Windows).
    reuse_sim: bool = True

    # Live Vizard visualization (for evaluation playback, not training).
    enable_viz: bool = False

    # Basilisk instrument camera (requires Vizard; body-fixed on the hub).
    enable_camera: bool = False
    camera_width: int = 64
    camera_height: int = 64

    # Scripted pointing + light domain randomization (no Scenic).
    auto_point: bool = True
    point_every_step: bool = False
    light_randomize: bool = False

    randomize_initial_distance: bool = False
    initial_distance_delta: float = 0.0
    randomize_initial_vertical_velocity: bool = False
    initial_vertical_velocity_delta: float = 0.0
    randomize_lateral_offset: bool = False
    lateral_offset_delta: float = 0.0

    def apply_light_randomize_defaults(self) -> "LandingEnvConfig":
        """Enable a mild randomization bundle when ``light_randomize`` is True.

        Returns:
            ``self`` for fluent use.
        """
        if not self.light_randomize:
            return self
        self.randomize_reset = True
        self.randomize_initial_distance = True
        self.initial_distance_delta = max(float(self.initial_distance_delta), 10.0)
        self.randomize_lateral_offset = True
        self.lateral_offset_delta = max(float(self.lateral_offset_delta), 4.0)
        self.randomize_initial_vertical_velocity = True
        self.initial_vertical_velocity_delta = max(
            float(self.initial_vertical_velocity_delta), 0.2
        )
        return self

    def target_array(self) -> np.ndarray:
        """Return the surface landing site as a length-3 ``float64`` array.

        Returns:
            Shape ``(3,)`` site. For flat-surface mode, ``z`` is forced to
            ``flat_surface_z`` while keeping the configured ``x,y``.
        """
        target = np.asarray(self.target_position_N, dtype=np.float64).reshape(3).copy()
        if self.use_flat_surface:
            target[2] = float(self.flat_surface_z)
        return target


LandingConfig = LandingEnvConfig


@dataclass
class SimHandles:
    """Live handles for one constructed Basilisk/MuJoCo simulation.

    Python SysModel references must be retained here for the sim lifetime;
    Basilisk only keeps C++-side pointers and GC of the Python objects will
    segfault on later ``ExecuteSimulation`` calls.

    Attributes:
        scSim: Basilisk ``SimBaseClass`` instance.
        scene: MuJoCo ``MJScene`` dynamics object.
        thrust_msg: Scalar thruster command message writer.
        state_recorder: Recorder on the hub origin state output.
        config: Env config used when this sim was built.
        gravity_model: Retained ``ConstantGravity`` SysModel reference.
        gravity_actuator: Retained MuJoCo force actuator reference.
        thruster_viz_writer: Optional Vizard thruster writer reference.
        viz: Optional Vizard interface object.
        camera_mod: Optional Basilisk ``camera.Camera`` instrument module.
        absolute_sim_time_sec: Monotonic Basilisk stop-time clock, seconds.
        episode_time_sec: Time elapsed within the current episode, seconds.
        previous_throttle: Last applied throttle in ``[0, 1]``.
        initial_position_N: Episode start position in inertial frame, meters.
        initial_velocity_N: Episode start velocity in inertial frame, m/s.
    """

    scSim: Any
    scene: Any
    thrust_msg: Any
    state_recorder: Any
    config: LandingEnvConfig
    # Keep Python SysModel refs alive; Basilisk only holds C++-side pointers.
    gravity_model: Any = None
    gravity_actuator: Any = None
    thruster_viz_writer: Any = None
    viz: Any = None
    camera_mod: Any = None
    absolute_sim_time_sec: float = 0.0
    episode_time_sec: float = 0.0
    previous_throttle: float = 0.0
    initial_position_N: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_INITIAL_POSITION, dtype=np.float64)
    )
    initial_velocity_N: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_INITIAL_VELOCITY, dtype=np.float64)
    )

    @property
    def sim_time_sec(self) -> float:
        """Episode-relative simulation time in seconds.

        Returns:
            Value of ``episode_time_sec``.
        """
        return self.episode_time_sec


class AsteroidLandingEnv(gym.Env):
    """Gymnasium environment for fixed-site asteroid landing control.

    Action is scalar throttle in ``[0, 1]``. Policy observation depends on
    ``config.obs_mode`` (``truth`` / ``sensors`` / ``perception``). Reward and
    episode termination always use clean privileged simulator state.

    Attributes:
        metadata: Gymnasium metadata (``rgb_array`` when camera enabled).
        config: Active ``LandingEnvConfig``.
        handles: Current ``SimHandles``, or ``None`` before first reset.
        surface: Shared asteroid ``SurfaceMap`` for altitude queries.
        action_space: Box with shape ``(1,)`` for throttle.
        observation_space: Box sized for the active ``obs_mode``.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[LandingEnvConfig] = None):
        """Create the environment wrapper (does not build Basilisk until reset).

        Args:
            config: Optional env configuration. If ``None``, defaults from
                ``LandingEnvConfig`` are used. ``enable_camera`` attaches a
                Basilisk body-fixed camera and requires a Vizard connection.
        """
        super().__init__()
        self.config = (config or LandingEnvConfig()).apply_light_randomize_defaults()
        self.config.obs_mode = validate_obs_mode(self.config.obs_mode)
        self.handles: Optional[SimHandles] = None
        self._np_random: Optional[np.random.Generator] = None
        self.surface = get_surface_map()
        self._sigma_BN = np.zeros(3, dtype=np.float64)

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        dim = observation_dim(self.config.obs_mode)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(dim,),
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the lander to an initial state and return the first observation.

        Args:
            seed: Optional episode seed. When provided (or on first reset),
                re-seeds the internal NumPy generator.
            options: Gymnasium options dict (accepted for API compatibility;
                currently unused).

        Returns:
            A tuple ``(obs, info)`` where ``obs`` is the agent observation for
            ``obs_mode`` and ``info`` contains privileged telemetry plus reward
            terms (for logging / scripted baselines).
        """
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self.config.seed
        if seed is not None or self._np_random is None:
            self._np_random = np.random.default_rng(effective_seed)

        initial_position, initial_velocity = self._sample_initial_state()

        if self.handles is None or not self.config.reuse_sim:
            self.handles = build_sim(
                self.config,
                initial_position_N=initial_position,
                initial_velocity_N=initial_velocity,
            )
        else:
            self._soft_reset_state(initial_position, initial_velocity)

        self.handles.episode_time_sec = 0.0
        self.handles.previous_throttle = 0.0
        self.handles.initial_position_N = initial_position
        self.handles.initial_velocity_N = initial_velocity

        if self.config.auto_point:
            self._point_at_target(initial_position)

        # One dynamics tick so recorder/obs are valid at episode start.
        self._write_throttle(0.0)
        self._advance_absolute(SIM_DT)
        self.handles.episode_time_sec = SIM_DT
        self._refresh_attitude_cache()

        truth = self._get_truth_vector()
        info = self._base_info(truth, throttle=0.0, thrust_N=0.0, reason=None)
        obs = self._get_agent_obs(truth, perception=info.get("perception"))
        info.update(
            {
                "reward_total": 0.0,
                "reward_progress": 0.0,
                "reward_speed_penalty": 0.0,
                "reward_fuel_penalty": 0.0,
                "reward_terminal": 0.0,
                "initial_position_N": initial_position.tolist(),
                "initial_velocity_N": initial_velocity.tolist(),
                "obs_mode": self.config.obs_mode,
                "truth_state": truth.copy(),
            }
        )
        return obs, info

    def _soft_reset_state(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Reuse an existing sim by rewriting hub kinematics and zeroing thrust.

        Args:
            position: Desired inertial position for the spacecraft hub, meters.
            velocity: Desired inertial velocity for the spacecraft hub, m/s.
        """
        hub = self.handles.scene.getBody(SPACECRAFT_BODY_NAME)
        if hasattr(hub, "setPosition"):
            hub.setPosition(np.asarray(position, dtype=np.float64).reshape(3).tolist())
        hub.setVelocity(np.asarray(velocity, dtype=np.float64).reshape(3).tolist())
        self.handles.thrust_msg.write(messaging.SingleActuatorMsgPayload(input=0.0))

    def step(self, action):
        """Apply one throttle action and advance the simulation by ``control_dt``.

        Args:
            action: Scalar throttle or length-1 array in ``[0, 1]``. Values
                outside the range are clipped.

        Returns:
            Gymnasium tuple ``(obs, reward, terminated, truncated, info)``.

        Raises:
            RuntimeError: If ``reset`` has not been called yet.
        """
        if self.handles is None:
            raise RuntimeError("Environment must be reset before calling step().")

        throttle = float(
            np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], 0.0, 1.0)
        )
        thrust_N = float(throttle * self.config.max_thrust)

        prev_truth = self._get_truth_vector()
        if self.config.auto_point and self.config.point_every_step:
            r_now, _v_now = self._get_latest_state()
            self._point_at_target(r_now)
        self._write_throttle(throttle)
        self._advance_sim(self.config.control_dt)
        self._refresh_attitude_cache()
        truth = self._get_truth_vector()

        # Reward / termination are privileged; policy obs may omit this state.
        reward, terms = self._compute_reward(prev_truth, truth, throttle)
        terminated, reason = self._check_terminated(truth)
        truncated = bool(self.handles.episode_time_sec >= self.config.time_limit)

        if truncated and not terminated:
            timeout_term = -float(self.config.timeout_penalty)
            terms["reward_terminal"] = float(terms["reward_terminal"]) + timeout_term
            terms["reward_total"] = float(terms["reward_total"]) + timeout_term
            reward = float(terms["reward_total"])
            reason = "timeout"

        info = self._base_info(truth, throttle=throttle, thrust_N=thrust_N, reason=reason)
        obs = self._get_agent_obs(truth, perception=info.get("perception"))
        info.update(
            {
                "reward_total": float(terms["reward_total"]),
                "reward_progress": float(terms["reward_progress"]),
                "reward_speed_penalty": float(terms["reward_speed_penalty"]),
                "reward_fuel_penalty": float(terms["reward_fuel_penalty"]),
                "reward_terminal": float(terms["reward_terminal"]),
                "timeout": reason == "timeout" or (truncated and not terminated),
                "obs_mode": self.config.obs_mode,
                "truth_state": truth.copy(),
            }
        )

        self.handles.previous_throttle = throttle
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Optional[np.ndarray]:
        """Return an RGB frame from the Basilisk hub-mounted instrument camera.

        Requires ``config.enable_camera`` and a live Vizard connection that has
        already delivered at least one OpNav image.

        Returns:
            ``uint8`` array of shape ``(H, W, 3)``, or ``None`` when the camera
            is disabled / not yet producing frames.
        """
        if self.handles is None or self.handles.camera_mod is None:
            return None
        from asteroid_rl.camera import read_camera_rgb

        return read_camera_rgb(
            self.handles.camera_mod,
            width=self.config.camera_width,
            height=self.config.camera_height,
        )

    def close(self) -> None:
        """Release environment resources (no-op placeholder for Gym API)."""
        return None

    def _base_info(
        self,
        truth: np.ndarray,
        *,
        throttle: float,
        thrust_N: float,
        reason: Optional[str],
    ) -> dict:
        """Build the common info dict shared by ``reset`` and ``step``.

        Args:
            truth: Current privileged 5-D truth vector.
            throttle: Applied throttle in ``[0, 1]``.
            thrust_N: Applied thrust in Newtons.
            reason: Termination reason string, or ``None``.

        Returns:
            Info dictionary with privileged telemetry and termination flags.
        """
        perception = self._build_perception(truth)
        return {
            "sim_time_sec": self.handles.sim_time_sec if self.handles else 0.0,
            "termination_reason": reason,
            "throttle": float(throttle),
            "thrust_N": float(thrust_N),
            "altitude": float(truth[0]),
            "distance_to_target": float(truth[2]),
            "speed": float(truth[3]),
            "vertical_velocity": float(truth[1]),
            "success": reason == "safe_landing",
            "crash": reason == "crash",
            "escape": reason == "escaped",
            "timeout": reason == "timeout",
            "perception": perception,
            "target_visible": bool(perception.get("target_visible", False)),
            "hazard_score": float(perception.get("hazard_score", 1.0)),
        }

    def _point_at_target(self, position_N: np.ndarray) -> None:
        """Slew the hub so body -z points at the configured landing site.

        Args:
            position_N: Current hub inertial position, meters.
        """
        if self.handles is None:
            return
        hub = self.handles.scene.getBody(SPACECRAFT_BODY_NAME)
        target = self.config.target_array()
        apply_pointing(hub, position_N, target)
        self._sigma_BN = np.asarray(
            mrp_point_boresight_at(position_N, target), dtype=np.float64
        )

    def _refresh_attitude_cache(self) -> None:
        """Update cached ``sigma_BN`` from the state recorder when available."""
        if self.handles is None:
            return
        rec = self.handles.state_recorder
        try:
            if len(rec.times()) > 0 and hasattr(rec, "sigma_BN"):
                self._sigma_BN = np.array(rec.sigma_BN[-1], dtype=np.float64).reshape(3)
        except Exception:
            pass

    def _build_perception(self, truth: np.ndarray) -> dict:
        """Build the geometry perception stub for the current state.

        Args:
            truth: Current privileged 5-D truth vector (altitude at index 0).

        Returns:
            Perception dictionary matching the planning-document schema.
        """
        if self.handles is None:
            return {
                "target_visible": False,
                "landing_site_box": [0.0, 0.0, 0.0, 0.0],
                "hazard_score": 1.0,
                "progress_assessment": "no sim",
            }
        r, v = self._get_latest_state()
        return build_perception_stub(
            position_N=r,
            velocity_N=v,
            sigma_BN=self._sigma_BN,
            target_N=self.config.target_array(),
            altitude_m=float(truth[0]),
        )

    def _sample_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample (or return default) inertial position and velocity for reset.

        When randomization flags in ``config`` are enabled, perturbs the default
        approach state along the radial/lateral axes and/or with isotropic noise.

        Returns:
            Tuple ``(position_N, velocity_N)`` as shape ``(3,)`` ``float64`` arrays.
        """
        rng = self._np_random or np.random.default_rng(self.config.seed)
        target = self.config.target_array()
        position = np.array(DEFAULT_INITIAL_POSITION, dtype=np.float64)
        velocity = np.array(DEFAULT_INITIAL_VELOCITY, dtype=np.float64)

        approach = position - target
        approach_norm = float(np.linalg.norm(approach))
        radial = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if approach_norm < 1e-9
            else approach / approach_norm
        )

        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(helper, radial))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        lateral_1 = np.cross(radial, helper)
        lateral_1 /= max(float(np.linalg.norm(lateral_1)), 1e-12)
        lateral_2 = np.cross(radial, lateral_1)

        cfg = self.config
        use_random = bool(cfg.randomize_reset) or any(
            [
                cfg.randomize_initial_distance,
                cfg.randomize_initial_vertical_velocity,
                cfg.randomize_lateral_offset,
            ]
        )

        if use_random and cfg.randomize_initial_distance and cfg.initial_distance_delta > 0:
            delta = float(rng.uniform(-cfg.initial_distance_delta, cfg.initial_distance_delta))
            position = position + radial * delta

        if use_random and cfg.randomize_lateral_offset and cfg.lateral_offset_delta > 0:
            dx = float(rng.uniform(-cfg.lateral_offset_delta, cfg.lateral_offset_delta))
            dy = float(rng.uniform(-cfg.lateral_offset_delta, cfg.lateral_offset_delta))
            position = position + lateral_1 * dx + lateral_2 * dy

        if (
            use_random
            and cfg.randomize_initial_vertical_velocity
            and cfg.initial_vertical_velocity_delta > 0
        ):
            dv = float(
                rng.uniform(
                    -cfg.initial_vertical_velocity_delta,
                    cfg.initial_vertical_velocity_delta,
                )
            )
            velocity = velocity - radial * dv

        if use_random and cfg.random_position_delta > 0:
            position = position + rng.uniform(
                -cfg.random_position_delta, cfg.random_position_delta, size=3
            )
        if use_random and cfg.random_velocity_delta > 0:
            velocity = velocity + rng.uniform(
                -cfg.random_velocity_delta, cfg.random_velocity_delta, size=3
            )

        return position, velocity

    def _write_throttle(self, throttle: float) -> None:
        """Write the MuJoCo scalar thruster command for the given throttle.

        Args:
            throttle: Commanded throttle in ``[0, 1]``; converted to Newtons via
                ``config.max_thrust``.
        """
        thrust_N = float(throttle * self.config.max_thrust)
        self.handles.thrust_msg.write(
            messaging.SingleActuatorMsgPayload(input=thrust_N)
        )

    def _advance_absolute(self, dt_sec: float) -> None:
        """Advance the Basilisk stop-time clock and run the integrator.

        Args:
            dt_sec: Duration to advance the absolute simulation clock, seconds.
        """
        self.handles.absolute_sim_time_sec += float(dt_sec)
        self.handles.scSim.ConfigureStopTime(
            macros.sec2nano(self.handles.absolute_sim_time_sec)
        )
        self.handles.scSim.ExecuteSimulation()

    def _advance_sim(self, dt_sec: float) -> None:
        """Advance both episode time and the absolute Basilisk clock.

        Args:
            dt_sec: Control-step duration to add to episode time, seconds.
        """
        dt = float(dt_sec)
        self.handles.episode_time_sec += dt
        self._advance_absolute(dt)

    def _get_latest_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read the most recent inertial position and velocity from the recorder.

        Returns:
            Tuple ``(r_BN_N, v_BN_N)`` as shape ``(3,)`` ``float64`` arrays.

        Raises:
            RuntimeError: If recorder samples cannot be read.
        """
        rec = self.handles.state_recorder
        try:
            if len(rec.times()) == 0:
                raise IndexError("recorder has no samples yet")
            r = np.array(rec.r_BN_N[-1], dtype=np.float64).reshape(3)
            v = np.array(rec.v_BN_N[-1], dtype=np.float64).reshape(3)
        except Exception as exc:
            print(
                "State recorder fields available:",
                [x for x in dir(rec) if not x.startswith("_")],
            )
            raise RuntimeError(
                "Could not read r_BN_N/v_BN_N from state recorder."
            ) from exc
        return r, v

    def _altitude_above_terrain(self, position_N: np.ndarray) -> float:
        """Hub altitude above the active terrain model.

        Args:
            position_N: Hub inertial position, meters.

        Returns:
            Altitude in meters (flat plane or Itokawa heightmap).
        """
        p = np.asarray(position_N, dtype=np.float64).reshape(3)
        if self.config.use_flat_surface:
            return float(p[2] - self.config.flat_surface_z)
        return float(self.surface.altitude(p))

    def _get_truth_vector(self) -> np.ndarray:
        """Build the clean privileged 5-D state used for reward / termination.

        Returns:
            ``float32`` array
            ``[altitude, vertical_velocity, distance_to_site, speed, previous_throttle]``.
        """
        r, v = self._get_latest_state()
        target = self.config.target_array()
        prev_throttle = 0.0 if self.handles is None else self.handles.previous_throttle
        return pack_truth_vector(
            altitude=self._altitude_above_terrain(r),
            vertical_velocity=float(v[2]),
            distance=float(np.linalg.norm(r - target)),
            speed=float(np.linalg.norm(v)),
            previous_throttle=float(prev_throttle),
        )

    def _get_agent_obs(
        self,
        truth: np.ndarray,
        *,
        perception: Optional[dict] = None,
    ) -> np.ndarray:
        """Encode the policy observation for the configured ``obs_mode``.

        Args:
            truth: Clean privileged truth vector.
            perception: Optional perception stub (built if omitted).

        Returns:
            Agent-facing observation for PPO / play.
        """
        if perception is None:
            perception = self._build_perception(truth)
        return encode_agent_observation(
            self.config.obs_mode,
            truth,
            perception=perception,
            noise_std=float(self.config.obs_noise_std),
            rng=self._np_random or np.random.default_rng(self.config.seed),
        )

    def _get_obs(self) -> np.ndarray:
        """Backward-compatible alias returning the agent observation.

        Returns:
            Policy observation for the active ``obs_mode``.
        """
        truth = self._get_truth_vector()
        return self._get_agent_obs(truth)

    def _compute_reward(
        self, prev_obs: np.ndarray, obs: np.ndarray, throttle: float
    ) -> Tuple[float, dict]:
        """Compute shaped reward and per-term breakdown for one control step.

        Progress from falling is down-weighted when speed is already high, and an
        extra near-ground ``speed**2`` penalty discourages coast-into-crash.

        Args:
            prev_obs: Observation before the step, shape ``(5,)``.
            obs: Observation after the step, shape ``(5,)``.
            throttle: Applied throttle in ``[0, 1]``.

        Returns:
            Tuple ``(reward_total, terms)`` where ``terms`` maps reward component
            names to floats (progress, speed penalty, fuel penalty, terminal).
        """
        altitude = float(obs[0])
        speed = float(obs[3])
        site_progress = float(prev_obs[2]) - float(obs[2])
        alt_progress = float(prev_obs[0]) - float(obs[0])
        # Falling fast should not earn full "progress" credit.
        safe_progress_scale = 1.0 / (1.0 + max(0.0, speed - 0.75) ** 2)
        reward_progress = (
            self.config.progress_weight * site_progress
            + self.config.altitude_progress_weight * alt_progress
        ) * safe_progress_scale

        proximity = float(np.clip(1.0 - altitude / 40.0, 0.0, 1.0))
        reward_speed_penalty = (
            -self.config.speed_weight * speed
            - self.config.impact_speed_weight * proximity * (speed ** 2)
        )
        reward_fuel_penalty = -self.config.fuel_weight * (throttle ** 2)
        reward_terminal = 0.0

        _terminated, reason = self._check_terminated(obs)
        if reason == "safe_landing":
            reward_terminal += self.config.success_bonus
        elif reason == "crash":
            reward_terminal -= self.config.crash_penalty
        elif reason == "escaped":
            reward_terminal -= self.config.escape_penalty

        reward_total = (
            reward_progress + reward_speed_penalty + reward_fuel_penalty + reward_terminal
        )
        terms = {
            "reward_total": float(reward_total),
            "reward_progress": float(reward_progress),
            "reward_speed_penalty": float(reward_speed_penalty),
            "reward_fuel_penalty": float(reward_fuel_penalty),
            "reward_terminal": float(reward_terminal),
        }
        return float(reward_total), terms

    def _check_terminated(self, obs: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Evaluate terminal success / crash / escape against the surface site.

        Args:
            obs: Current observation, shape ``(5,)``. Uses altitude (index 0),
                distance-to-site (index 2), and speed (index 3).

        Returns:
            Tuple ``(terminated, reason)`` where ``reason`` is one of
            ``"safe_landing"``, ``"crash"``, ``"escaped"``, or ``None``.
        """
        altitude = float(obs[0])
        distance = float(obs[2])
        speed = float(obs[3])
        target = self.config.target_array()
        # Lateral miss uses current hub x/y vs site x/y.
        if self.handles is not None:
            r, _v = self._get_latest_state()
            lateral = float(np.linalg.norm(r[:2] - target[:2]))
        else:
            lateral = distance

        if altitude < self.config.penetration_altitude:
            return True, "crash"
        if (
            self.config.min_success_altitude
            <= altitude
            <= self.config.success_altitude
            and speed <= self.config.success_speed
            and lateral <= self.config.success_lateral
        ):
            return True, "safe_landing"
        if altitude <= self.config.crash_altitude and speed > self.config.crash_speed:
            return True, "crash"
        if distance >= self.config.escape_radius:
            return True, "escaped"
        return False, None


def _get_body_geom_info(scene: mujoco.MJScene, body_name: str):
    """Look up MuJoCo geom metadata for a named body.

    Args:
        scene: Constructed MuJoCo ``MJScene``.
        body_name: Body name as defined in the scene XML.

    Returns:
        Geom info object for the first geom attached to ``body_name``.

    Raises:
        ValueError: If no geom is found for ``body_name``.
    """
    geom_infos = scene.getGeomInfos()
    for geom_index in range(len(geom_infos)):
        geom_info = geom_infos[geom_index]
        if geom_info.bodyName == body_name:
            return geom_info
    raise ValueError(f"Could not find a MuJoCo geom for body '{body_name}'.")


def _attach_thruster_visualization(viz, spacecraft_name: str, writer) -> None:
    """Wire a thruster writer into Vizard spacecraft thruster HUD data.

    Args:
        viz: Vizard interface object returned by ``enableUnityVisualization``.
        spacecraft_name: Spacecraft name that must match an entry in ``viz.scData``.
        writer: ``ThrusterVizMessageWriter`` whose ``thrOutMsg`` feeds Vizard.

    Raises:
        ValueError: If ``spacecraft_name`` is not present in ``viz.scData``.
    """
    from Basilisk.simulation import vizInterface

    for sc_data_index in range(len(viz.scData)):
        sc_data = viz.scData[sc_data_index]
        if sc_data.spacecraftName != spacecraft_name:
            continue
        thr_info = vizInterface.ThrClusterMap()
        thr_info.thrTag = writer.ModelTag
        thr_info.color = vizSupport.toRGBA255("turquoise")
        sc_data.thrInMsgs = messaging.THROutputMsgInMsgsVector(
            [writer.thrOutMsg.addSubscriber()]
        )
        sc_data.thrInfo = vizInterface.ThrClusterVector([thr_info])
        vizSupport.setActuatorGuiSetting(
            viz,
            spacecraftName=spacecraft_name,
            viewThrusterHUD=True,
        )
        return
    raise ValueError(
        f"Could not find spacecraft '{spacecraft_name}' in Vizard spacecraft data."
    )


def _find_vizard_app() -> Optional[str]:
    """Locate a Vizard.app bundle on common macOS install paths.

    Returns:
        Absolute path to ``Vizard.app`` if found, otherwise ``None``.
    """
    candidates = [
        "/Applications/Vizard.app",
        os.path.expanduser("~/Applications/Vizard.app"),
        "/Applications/Vizard/Vizard.app",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def _launch_vizard_livestream(port: str = "5556") -> None:
    """Open Vizard in visible liveStream mode.

    Args:
        port: TCP port that Basilisk ``vizInterface`` is listening on.
    """
    from asteroid_rl.camera import launch_vizard_for_camera

    launch_vizard_for_camera(
        port=port,
        show_gui=True,
        find_app_fn=_find_vizard_app,
        sleep_fn=time.sleep,
        popen_fn=subprocess.Popen,
    )


def _setup_vizard(
    scSim,
    scene,
    thrust_msg,
    max_thrust: float,
    *,
    show_gui: bool = True,
    enable_camera: bool = False,
    camera_width: int = 64,
    camera_height: int = 64,
    camera_render_rate_sec: float = 0.25,
):
    """Attach Vizard, optional thruster HUD, and optional Basilisk instrument camera.

    Args:
        scSim: Basilisk simulation object to attach viz modules to.
        scene: MuJoCo scene providing spacecraft / asteroid bodies.
        thrust_msg: Scalar thruster command message mirrored into Vizard.
        max_thrust: Nominal max thrust used for Vizard thruster scaling, Newtons.
        show_gui: If True, launch Vizard with a visible window (``-directComm``).
            If False (camera-only), launch headless OpNav mode (``-noDisplay``).
        enable_camera: If True, attach a body-fixed ``camera.Camera`` on the hub
            and register it with ``vizInterface`` for OpNav image requests.
        camera_width: Instrument camera width in pixels.
        camera_height: Instrument camera height in pixels.
        camera_render_rate_sec: Image request period in seconds.

    Returns:
        Tuple ``(viz, thruster_viz_writer, camera_mod)`` for retention on
        ``SimHandles``. ``camera_mod`` is ``None`` when ``enable_camera`` is False.

    Raises:
        RuntimeError: If Basilisk was built without vizInterface support.
    """
    if not vizSupport.vizFound:
        raise RuntimeError(
            "Basilisk vizInterface is unavailable. Install bsk with viz support."
        )

    thruster_viz_writer = ThrusterVizMessageWriter(
        THRUSTER_NAME,
        thrust_msg,
        max_thrust,
        THRUSTER_LOCATION,
        THRUSTER_DIRECTION,
        THRUSTER_VIZ_SCALE,
    )
    scSim.AddModelToTask(SIM_TASK_NAME, thruster_viz_writer)

    # liveStream enables 2-way TCP (needed for instrument-camera image requests).
    viz = vizSupport.enableUnityVisualization(
        scSim,
        SIM_TASK_NAME,
        scene,
        liveStream=True,
    )
    viz.reqPortNumber = "5556"
    viz.noDisplay = not bool(show_gui)
    _attach_thruster_visualization(viz, SPACECRAFT_BODY_NAME, thruster_viz_writer)
    viz.settings.showSpacecraftAsSprites = -1
    viz.settings.ambient = 0.1
    viz.settings.spacecraftShadowBrightness = 0.07

    asteroid_geom = _get_body_geom_info(scene, ASTEROID_BODY_NAME)
    custom_kwargs = dict(
        modelPath=AST_OBJ_PATH,
        simBodiesToModify=[ASTEROID_BODY_NAME],
        scale=[ASTEROID_VIZ_SCALE, ASTEROID_VIZ_SCALE, ASTEROID_VIZ_SCALE],
        offset=list(asteroid_geom.pos),
        rotation=list(rbk.EP2Euler321(list(asteroid_geom.quat))),
        shader=1,
    )
    if AST_TEXTURE_PATH:
        custom_kwargs["customTexturePath"] = AST_TEXTURE_PATH
    vizSupport.createCustomModel(viz, **custom_kwargs)

    camera_mod = None
    if enable_camera:
        from asteroid_rl.camera import create_instrument_camera

        camera_mod = create_instrument_camera(
            parent_name=SPACECRAFT_BODY_NAME,
            width=camera_width,
            height=camera_height,
            render_rate_sec=camera_render_rate_sec,
        )
        # Camera must be on the task before InitializeSimulation so config msgs exist.
        scSim.AddModelToTask(SIM_TASK_NAME, camera_mod)
        viz.addCamMsgToModule(camera_mod.cameraConfigOutMsg)
        camera_mod.imageInMsg.subscribeTo(viz.opnavImageOutMsgs[-1])
        viz.settings.viewCameraViewHUD = 1

    from asteroid_rl.camera import launch_vizard_for_camera

    launch_vizard_for_camera(
        port=str(viz.reqPortNumber),
        show_gui=show_gui,
        find_app_fn=_find_vizard_app,
        sleep_fn=time.sleep,
        popen_fn=subprocess.Popen,
    )
    return viz, thruster_viz_writer, camera_mod


def build_sim(
    config: LandingEnvConfig,
    initial_position_N: Optional[np.ndarray] = None,
    initial_velocity_N: Optional[np.ndarray] = None,
) -> SimHandles:
    """Construct a Basilisk/MuJoCo landing simulation and return live handles.

    Loads the MuJoCo XML and asteroid mesh, wires constant gravity plus a scalar
    thruster, optionally attaches Vizard, initializes the sim, and applies the
    requested initial kinematics.

    Args:
        config: Environment configuration controlling thrust, viz, and related
            options used at construction time.
        initial_position_N: Optional inertial start position, meters. Defaults to
            ``DEFAULT_INITIAL_POSITION`` when ``None``.
        initial_velocity_N: Optional inertial start velocity, m/s. Defaults to
            ``DEFAULT_INITIAL_VELOCITY`` when ``None``.

    Returns:
        Populated ``SimHandles`` with retained Python SysModel references.

    Raises:
        FileNotFoundError: If the MuJoCo XML or asteroid mesh path is missing.
    """
    if not os.path.isfile(XML_PATH):
        raise FileNotFoundError(f"Missing MuJoCo XML: {XML_PATH}")
    if not os.path.isfile(AST_OBJ_PATH):
        raise FileNotFoundError(f"Missing asteroid mesh: {AST_OBJ_PATH}")

    position = (
        np.asarray(initial_position_N, dtype=np.float64).reshape(3)
        if initial_position_N is not None
        else np.array(DEFAULT_INITIAL_POSITION, dtype=np.float64)
    )
    velocity = (
        np.asarray(initial_velocity_N, dtype=np.float64).reshape(3)
        if initial_velocity_N is not None
        else np.array(DEFAULT_INITIAL_VELOCITY, dtype=np.float64)
    )

    scSim = SimulationBaseClass.SimBaseClass()
    process = scSim.CreateNewProcess(SIM_PROCESS_NAME)
    task = scSim.CreateNewTask(SIM_TASK_NAME, macros.sec2nano(SIM_DT))
    process.addTask(task)

    scene = mujoco.MJScene.fromFile(XML_PATH, files=[AST_OBJ_PATH])
    scSim.AddModelToTask(SIM_TASK_NAME, scene)

    integ = svIntegrators.svIntegratorRKF45(scene)
    integ.setRelativeTolerance(1e-3)
    integ.setAbsoluteTolerance(1e-3)
    scene.setIntegrator(integ)

    gravity = ConstantGravity(force_N=[0.0, 0.0, -200.0])
    scene.AddModelToDynamicsTask(gravity)

    gravityApplicationSite = scene.getBody(SPACECRAFT_BODY_NAME).getOrigin()
    gravityActuator = scene.addForceActuator("hub_gravity", gravityApplicationSite)
    gravityActuator.forceInMsg.subscribeTo(gravity.forceOutMsg)
    gravity.frameInMsg.subscribeTo(gravityApplicationSite.stateOutMsg)
    # Must retain these Python objects for the sim lifetime (see SimHandles).

    thrust_msg = messaging.SingleActuatorMsg()
    thrust_msg.write(messaging.SingleActuatorMsgPayload(input=0.0))
    scene.getSingleActuator(THRUSTER_NAME).actuatorInMsg.subscribeTo(thrust_msg)

    state_recorder = (
        scene.getBody(SPACECRAFT_BODY_NAME).getOrigin().stateOutMsg.recorder()
    )
    scSim.AddModelToTask(SIM_TASK_NAME, state_recorder)

    viz = None
    thruster_viz_writer = None
    camera_mod = None
    need_viz = bool(config.enable_viz or config.enable_camera)
    if need_viz:
        viz, thruster_viz_writer, camera_mod = _setup_vizard(
            scSim,
            scene,
            thrust_msg,
            config.max_thrust,
            show_gui=bool(config.enable_viz),
            enable_camera=bool(config.enable_camera),
            camera_width=int(config.camera_width),
            camera_height=int(config.camera_height),
            camera_render_rate_sec=float(config.control_dt),
        )

    scSim.InitializeSimulation()

    hub = scene.getBody(SPACECRAFT_BODY_NAME)
    if hasattr(hub, "setPosition"):
        hub.setPosition(position.tolist())
    hub.setVelocity(velocity.tolist())

    scSim.ConfigureStopTime(macros.sec2nano(SIM_DT))
    scSim.ExecuteSimulation()

    return SimHandles(
        scSim=scSim,
        scene=scene,
        thrust_msg=thrust_msg,
        state_recorder=state_recorder,
        config=config,
        gravity_model=gravity,
        gravity_actuator=gravityActuator,
        thruster_viz_writer=thruster_viz_writer,
        viz=viz,
        camera_mod=camera_mod,
        absolute_sim_time_sec=SIM_DT,
        episode_time_sec=SIM_DT,
        previous_throttle=0.0,
        initial_position_N=position,
        initial_velocity_N=velocity,
    )
