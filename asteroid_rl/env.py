"""Fixed-site asteroid landing Gymnasium environment backed by Basilisk/MuJoCo.

Builds the Itokawa landing scene, exposes a scalar throttle action, returns
truth-state numerical observations, and computes a shaped reward. Optional
Vizard live-streaming is supported for evaluation playback.

Scope of this module:
    - Fixed target proxy and (by default) fixed initial state
    - No Scenic randomization of the full scenario graph
    - No VLM / camera perception
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
THRUSTER_VIZ_SCALE = 100.0
ASTEROID_VIZ_SCALE = 1000.0
SIM_DT = 0.02
SIM_PROCESS_NAME = "asteroid_rl"
SIM_TASK_NAME = "asteroid_rl"

DEFAULT_TARGET = (0.0, 0.0, -150.0)
DEFAULT_INITIAL_POSITION = (0.0, 0.0, 0.0)
DEFAULT_INITIAL_VELOCITY = (0.0, 0.0, -1.0)


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
        target_position_N: Landing target proxy in the inertial frame, meters.
        success_radius: Max distance for ``safe_landing``, meters.
        success_speed: Max speed for ``safe_landing``, m/s.
        crash_radius: Distance band used with ``crash_speed`` for crashes, meters.
        crash_speed: Speed above which near-target contact counts as crash, m/s.
        escape_radius: Distance at which the episode ends as escaped, meters.
        progress_weight: Reward weight on reduction in distance-to-target.
        speed_weight: Penalty weight on speed magnitude.
        fuel_weight: Penalty weight on squared throttle.
        success_bonus: Terminal reward added on safe landing.
        crash_penalty: Terminal reward subtracted on crash.
        timeout_penalty: Terminal reward subtracted on timeout truncation.
        escape_penalty: Terminal reward subtracted on escape.
        randomize_reset: Master switch enabling curriculum-style randomization.
        random_position_delta: Uniform position jitter half-range, meters.
        random_velocity_delta: Uniform velocity jitter half-range, m/s.
        seed: Default RNG seed used when ``reset`` is not given an explicit seed.
        reuse_sim: If True, soft-reset one Basilisk sim across episodes.
        enable_viz: If True, attach Vizard liveStream when building the sim.
        randomize_initial_distance: Perturb start distance along approach axis.
        initial_distance_delta: Half-range for distance randomization, meters.
        randomize_initial_vertical_velocity: Perturb approach-axis velocity.
        initial_vertical_velocity_delta: Half-range for velocity randomization, m/s.
        randomize_lateral_offset: Perturb start position in the lateral plane.
        lateral_offset_delta: Half-range for each lateral axis, meters.
    """

    max_thrust: float = 275.0
    control_dt: float = 0.25
    time_limit: float = 70.0

    target_position_N: Tuple[float, float, float] = DEFAULT_TARGET

    success_radius: float = 5.0
    success_speed: float = 0.75
    crash_radius: float = 5.0
    crash_speed: float = 2.0
    escape_radius: float = 1000.0

    progress_weight: float = 5.0
    speed_weight: float = 0.05
    fuel_weight: float = 0.01
    success_bonus: float = 100.0
    crash_penalty: float = 100.0
    timeout_penalty: float = 20.0
    escape_penalty: float = 50.0

    randomize_reset: bool = False
    random_position_delta: float = 0.0
    random_velocity_delta: float = 0.0
    seed: Optional[int] = None

    # Reuse one Basilisk sim across episodes (recommended on Windows).
    reuse_sim: bool = True

    # Live Vizard visualization (for evaluation playback, not training).
    enable_viz: bool = False

    randomize_initial_distance: bool = False
    initial_distance_delta: float = 0.0
    randomize_initial_vertical_velocity: bool = False
    initial_vertical_velocity_delta: float = 0.0
    randomize_lateral_offset: bool = False
    lateral_offset_delta: float = 0.0

    def target_array(self) -> np.ndarray:
        """Return the target proxy as a length-3 ``float64`` NumPy array.

        Returns:
            Shape ``(3,)`` array copy of ``target_position_N``.
        """
        return np.asarray(self.target_position_N, dtype=np.float64).reshape(3)


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

    Action is scalar throttle in ``[0, 1]``. Observation is a 5-D truth-state
    vector: ``[altitude, vertical_velocity, distance, speed, previous_throttle]``.

    Attributes:
        metadata: Gymnasium metadata (no render modes).
        config: Active ``LandingEnvConfig``.
        handles: Current ``SimHandles``, or ``None`` before first reset.
        action_space: Box with shape ``(1,)`` for throttle.
        observation_space: Box with shape ``(5,)`` for truth-state features.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[LandingEnvConfig] = None):
        """Create the environment wrapper (does not build Basilisk until reset).

        Args:
            config: Optional env configuration. If ``None``, defaults from
                ``LandingEnvConfig`` are used.
        """
        super().__init__()
        self.config = config or LandingEnvConfig()
        self.handles: Optional[SimHandles] = None
        self._np_random: Optional[np.random.Generator] = None

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
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
            A tuple ``(obs, info)`` where ``obs`` is the 5-D observation and
            ``info`` contains telemetry and zeroed reward-term fields.
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

        # One dynamics tick so recorder/obs are valid at episode start.
        self._write_throttle(0.0)
        self._advance_absolute(SIM_DT)
        self.handles.episode_time_sec = SIM_DT

        obs = self._get_obs()
        info = {
            "sim_time_sec": self.handles.sim_time_sec,
            "termination_reason": None,
            "throttle": 0.0,
            "thrust_N": 0.0,
            "distance_to_target": float(obs[2]),
            "speed": float(obs[3]),
            "vertical_velocity": float(obs[1]),
            "reward_total": 0.0,
            "reward_progress": 0.0,
            "reward_speed_penalty": 0.0,
            "reward_fuel_penalty": 0.0,
            "reward_terminal": 0.0,
            "success": False,
            "crash": False,
            "escape": False,
            "timeout": False,
            "initial_position_N": initial_position.tolist(),
            "initial_velocity_N": initial_velocity.tolist(),
        }
        return obs, info

    def _soft_reset_state(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Reuse an existing sim by rewriting hub kinematics and zeroing thrust.

        Args:
            position: Desired inertial position for the spacecraft hub, meters.
            velocity: Desired inertial velocity for the spacecraft hub, m/s.
        """
        hub = self.handles.scene.getBody(SPACECRAFT_BODY_NAME)
        # Only set position when it differs from the XML default origin.
        if hasattr(hub, "setPosition") and not np.allclose(
            position, DEFAULT_INITIAL_POSITION
        ):
            hub.setPosition(position.tolist())
        elif hasattr(hub, "setPosition"):
            hub.setPosition(list(DEFAULT_INITIAL_POSITION))
        hub.setVelocity(velocity.tolist())
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

        prev_obs = self._get_obs()
        self._write_throttle(throttle)
        self._advance_sim(self.config.control_dt)
        obs = self._get_obs()

        reward, terms = self._compute_reward(prev_obs, obs, throttle)
        terminated, reason = self._check_terminated(obs)
        truncated = bool(self.handles.episode_time_sec >= self.config.time_limit)

        if truncated and not terminated:
            timeout_term = -float(self.config.timeout_penalty)
            terms["reward_terminal"] = float(terms["reward_terminal"]) + timeout_term
            terms["reward_total"] = float(terms["reward_total"]) + timeout_term
            reward = float(terms["reward_total"])
            reason = "timeout"

        info = {
            "sim_time_sec": self.handles.sim_time_sec,
            "termination_reason": reason,
            "throttle": throttle,
            "thrust_N": thrust_N,
            "distance_to_target": float(obs[2]),
            "speed": float(obs[3]),
            "vertical_velocity": float(obs[1]),
            "reward_total": float(terms["reward_total"]),
            "reward_progress": float(terms["reward_progress"]),
            "reward_speed_penalty": float(terms["reward_speed_penalty"]),
            "reward_fuel_penalty": float(terms["reward_fuel_penalty"]),
            "reward_terminal": float(terms["reward_terminal"]),
            "success": reason == "safe_landing",
            "crash": reason == "crash",
            "escape": reason == "escaped",
            "timeout": reason == "timeout" or (truncated and not terminated),
        }

        self.handles.previous_throttle = throttle
        return obs, float(reward), bool(terminated), bool(truncated), info

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

    def _get_obs(self) -> np.ndarray:
        """Build the 5-D truth-state observation from the latest kinematics.

        Returns:
            ``float32`` array
            ``[altitude, vertical_velocity, distance, speed, previous_throttle]``.
            Currently altitude and distance both use distance-to-target.
        """
        r, v = self._get_latest_state()
        target = self.config.target_array()
        rel = r - target
        distance = float(np.linalg.norm(rel))
        speed = float(np.linalg.norm(v))
        if distance > 1e-9:
            vertical_velocity_proxy = float(np.dot(v, rel / distance))
        else:
            vertical_velocity_proxy = 0.0
        prev_throttle = 0.0 if self.handles is None else self.handles.previous_throttle
        return np.array(
            [distance, vertical_velocity_proxy, distance, speed, prev_throttle],
            dtype=np.float32,
        )

    def _compute_reward(
        self, prev_obs: np.ndarray, obs: np.ndarray, throttle: float
    ) -> Tuple[float, dict]:
        """Compute shaped reward and per-term breakdown for one control step.

        Args:
            prev_obs: Observation before the step, shape ``(5,)``.
            obs: Observation after the step, shape ``(5,)``.
            throttle: Applied throttle in ``[0, 1]``.

        Returns:
            Tuple ``(reward_total, terms)`` where ``terms`` maps reward component
            names to floats (progress, speed penalty, fuel penalty, terminal).
        """
        progress = float(prev_obs[2]) - float(obs[2])
        reward_progress = self.config.progress_weight * progress
        reward_speed_penalty = -self.config.speed_weight * float(obs[3])
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
        """Evaluate terminal success / crash / escape conditions.

        Args:
            obs: Current observation, shape ``(5,)``. Uses distance (index 2)
                and speed (index 3).

        Returns:
            Tuple ``(terminated, reason)`` where ``reason`` is one of
            ``"safe_landing"``, ``"crash"``, ``"escaped"``, or ``None``.
        """
        distance = float(obs[2])
        speed = float(obs[3])
        if distance <= self.config.success_radius and speed <= self.config.success_speed:
            return True, "safe_landing"
        if distance <= self.config.crash_radius and speed > self.config.crash_speed:
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
    """Open Vizard and connect it to Basilisk's live TCP stream.

    Args:
        port: TCP port that Basilisk ``vizInterface`` is listening on.
    """
    app = _find_vizard_app()
    address = f"tcp://localhost:{port}"
    if app is None:
        print(
            "Vizard.app not found in /Applications.\n"
            f"Open Vizard manually, choose Direct Communication + Live Streaming,\n"
            f"and connect to {address}"
        )
        return

    # macOS: `open` launches the GUI app; --args forwards Unity CLI flags.
    cmd = ["open", app, "--args", "-directComm", address]
    print(f"Launching Vizard: {' '.join(cmd)}")
    subprocess.Popen(cmd)
    # Give Unity a moment to boot before Basilisk blocks waiting for a client.
    time.sleep(2.5)


def _setup_vizard(scSim, scene, thrust_msg, max_thrust: float):
    """Attach live Vizard visualization and launch the Vizard client.

    Args:
        scSim: Basilisk simulation object to attach viz modules to.
        scene: MuJoCo scene providing spacecraft / asteroid bodies.
        thrust_msg: Scalar thruster command message mirrored into Vizard.
        max_thrust: Nominal max thrust used for Vizard thruster scaling, Newtons.

    Returns:
        Tuple ``(viz, thruster_viz_writer)`` for retention on ``SimHandles``.

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

    # liveStream=True is required for Vizard to show the run as it happens.
    # Without it (and without saveFile), vizInterface is attached but nothing opens.
    viz = vizSupport.enableUnityVisualization(
        scSim,
        SIM_TASK_NAME,
        scene,
        liveStream=True,
    )
    viz.reqPortNumber = "5556"
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

    _launch_vizard_livestream(port=str(viz.reqPortNumber))
    return viz, thruster_viz_writer


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
    if config.enable_viz:
        viz, thruster_viz_writer = _setup_vizard(
            scSim, scene, thrust_msg, config.max_thrust
        )

    scSim.InitializeSimulation()

    hub = scene.getBody(SPACECRAFT_BODY_NAME)
    if hasattr(hub, "setPosition") and not np.allclose(position, DEFAULT_INITIAL_POSITION):
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
        absolute_sim_time_sec=SIM_DT,
        episode_time_sec=SIM_DT,
        previous_throttle=0.0,
        initial_position_N=position,
        initial_velocity_N=velocity,
    )
