"""
Minimal fixed-site asteroid landing Gymnasium environment.

Scope:
- No Scenic / VLM / camera / full BSK-RL
- Fixed target, truth-state observation, scalar throttle
"""

from __future__ import annotations

import os
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

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VENDOR_DIR = os.path.join(_PKG_ROOT, "vendor_examples")
_EXAMPLES_MUJOCO = os.path.join(_PKG_ROOT, "examples", "mujoco")

if os.path.isfile(os.path.join(_VENDOR_DIR, "sat_ast_landing.xml")):
    XML_PATH = os.path.join(_VENDOR_DIR, "sat_ast_landing.xml")
else:
    XML_PATH = os.path.join(_EXAMPLES_MUJOCO, "sat_ast_landing.xml")

_VENDOR_OBJ = os.path.join(_VENDOR_DIR, "Itokawa", "ItokawaHayabusa.obj")
_DATA_OBJ = os.path.join(_PKG_ROOT, "dataForExamples", "Itokawa", "ItokawaHayabusa.obj")
_EXAMPLES_OBJ = os.path.join(
    _PKG_ROOT, "examples", "dataForExamples", "Itokawa", "ItokawaHayabusa.obj"
)

if os.path.isfile(_VENDOR_OBJ):
    AST_OBJ_PATH = _VENDOR_OBJ
elif os.path.isfile(_DATA_OBJ):
    AST_OBJ_PATH = _DATA_OBJ
else:
    AST_OBJ_PATH = _EXAMPLES_OBJ

SPACECRAFT_BODY_NAME = "hub"
THRUSTER_NAME = "thrust"
SIM_DT = 0.02

DEFAULT_TARGET = (0.0, 0.0, -150.0)
DEFAULT_INITIAL_POSITION = (0.0, 0.0, 0.0)
DEFAULT_INITIAL_VELOCITY = (0.0, 0.0, -1.0)


class ConstantGravity(sysModel.SysModel):
    def __init__(self, force_N: Sequence[float], *args: Any):
        super().__init__(*args)
        self.force_N = force_N
        self.frameInMsg = messaging.SCStatesMsgReader()
        self.forceOutMsg = messaging.ForceAtSiteMsg()

    def UpdateState(self, CurrentSimNanos: int):
        frame: messaging.SCStatesMsgPayload = self.frameInMsg()
        dcm_BN = rbk.MRP2C(frame.sigma_BN)
        force_B = np.dot(dcm_BN, self.force_N)
        payload = messaging.ForceAtSiteMsgPayload(force_S=force_B)
        self.forceOutMsg.write(payload, CurrentSimNanos, self.moduleID)


@dataclass
class LandingEnvConfig:
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

    randomize_initial_distance: bool = False
    initial_distance_delta: float = 0.0
    randomize_initial_vertical_velocity: bool = False
    initial_vertical_velocity_delta: float = 0.0
    randomize_lateral_offset: bool = False
    lateral_offset_delta: float = 0.0

    def target_array(self) -> np.ndarray:
        return np.asarray(self.target_position_N, dtype=np.float64).reshape(3)


LandingConfig = LandingEnvConfig


@dataclass
class SimHandles:
    scSim: Any
    scene: Any
    thrust_msg: Any
    state_recorder: Any
    config: LandingEnvConfig
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
        return self.episode_time_sec


class AsteroidLandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[LandingEnvConfig] = None):
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

    def _soft_reset_state(self, position: np.ndarray, velocity: np.ndarray):
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

    def _write_throttle(self, throttle: float):
        thrust_N = float(throttle * self.config.max_thrust)
        self.handles.thrust_msg.write(
            messaging.SingleActuatorMsgPayload(input=thrust_N)
        )

    def _advance_absolute(self, dt_sec: float):
        self.handles.absolute_sim_time_sec += float(dt_sec)
        self.handles.scSim.ConfigureStopTime(
            macros.sec2nano(self.handles.absolute_sim_time_sec)
        )
        self.handles.scSim.ExecuteSimulation()

    def _advance_sim(self, dt_sec: float):
        dt = float(dt_sec)
        self.handles.episode_time_sec += dt
        self._advance_absolute(dt)

    def _get_latest_state(self) -> Tuple[np.ndarray, np.ndarray]:
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

    def _check_terminated(self, obs: np.ndarray):
        distance = float(obs[2])
        speed = float(obs[3])
        if distance <= self.config.success_radius and speed <= self.config.success_speed:
            return True, "safe_landing"
        if distance <= self.config.crash_radius and speed > self.config.crash_speed:
            return True, "crash"
        if distance >= self.config.escape_radius:
            return True, "escaped"
        return False, None


def build_sim(
    config: LandingEnvConfig,
    initial_position_N: Optional[np.ndarray] = None,
    initial_velocity_N: Optional[np.ndarray] = None,
) -> SimHandles:
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
    process = scSim.CreateNewProcess("asteroid_rl")
    task = scSim.CreateNewTask("asteroid_rl", macros.sec2nano(SIM_DT))
    process.addTask(task)

    scene = mujoco.MJScene.fromFile(XML_PATH, files=[AST_OBJ_PATH])
    scSim.AddModelToTask("asteroid_rl", scene)

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

    thrust_msg = messaging.SingleActuatorMsg()
    thrust_msg.write(messaging.SingleActuatorMsgPayload(input=0.0))
    scene.getSingleActuator(THRUSTER_NAME).actuatorInMsg.subscribeTo(thrust_msg)

    state_recorder = (
        scene.getBody(SPACECRAFT_BODY_NAME).getOrigin().stateOutMsg.recorder()
    )
    scSim.AddModelToTask("asteroid_rl", state_recorder)

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
        absolute_sim_time_sec=SIM_DT,
        episode_time_sec=SIM_DT,
        previous_throttle=0.0,
        initial_position_N=position,
        initial_velocity_N=velocity,
    )
