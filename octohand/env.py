"""Gymnasium environment for the OctoHand tentacle gripper."""

from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import mujoco

from octohand.math3d import LAY_FLAT_X, axis_from_quat, quat_mul, yaw_quat
from octohand.model import (
    CURL_SCALES,
    HAND_Z0,
    MAX_CURL,
    TASKS,
    XY_LIMIT,
    YAW_LIMIT,
    Z_LIMIT,
    HandModel,
    build_model,
)

_BASE_LIMITS = np.array([XY_LIMIT, XY_LIMIT, Z_LIMIT, YAW_LIMIT], dtype=np.float64)


class OctoHandEnv(gym.Env):
    """Soft-gripper manipulation tasks.

    The hand is a position-controlled palm (x, y, z, yaw) with one curl command
    per tentacle. Curl is the pneumatic analogy: every segment of a tentacle
    tracks the same pressure, with the tip bending a little more than the root.

    Tasks
    -----
    block, egg:
        Move the object to a target position in the air.
    pen:
        Move the capsule to a target position and align its long axis.

    Action (shape ``4 + n_tentacles``, each component in ``[-1, 1]``)
    -----------------------------------------------------------------
    0..2 palm x, y, z. 0 holds the home pose (above the table center).
    3    yaw. 0 is no rotation.
    4..  tentacle curl. -1 is fully open, +1 is fully closed.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        task: str = "block",
        n_tentacles: int = 3,
        render_mode: str | None = None,
        max_steps: int = 700,
        frame_skip: int = 5,
        pos_tolerance: float = 0.05,
        align_tolerance: float = 0.80,
        hold_steps: int = 12,
        require_orientation: bool | None = None,
        settle_steps: int = 25,
    ):
        super().__init__()
        self.task = task
        self.spec_task = TASKS[task]
        self.n_tentacles = int(n_tentacles)
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.pos_tolerance = float(pos_tolerance)
        self.align_tolerance = float(align_tolerance)
        self.hold_steps = int(hold_steps)
        self.require_orientation = (
            bool(self.spec_task["orient"])
            if require_orientation is None
            else bool(require_orientation)
        )
        self.settle_steps = int(settle_steps)

        self.hand: HandModel = build_model(task, self.n_tentacles)
        self.model = mujoco.MjModel.from_xml_string(self.hand.xml)
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(self.model.opt.timestep * self.frame_skip)

        self._index_model()
        self.n_actions = 4 + self.n_tentacles
        n_obs = 27 + 4 * self.n_tentacles
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
        )
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = int(round(1.0 / self.control_dt))

        self.goal_pos = np.zeros(3)
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.rest_z = float(self.spec_task["rest_z"])
        self.episode_id = 0
        self.step_count = 0
        self._hold = 0
        self._prev_dist = 0.0
        self._renderer = None
        self._viewer = None

    def _index_model(self) -> None:
        m = self.model
        self._qpos_adr = {}
        self._dof_adr = {}
        for name in self.hand.joint_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._qpos_adr[name] = int(m.jnt_qposadr[jid])
            self._dof_adr[name] = int(m.jnt_dofadr[jid])
        free_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self._obj_qadr = int(m.jnt_qposadr[free_id])
        self._obj_vadr = int(m.jnt_dofadr[free_id])
        self._palm_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "palm")
        self._object_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._object_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object")
        self._tip_sites = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.hand.tip_site_names
        ]
        self._curl_act = []
        self._tentacle_of_geom: dict[int, int] = {}
        for t, (acts, geoms) in enumerate(
            zip(self.hand.curl_actuator_names, self.hand.tentacle_geom_names)
        ):
            ids = [
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in acts
            ]
            self._curl_act.append(ids)
            for name in geoms:
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                self._tentacle_of_geom[int(gid)] = t
        self._base_act = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("base_x", "base_y", "base_z", "base_yaw")
        ]

    def observation_labels(self) -> list[str]:
        labels = [
            "palm_x",
            "palm_y",
            "palm_z",
            "palm_vx",
            "palm_vy",
            "palm_vz",
            "yaw",
        ]
        labels += [f"curl_{i}" for i in range(self.n_tentacles)]
        for i in range(self.n_tentacles):
            labels += [f"tip{i}_x", f"tip{i}_y", f"tip{i}_z"]
        labels += [
            "obj_rel_x",
            "obj_rel_y",
            "obj_rel_z",
            "obj_qw",
            "obj_qx",
            "obj_qy",
            "obj_qz",
            "obj_vx",
            "obj_vy",
            "obj_vz",
            "obj_wx",
            "obj_wy",
            "obj_wz",
            "goal_rel_x",
            "goal_rel_y",
            "goal_rel_z",
            "axis_x",
            "axis_y",
            "axis_z",
            "contacts",
        ]
        return labels

    def home_action(self) -> np.ndarray:
        action = -np.ones(self.n_actions, dtype=np.float32)
        action[:4] = 0.0
        return action

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)

        obj_pos, obj_quat = self._sample_object(options)
        self.data.qpos[self._obj_qadr : self._obj_qadr + 3] = obj_pos
        self.data.qpos[self._obj_qadr + 3 : self._obj_qadr + 7] = obj_quat
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)

        # Park the hand at home. Keep whatever resting pose the object settled into
        # unless the caller asked for an exact placement.
        self.data.qpos[: self._obj_qadr] = 0.0
        if "object_pos" in options:
            self.data.qpos[self._obj_qadr : self._obj_qadr + 3] = np.asarray(
                options["object_pos"], dtype=np.float64
            )
        if "object_quat" in options:
            self.data.qpos[self._obj_qadr + 3 : self._obj_qadr + 7] = np.asarray(
                options["object_quat"], dtype=np.float64
            )
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.goal_pos, self.goal_quat = self._sample_goal(options)
        self.data.mocap_pos[0] = self.goal_pos
        self.data.mocap_quat[0] = self.goal_quat
        mujoco.mj_forward(self.model, self.data)

        self.episode_id += 1
        self.step_count = 0
        self._hold = 0
        self._prev_dist = float(np.linalg.norm(self.object_position - self.goal_pos))
        obs = self._get_obs()
        return obs, self._info(0.0, {})

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.n_actions,):
            raise ValueError(
                f"Expected action shape {(self.n_actions,)}, got {action.shape}."
            )
        action = np.clip(action, -1.0, 1.0)
        self._apply_action(action)
        for _ in range(self.frame_skip):
            self.data.mocap_pos[0] = self.goal_pos
            self.data.mocap_quat[0] = self.goal_quat
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        if not np.isfinite(self.data.qpos).all():
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = self._info(-5.0, {"progress": 0.0, "invalid": 1.0})
            info["success"] = False
            return obs, -5.0, True, False, info

        obj = self.object_position
        dist = float(np.linalg.norm(obj - self.goal_pos))
        progress = self._prev_dist - dist
        self._prev_dist = dist
        tentacles, _geoms = self.contact_stats()
        lifted = bool(obj[2] > self.rest_z + 0.035)
        grasped = tentacles >= 2 and lifted
        align = self.alignment
        placed = dist < self.pos_tolerance and lifted
        if self.require_orientation:
            placed = placed and align >= self.align_tolerance
        self._hold = self._hold + 1 if placed else 0
        success = self._hold >= self.hold_steps
        out = bool(abs(obj[0]) > 0.32 or abs(obj[1]) > 0.32 or obj[2] < 0.0 or obj[2] > 0.55)

        reward = progress
        if tentacles >= 2:
            reward += 0.01
        if lifted:
            reward += 0.015
        if grasped:
            reward += 0.01
        if self.require_orientation:
            reward += 0.02 * (align - 1.0)
        reward -= 0.0005 * float(np.dot(action, action))
        if success:
            reward += 2.0
        if out:
            reward -= 2.0

        terminated = bool(success or out)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        obs = self._get_obs()
        terms = {
            "progress": progress,
            "align": align,
            "lifted": float(lifted),
            "grasped": float(grasped),
        }
        return obs, float(reward), terminated, truncated, self._info(reward, terms)

    def _apply_action(self, action: np.ndarray) -> None:
        base = (action[:4] + 1.0) * 0.5 * (2.0 * _BASE_LIMITS) - _BASE_LIMITS
        for act_id, target in zip(self._base_act, base):
            self.data.ctrl[act_id] = target
        for t in range(self.n_tentacles):
            frac = (action[4 + t] + 1.0) * 0.5
            angle = frac * MAX_CURL
            for act_id, scale in zip(self._curl_act[t], CURL_SCALES):
                self.data.ctrl[act_id] = float(np.clip(angle * scale, 0.0, MAX_CURL * 1.2))

    def _sample_object(self, options: dict) -> tuple[np.ndarray, np.ndarray]:
        if "object_pos" in options and "object_quat" in options:
            return (
                np.asarray(options["object_pos"], dtype=np.float64),
                np.asarray(options["object_quat"], dtype=np.float64),
            )
        rng = self.np_random
        xy = rng.uniform(-0.035, 0.035, size=2) if "object_pos" not in options else None
        yaw = float(rng.uniform(-np.pi, np.pi))
        quat = self._object_quat(yaw)
        if "object_quat" in options:
            quat = np.asarray(options["object_quat"], dtype=np.float64)
        if "object_pos" in options:
            pos = np.asarray(options["object_pos"], dtype=np.float64)
        else:
            pos = np.array([xy[0], xy[1], self.rest_z])
        return pos, quat

    def _sample_goal(self, options: dict) -> tuple[np.ndarray, np.ndarray]:
        obj_xy = self.object_position[:2]
        if "goal_pos" in options:
            goal = np.asarray(options["goal_pos"], dtype=np.float64).copy()
        else:
            rng = self.np_random
            goal_xy = obj_xy.copy()
            for _ in range(40):
                goal_xy = rng.uniform(-0.10, 0.10, size=2)
                if np.linalg.norm(goal_xy - obj_xy) >= 0.08:
                    break
            goal = np.array(
                [goal_xy[0], goal_xy[1], float(rng.uniform(0.10, 0.15))]
            )
        if "goal_quat" in options:
            quat = np.asarray(options["goal_quat"], dtype=np.float64)
        elif self.task == "pen":
            if "goal_yaw" in options and options["goal_yaw"] is not None:
                yaw = float(options["goal_yaw"])
            elif "goal_quat" not in options and options.get("match_orientation"):
                yaw = self._yaw_of_axis(self.object_axis)
            else:
                yaw = float(self.np_random.uniform(-np.pi, np.pi))
            quat = self._object_quat(yaw)
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        return goal, quat

    def _object_quat(self, yaw: float) -> np.ndarray:
        spin = yaw_quat(yaw)
        if self.task == "pen":
            return quat_mul(spin, LAY_FLAT_X)
        return spin

    @staticmethod
    def _yaw_of_axis(axis: np.ndarray) -> float:
        return float(np.arctan2(axis[1], axis[0]))

    @property
    def palm_position(self) -> np.ndarray:
        return self.data.site_xpos[self._palm_site].copy()

    @property
    def palm_velocity(self) -> np.ndarray:
        vx = self.data.qvel[self._dof_adr["base_x"]]
        vy = self.data.qvel[self._dof_adr["base_y"]]
        vz = self.data.qvel[self._dof_adr["base_z"]]
        return np.array([vx, vy, vz], dtype=np.float64)

    @property
    def hand_yaw(self) -> float:
        return float(self.data.qpos[self._qpos_adr["base_yaw"]])

    @property
    def object_position(self) -> np.ndarray:
        return self.data.xpos[self._object_body].copy()

    @property
    def object_quat(self) -> np.ndarray:
        return self.data.xquat[self._object_body].copy()

    @property
    def object_axis(self) -> np.ndarray:
        return axis_from_quat(self.object_quat)

    @property
    def goal_axis(self) -> np.ndarray:
        return axis_from_quat(self.goal_quat)

    @property
    def alignment(self) -> float:
        return float(abs(np.dot(self.object_axis, self.goal_axis)))

    @property
    def curl_fractions(self) -> np.ndarray:
        fracs = np.zeros(self.n_tentacles, dtype=np.float64)
        for t in range(self.n_tentacles):
            angles = [
                self.data.qpos[self._qpos_adr[f"t{t}_s{s}_j"]]
                for s in range(self.hand.n_segments)
            ]
            fracs[t] = float(np.clip(np.mean(angles) / MAX_CURL, 0.0, 1.5))
        return fracs

    @property
    def curl_fraction(self) -> float:
        return float(np.mean(self.curl_fractions))

    def contact_stats(self) -> tuple[int, int]:
        tentacles: set[int] = set()
        geoms = 0
        obj = self._object_geom
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            other = None
            if g1 == obj and g2 in self._tentacle_of_geom:
                other = g2
            elif g2 == obj and g1 in self._tentacle_of_geom:
                other = g1
            if other is not None:
                geoms += 1
                tentacles.add(self._tentacle_of_geom[other])
        return len(tentacles), geoms

    def _get_obs(self) -> np.ndarray:
        palm = self.palm_position
        obj = self.object_position
        tips = []
        for site in self._tip_sites:
            tips.append((self.data.site_xpos[site] - obj) / 0.30)
        lin = np.clip(self.data.qvel[self._obj_vadr : self._obj_vadr + 3], -2.0, 2.0)
        ang = np.clip(self.data.qvel[self._obj_vadr + 3 : self._obj_vadr + 6], -5.0, 5.0)
        tentacles, _geoms = self.contact_stats()
        parts = [
            palm / 0.30,
            np.clip(self.palm_velocity, -1.5, 1.5) / 1.5,
            [self.hand_yaw / np.pi],
            np.clip(self.curl_fractions, 0.0, 1.0),
            np.concatenate(tips) if tips else np.zeros(0),
            (obj - palm) / 0.30,
            self.object_quat,
            lin / 2.0,
            ang / 5.0,
            (self.goal_pos - obj) / 0.30,
            self.goal_axis,
            [tentacles / max(self.n_tentacles, 1)],
        ]
        obs = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Observation length {obs.shape[0]} != {self.observation_space.shape[0]}"
            )
        return obs.astype(np.float32)

    def _info(self, reward: float, terms: dict) -> dict:
        tentacles, geoms = self.contact_stats()
        return {
            "task": self.task,
            "success": bool(self._hold >= self.hold_steps),
            "pos_error": float(np.linalg.norm(self.object_position - self.goal_pos)),
            "align": self.alignment,
            "contacts": tentacles,
            "contact_geoms": geoms,
            "palm_z": float(self.palm_position[2]),
            "object_z": float(self.object_position[2]),
            "curl": self.curl_fraction,
            "goal_pos": self.goal_pos.copy(),
            "reward": float(reward),
            "reward_terms": terms,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        if self.render_mode == "human":
            self._render_human()
            return None
        return None

    def _camera(self):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.distance = 0.85
        cam.azimuth = 135
        cam.elevation = -28
        cam.lookat[:] = np.array([0.0, 0.0, 0.08])
        return cam

    def _render_rgb(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=self._camera())
        return self._renderer.render()

    def _render_human(self) -> None:
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._renderer is not None:
            close = getattr(self._renderer, "close", None)
            if close:
                close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        super().close()
