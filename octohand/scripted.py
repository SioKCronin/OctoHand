"""Closed-loop motion sequence for grasp-and-place.

This is the motion terminal from the original notes: move over the object,
lower the hand, pressurize the tentacles, lift, and carry to the goal. It is
a baseline controller, not a learned policy.
"""

from __future__ import annotations

import numpy as np

from octohand.model import HAND_Z0, SEG_LEN, XY_LIMIT, YAW_LIMIT, Z_LIMIT


class ScriptedPolicy:
    def __init__(self):
        self.phase = "seek"
        self._episode = None
        self._cmd = np.zeros(3)
        self._yaw = 0.0
        self._steps = 0

    def act(self, env) -> np.ndarray:
        if self._episode != env.episode_id:
            self._episode = env.episode_id
            self.phase = "seek"
            self._cmd = env.palm_position.copy()
            self._yaw = float(env.hand_yaw)
            self._steps = 0

        obj = env.object_position
        goal = env.goal_pos
        palm = env.palm_position
        contacts, _geoms = env.contact_stats()
        horiz = float(np.linalg.norm(palm[:2] - obj[:2]))
        grasp_z = _grasp_z(env)
        yaw_target = _approach_yaw(env)
        curl = 0.0
        xy_rate, z_rate, yaw_rate = 0.35, 0.10, 1.4

        self._steps += 1
        phase = self.phase
        if phase == "seek":
            xy, z, curl = obj[:2], HAND_Z0, 0.0
            if horiz < 0.012 and abs(_wrap(yaw_target - self._yaw)) < 0.25:
                self.phase = "descend"
                self._steps = 0
        elif phase == "descend":
            xy, z, curl = obj[:2], grasp_z, 0.0
            yaw_rate = 0.8
            # A tentacle on the object will stop the palm short of the table.
            # Close there. Also close if the palm actually arrives.
            stalled = (
                contacts >= 1
                and palm[2] > self._cmd[2] + 0.018
                and self._steps > 20
            )
            arrived = palm[2] <= grasp_z + 0.016
            if horiz > 0.04:
                self.phase = "seek"
                self._steps = 0
            elif contacts >= 2 or arrived or (env.task != "pen" and stalled):
                self._freeze(env)
                self.phase = "close"
                self._steps = 0
        elif phase == "close":
            xy, z = self._cmd[:2], self._cmd[2]
            curl = 0.7 if env.task == "pen" else 0.4
            yaw_target = self._yaw
            xy_rate = z_rate = yaw_rate = 0.0
            if contacts >= 2 and self._steps >= (24 if env.task == "pen" else 14):
                self.phase = "lift"
                self._steps = 0
            elif self._steps > 80:
                self.phase = "seek"
                self._steps = 0
        elif phase == "lift":
            xy, z = self._cmd[:2], min(self._cmd[2] + 0.10, HAND_Z0 + 0.08)
            curl = 0.7 if env.task == "pen" else 0.4
            yaw_target = self._yaw
            xy_rate, yaw_rate, z_rate = 0.0, 0.0, 0.22
            lifted = obj[2] > env.rest_z + 0.028 and contacts >= 2
            if lifted:
                self.phase = "carry"
                self._steps = 0
            elif self._steps > 80 and obj[2] < env.rest_z + 0.025:
                self.phase = "seek"
                self._steps = 0
        else:
            # Carry and hold. Servo the object, and for the pen also its axis.
            xy = palm[:2] + (goal[:2] - obj[:2])
            z = float(np.clip(palm[2] + (goal[2] - obj[2]), grasp_z, HAND_Z0 + Z_LIMIT - 0.02))
            curl = 0.7 if env.task == "pen" else 0.4
            yaw_target = self._yaw + _axis_error(env) if env.task == "pen" else self._yaw
            # Line the pen up before translating much, or the grasp slips.
            xy_rate = 0.06 if env.task == "pen" else 0.10
            if env.task == "pen" and abs(_axis_error(env)) > 0.30:
                xy = palm[:2]
                xy_rate = 0.0
            z_rate, yaw_rate = 0.08, 1.1
            dropped = contacts == 0 and obj[2] < env.rest_z + 0.03
            if dropped:
                self.phase = "seek"
                self._steps = 0
            elif float(np.linalg.norm(obj - goal)) < env.pos_tolerance * 0.8:
                self.phase = "hold"

        self._cmd[:2] = _slew(self._cmd[:2], xy, xy_rate * env.control_dt)
        self._cmd[2] = float(_slew(self._cmd[2], z, z_rate * env.control_dt))
        self._yaw = float(_slew(self._yaw, yaw_target, yaw_rate * env.control_dt))
        return _targets_to_action(self._cmd, self._yaw, curl, env.n_actions)

    def _freeze(self, env) -> None:
        self._cmd = env.palm_position.copy()
        self._yaw = float(env.hand_yaw)


def _grasp_z(env) -> float:
    hang = env.hand.n_segments * SEG_LEN
    if env.task == "pen":
        return hang + 0.016
    return hang + 0.028


def _approach_yaw(env) -> float:
    if env.task != "pen":
        return 0.0
    axis = env.object_axis
    # Offset by a quarter turn so a tentacle is not pointed along the pen.
    return float(np.arctan2(axis[1], axis[0]) + np.pi / 2.0)


def _axis_error(env) -> float:
    """Smallest rotation that lines the pen up with the goal axis."""
    current = float(np.arctan2(env.object_axis[1], env.object_axis[0]))
    desired = float(np.arctan2(env.goal_axis[1], env.goal_axis[0]))
    err = _wrap(desired - current)
    if err > np.pi / 2:
        err -= np.pi
    elif err < -np.pi / 2:
        err += np.pi
    return float(err)


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _slew(current, target, max_step: float):
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = np.clip(target - current, -max_step, max_step)
    return current + delta


def _targets_to_action(palm_xyz, yaw, curl_fraction: float, n_actions: int) -> np.ndarray:
    limits = np.array([XY_LIMIT, XY_LIMIT, Z_LIMIT])
    joints = np.array([palm_xyz[0], palm_xyz[1], palm_xyz[2] - HAND_Z0])
    base = np.clip(joints / limits, -1.0, 1.0)
    action = np.empty(n_actions, dtype=np.float32)
    action[0:3] = base.astype(np.float32)
    action[3] = np.float32(np.clip(yaw / YAW_LIMIT, -1.0, 1.0))
    action[4:] = np.float32(np.clip(curl_fraction, 0.0, 1.0) * 2.0 - 1.0)
    return action
