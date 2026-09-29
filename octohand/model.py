"""MuJoCo model of a pneumatic tentacle gripper.

Each tentacle is a chain of hinge joints that curl in the radial plane, driven
by one command — the discrete-segment stand-in for a single pressure line in
Festo's OctopusGripper. The palm translates in x, y, z and yaws about z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Tentacles are short on purpose. Longer chains meet each other in the middle
# of the palm before they ever close on the object.
N_SEGMENTS = 3
SEG_LEN = 0.032
PALM_RADIUS = 0.028
PALM_HALF_HEIGHT = 0.010
MOUNT_R = 0.042
HAND_Z0 = 0.22
XY_LIMIT = 0.16
Z_LIMIT = 0.12
YAW_LIMIT = float(np.pi)
MAX_CURL = 0.85
JOINT_LIMIT = 1.15
MIN_TENTACLES = 2
MAX_TENTACLES = 8

# Distal joints travel a little farther so the tentacle cups instead of arcing flat.
CURL_SCALES = np.linspace(0.82, 1.18, N_SEGMENTS)

TENTACLE_COLORS = (
    "0.86 0.32 0.24 1",
    "0.78 0.22 0.34 1",
    "0.93 0.46 0.28 1",
    "0.68 0.20 0.32 1",
    "0.90 0.38 0.36 1",
    "0.72 0.28 0.40 1",
    "0.84 0.30 0.30 1",
    "0.96 0.50 0.32 1",
)

# geom type, size, mass, friction, rgba, rest z (object center above the table)
TASKS: dict[str, dict] = {
    "block": {
        "geom": "box",
        "size": "0.026 0.026 0.026",
        "mass": 0.025,
        "friction": "1.6 0.04 0.002",
        "rgba": "0.62 0.40 0.20 1",
        "rest_z": 0.0265,
        "orient": False,
    },
    "egg": {
        "geom": "ellipsoid",
        "size": "0.024 0.017 0.028",
        "mass": 0.018,
        "friction": "0.85 0.01 0.001",
        "rgba": "0.94 0.91 0.84 1",
        "rest_z": 0.0285,
        "orient": False,
    },
    "pen": {
        "geom": "capsule",
        "size": "0.013 0.036",
        "mass": 0.012,
        "friction": "1.4 0.03 0.002",
        "rgba": "0.14 0.17 0.34 1",
        "rest_z": 0.0135,
        "orient": True,
    },
}


@dataclass(frozen=True)
class HandModel:
    xml: str
    task: str
    n_tentacles: int
    n_segments: int
    joint_names: tuple[str, ...]
    curl_actuator_names: tuple[tuple[str, ...], ...]
    tentacle_geom_names: tuple[tuple[str, ...], ...]
    tip_site_names: tuple[str, ...]


def build_model(task: str = "block", n_tentacles: int = 3) -> HandModel:
    if task not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task {task!r}. Expected one of: {known}.")
    if not MIN_TENTACLES <= int(n_tentacles) <= MAX_TENTACLES:
        raise ValueError(
            f"n_tentacles must be between {MIN_TENTACLES} and {MAX_TENTACLES}."
        )
    spec = TASKS[task]
    n_tentacles = int(n_tentacles)

    joint_names: list[str] = ["base_x", "base_y", "base_z", "base_yaw"]
    curl_actuators: list[tuple[str, ...]] = []
    tentacle_geoms: list[tuple[str, ...]] = []
    tip_sites: list[str] = []
    body_xml: list[str] = []
    actuator_xml: list[str] = [
        f'<position name="base_x" joint="base_x" kp="300" kv="40" '
        f'ctrlrange="-{XY_LIMIT} {XY_LIMIT}" forcerange="-40 40"/>',
        f'<position name="base_y" joint="base_y" kp="300" kv="40" '
        f'ctrlrange="-{XY_LIMIT} {XY_LIMIT}" forcerange="-40 40"/>',
        f'<position name="base_z" joint="base_z" kp="220" kv="36" '
        f'ctrlrange="-{Z_LIMIT} {Z_LIMIT}" forcerange="-18 18"/>',
        f'<position name="base_yaw" joint="base_yaw" kp="6" kv="0.8" '
        f'ctrlrange="-{YAW_LIMIT:.5f} {YAW_LIMIT:.5f}" forcerange="-4 4"/>',
    ]
    excludes: list[tuple[str, str]] = []

    for i in range(n_tentacles):
        theta = 2.0 * np.pi * i / n_tentacles
        axis = (-np.sin(theta), np.cos(theta), 0.0)
        mount = (MOUNT_R * np.cos(theta), MOUNT_R * np.sin(theta), -PALM_HALF_HEIGHT)
        color = TENTACLE_COLORS[i % len(TENTACLE_COLORS)]
        actuators: list[str] = []
        geoms: list[str] = []
        opens: list[str] = []
        for s in range(N_SEGMENTS):
            name = f"t{i}_s{s}"
            radius = 0.0095 - 0.0012 * s
            if s == 0:
                pos = f"{mount[0]:.5f} {mount[1]:.5f} {mount[2]:.5f}"
            else:
                pos = f"0 0 {-SEG_LEN:.5f}"
            site = ""
            if s == N_SEGMENTS - 1:
                site = (
                    f'<site name="tip_{i}" pos="0 0 {-SEG_LEN:.5f}" '
                    f'size="0.004" group="4"/>'
                )
                tip_sites.append(f"tip_{i}")
            opens.append(
                f'<body name="{name}" pos="{pos}" gravcomp="1">'
                f'<joint name="{name}_j" type="hinge" '
                f'axis="{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}" '
                f'range="0 {JOINT_LIMIT}" damping="0.06" armature="0.001" stiffness="0.01"/>'
                f'<geom name="{name}_g" type="capsule" '
                f'fromto="0 0 0 0 0 {-SEG_LEN:.5f}" size="{radius:.4f}" '
                f'rgba="{color}" density="450" '
                f'condim="4" friction="2.2 0.12 0.01" '
                f'contype="2" conaffinity="7" '
                f'solref="0.015 1" solimp="0.92 0.97 0.001"/>'
                f"{site}"
            )
            joint_names.append(f"{name}_j")
            actuators.append(f"{name}_a")
            geoms.append(f"{name}_g")
            actuator_xml.append(
                f'<position name="{name}_a" joint="{name}_j" kp="0.7" kv="0.05" '
                f'ctrlrange="0 {JOINT_LIMIT}" forcerange="-0.18 0.18"/>'
            )
        for a in range(N_SEGMENTS):
            for b in range(a + 1, N_SEGMENTS):
                excludes.append((f"t{i}_s{a}", f"t{i}_s{b}"))
        excludes.append(("hand", f"t{i}_s0"))
        body_xml.append("".join(opens) + ("</body>" * N_SEGMENTS))
        curl_actuators.append(tuple(actuators))
        tentacle_geoms.append(tuple(geoms))

    exclude_xml = "\n".join(
        f'<exclude body1="{a}" body2="{b}"/>' for a, b in excludes
    )
    ghost_rgba = "0.25 0.75 0.40 0.35"
    xml = f"""
<mujoco model="octohand_{task}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"
          cone="elliptic" iterations="40" ls_iterations="20"
          noslip_iterations="5" noslip_tolerance="1e-6"/>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.75 0.75 0.75"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.90 0.88 0.84" rgb2="0.74 0.71 0.66"/>
    <material name="table" texture="grid" texrepeat="5 5" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0.55 0.55 0.1" material="table"
          friction="1.3 0.02 0.001" condim="4" contype="1" conaffinity="1"/>
    <camera name="side" pos="0.42 -0.38 0.34" zaxis="0.42 -0.38 0.22" fovy="48"/>
    <body name="hand" pos="0 0 {HAND_Z0}" gravcomp="1">
      <joint name="base_x" type="slide" axis="1 0 0" range="-{XY_LIMIT} {XY_LIMIT}" damping="12"/>
      <joint name="base_y" type="slide" axis="0 1 0" range="-{XY_LIMIT} {XY_LIMIT}" damping="12"/>
      <joint name="base_z" type="slide" axis="0 0 1" range="-{Z_LIMIT} {Z_LIMIT}" damping="16"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" range="-{YAW_LIMIT:.5f} {YAW_LIMIT:.5f}" damping="0.4"/>
      <geom name="palm" type="cylinder" size="{PALM_RADIUS} {PALM_HALF_HEIGHT}"
            mass="0.05" rgba="0.22 0.26 0.36 1" condim="4"
            friction="1.2 0.02 0.001" contype="1" conaffinity="1"/>
      <site name="palm" pos="0 0 0" size="0.006" group="4"/>
      {''.join(body_xml)}
    </body>
    <body name="object" pos="0 0 {spec['rest_z']}">
      <freejoint name="object_free"/>
      <geom name="object" type="{spec['geom']}" size="{spec['size']}"
            mass="{spec['mass']}" rgba="{spec['rgba']}"
            friction="{spec['friction']}" condim="4"
            contype="1" conaffinity="7"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
    <body name="target" mocap="true" pos="0.08 0 {0.12}">
      <geom name="target" type="{spec['geom']}" size="{spec['size']}"
            rgba="{ghost_rgba}" contype="0" conaffinity="0" group="1"/>
    </body>
  </worldbody>
  <contact>
    {exclude_xml}
  </contact>
  <actuator>
    {''.join(actuator_xml)}
  </actuator>
</mujoco>
""".strip()
    return HandModel(
        xml=xml,
        task=task,
        n_tentacles=n_tentacles,
        n_segments=N_SEGMENTS,
        joint_names=tuple(joint_names),
        curl_actuator_names=tuple(curl_actuators),
        tentacle_geom_names=tuple(tentacle_geoms),
        tip_site_names=tuple(tip_sites),
    )
