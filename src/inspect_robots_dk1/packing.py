"""Canonical 7-D joint packing for a TRLC-DK1 follower arm.

The DK1 follower is a 6-DOF arm plus a gripper. Its LeRobot plugin
(``lerobot_robot_trlc_dk1``) names the motors ``joint_1`` to ``joint_6`` and
``gripper`` and keys observations / actions by ``"<motor>.pos"``; Inspect
Robots, like the rest of this package, works in a flat **7-D** vector. This
module is the *one* place that defines how those 7 numbers map to the named
motors, so the policy (a LeRobot model) and the embodiment (the LeRobot driver)
can never disagree.

Convention (7-D): ``[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6,
gripper]``. The six arm joints are **radians** (the DK1 plugin reports and
commands radians in both of its control modes). The gripper is LeRobot's
normalized **0..1** (0 = open, 1 = closed), which the plugin maps onto the
gripper motor's own 0 to -4.7 rad travel.

This module is pure NumPy with no optional/hardware dependencies (no torch, no
``lerobot``), so it imports and tests anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import StateField, StateSpec

# The DK1 follower motor names, in the plugin's canonical order (see
# lerobot_robot_trlc_dk1/follower.py JOINT_NAMES). Six arm joints then the gripper.
MOTORS: tuple[str, ...] = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)
NUM_JOINTS = 6  # revolute arm joints
GRIPPER_DOF = 1  # one gripper
TOTAL_DIM = NUM_JOINTS + GRIPPER_DOF  # 7-D
GRIPPER_IDX = TOTAL_DIM - 1  # index 6

# LeRobot keys motor positions as "<motor>.pos" in observations and actions.
POS_SUFFIX = ".pos"

# The canonical proprioception key. The vector is one field so
# ``StateSpec.keys == {"joint_pos"}`` stays consistent with the ``state_keys``
# both components declare for compatibility.
STATE_KEY = "joint_pos"

# Radians for the six arm joints, LeRobot's normalized 0..1 for the gripper.
STATE_UNIT = "rad+normalized"

Vec = npt.NDArray[np.float64]


def state_spec() -> StateSpec:
    """Describe the packed motor positions: radians for the arm, 0..1 for the gripper."""
    return StateSpec(fields=(StateField(key=STATE_KEY, shape=(TOTAL_DIM,), unit=STATE_UNIT),))


def motor_keys() -> tuple[str, ...]:
    """The LeRobot ``"<motor>.pos"`` keys, in canonical order."""
    return tuple(f"{m}{POS_SUFFIX}" for m in MOTORS)


def validate_dim(vec: npt.ArrayLike, n: int = TOTAL_DIM) -> Vec:
    """Return ``vec`` as a 1-D float array, raising ``ValueError`` if not length ``n``."""
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(
            f"expected a {n}-D vector, got shape {np.shape(vec)} ({arr.shape[0]} elems)"
        )
    return arr


def to_action_dict(vec: npt.ArrayLike) -> dict[str, float]:
    """Turn a flat 7-D vector into LeRobot's ``{"<motor>.pos": value}`` action dict."""
    arr = validate_dim(vec)
    return {f"{m}{POS_SUFFIX}": float(arr[i]) for i, m in enumerate(MOTORS)}


def from_obs_dict(obs: Mapping[str, Any]) -> Vec:
    """Extract the flat 7-D joint vector from a LeRobot observation/action dict.

    Reads ``"<motor>.pos"`` for each motor in canonical order. Raises ``KeyError``
    if any motor position is missing (a misconfigured driver).
    """
    try:
        return np.asarray([float(obs[f"{m}{POS_SUFFIX}"]) for m in MOTORS], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"observation missing motor position {exc} (need {motor_keys()})") from exc
