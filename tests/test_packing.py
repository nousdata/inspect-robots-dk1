"""Tests for the pure 7-D DK1 packing module."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_dk1 import packing


def test_constants() -> None:
    assert packing.TOTAL_DIM == 7
    assert packing.NUM_JOINTS == 6
    assert packing.GRIPPER_IDX == 6
    assert packing.STATE_KEY == "joint_pos"
    assert packing.MOTORS[0] == "joint_1"
    assert packing.MOTORS[-1] == "gripper"


def test_state_spec_declares_radians_plus_normalized_gripper() -> None:
    spec = packing.state_spec()
    assert spec.keys == frozenset({"joint_pos"})
    assert spec.fields[0].shape == (7,)
    assert spec.fields[0].unit == "rad+normalized"


def test_motor_keys() -> None:
    assert packing.motor_keys() == (
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "joint_4.pos",
        "joint_5.pos",
        "joint_6.pos",
        "gripper.pos",
    )


def test_validate_dim_accepts_correct_length() -> None:
    out = packing.validate_dim(list(range(7)))
    assert out.shape == (7,)
    assert out.dtype == np.float64


def test_validate_dim_flattens() -> None:
    assert packing.validate_dim(np.zeros((1, 7))).shape == (7,)


def test_validate_dim_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected a 7-D vector"):
        packing.validate_dim(np.zeros(6))


def test_to_action_dict_maps_motors_in_order() -> None:
    out = packing.to_action_dict(np.arange(7, dtype=float))
    assert out == {
        "joint_1.pos": 0.0,
        "joint_2.pos": 1.0,
        "joint_3.pos": 2.0,
        "joint_4.pos": 3.0,
        "joint_5.pos": 4.0,
        "joint_6.pos": 5.0,
        "gripper.pos": 6.0,
    }


def test_to_action_dict_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected a 7-D vector"):
        packing.to_action_dict(np.zeros(6))


def test_from_obs_dict_reads_named_positions() -> None:
    obs = {f"{m}.pos": float(i) for i, m in enumerate(packing.MOTORS)}
    obs["wrist"] = np.zeros((4, 4, 3))  # cameras ignored
    assert np.array_equal(packing.from_obs_dict(obs), np.arange(7, dtype=float))


def test_from_obs_dict_roundtrips_to_action_dict() -> None:
    vec = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    assert np.array_equal(packing.from_obs_dict(packing.to_action_dict(vec)), vec)


def test_from_obs_dict_missing_motor_raises() -> None:
    obs = {f"{m}.pos": 0.0 for m in packing.MOTORS[:-1]}  # drop gripper
    with pytest.raises(KeyError, match="missing motor position"):
        packing.from_obs_dict(obs)
