"""Tests for DK1Config / LeRobotPolicyConfig and the shared space helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from inspect_robots.spaces import CameraSpec

from inspect_robots_dk1.config import (
    DEFAULT_CAMERAS,
    DEFAULT_MAX_RELATIVE_TARGET,
    JOINT_HIGH,
    JOINT_LOW,
    DK1Config,
    LeRobotPolicyConfig,
    action_box,
    action_semantics,
    camera_specs,
    parse_camera_devices,
)


def test_dk1_defaults() -> None:
    cfg = DK1Config()
    assert cfg.port == "/dev/ttyACM0"
    assert cfg.control_mode == "impedance"
    assert cfg.control_hz == 30.0
    assert cfg.cameras == DEFAULT_CAMERAS == ("wrist",)
    assert cfg.camera_devices is None
    assert cfg.low.shape == (7,)
    assert cfg.high.shape == (7,)
    # gripper slot (index 6) bounded [0, 1]; arm joints in radians.
    assert cfg.low[6] == 0.0 and cfg.high[6] == 1.0
    assert cfg.low[0] == pytest.approx(-2.0944 + 0.08, abs=1e-3)
    assert cfg.high[2] == pytest.approx(4.7124 - 0.08, abs=1e-3)
    assert cfg.max_relative_target == DEFAULT_MAX_RELATIVE_TARGET == 0.06
    assert cfg.settle_tolerance is None
    assert cfg.settle_timeout_s == pytest.approx(1.0)
    assert cfg.settle_timeout_budget == 20


def test_default_limits_are_finite_and_ordered() -> None:
    assert len(JOINT_LOW) == len(JOINT_HIGH) == 7
    assert all(math.isfinite(v) for v in JOINT_LOW + JOINT_HIGH)
    assert all(lo < hi for lo, hi in zip(JOINT_LOW, JOINT_HIGH, strict=True))
    # The plugin's own pos_vel clips: joint_4 +-100 deg, joint_5 +-90 deg. Ours sit inside.
    assert JOINT_LOW[3] >= -math.radians(100) and JOINT_HIGH[3] <= math.radians(90)
    assert JOINT_LOW[4] >= -math.radians(90) and JOINT_HIGH[4] <= math.radians(90)
    assert JOINT_LOW[3] == pytest.approx(-math.radians(100) + 0.08)


def test_policy_defaults() -> None:
    cfg = LeRobotPolicyConfig()
    assert cfg.policy_type == "act"
    assert cfg.pretrained_path == ""  # no public DK1 checkpoint to default to
    assert cfg.device == "cpu"
    assert cfg.state_key == "joint_pos"
    assert cfg.cameras == DEFAULT_CAMERAS
    assert cfg.chunk_size == 50


def test_policy_rejects_nonpositive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        LeRobotPolicyConfig(chunk_size=0)


def test_from_kwargs_populates_scalars() -> None:
    cfg = LeRobotPolicyConfig.from_kwargs(pretrained_path="my/ckpt", policy_type="smolvla")
    assert cfg.pretrained_path == "my/ckpt"
    assert cfg.policy_type == "smolvla"


def test_dk1_from_kwargs() -> None:
    cfg = DK1Config.from_kwargs(port="/dev/cu.usbmodem1", control_hz=25.0, camera_devices="wrist=1")
    assert cfg.port == "/dev/cu.usbmodem1"
    assert cfg.control_hz == 25.0
    assert cfg.camera_devices == "wrist=1"


def test_dk1_from_kwargs_accepts_settle_cli_scalars() -> None:
    cfg = DK1Config.from_kwargs(
        settle_tolerance=0.05, settle_timeout_s=2.5, settle_timeout_budget=7
    )
    assert cfg.settle_tolerance == pytest.approx(0.05)
    assert cfg.settle_timeout_s == pytest.approx(2.5)
    assert cfg.settle_timeout_budget == 7


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"settle_tolerance": 0.0}, "settle_tolerance must be finite and > 0"),
        ({"settle_tolerance": -0.01}, "settle_tolerance must be finite and > 0"),
        ({"settle_tolerance": np.inf}, "settle_tolerance must be finite and > 0"),
        ({"settle_tolerance": np.nan}, "settle_tolerance must be finite and > 0"),
        ({"settle_timeout_s": 0.0}, "settle_timeout_s must be finite and > 0"),
        ({"settle_timeout_s": -0.01}, "settle_timeout_s must be finite and > 0"),
        ({"settle_timeout_s": np.inf}, "settle_timeout_s must be finite and > 0"),
        ({"settle_timeout_s": np.nan}, "settle_timeout_s must be finite and > 0"),
        ({"settle_timeout_budget": 0}, "settle_timeout_budget must be a positive integer"),
        ({"settle_timeout_budget": 1.5}, "settle_timeout_budget must be a positive integer"),
        ({"settle_timeout_budget": True}, "settle_timeout_budget must be a positive integer"),
    ],
)
def test_settle_config_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DK1Config(**kwargs)


def test_from_kwargs_rejects_unknown() -> None:
    with pytest.raises(TypeError, match="unexpected config keys"):
        LeRobotPolicyConfig.from_kwargs(nope=1)
    with pytest.raises(TypeError, match="unexpected config keys"):
        DK1Config.from_kwargs(robot_id="x")  # an SO-ARM knob; the DK1 has no calibration id


def test_dk1_rejects_bad_joint_limits() -> None:
    with pytest.raises(ValueError, match="joint_low must have 7 entries"):
        DK1Config(joint_low=(0.0,) * 6)


def test_dk1_rejects_bad_home_pose() -> None:
    with pytest.raises(ValueError, match="home_pose must have 7 entries"):
        DK1Config(home_pose=(0.0,) * 4)
    with pytest.raises(ValueError, match="home_pose must contain finite numbers"):
        DK1Config(home_pose=(float("nan"),) * 7)
    with pytest.raises(ValueError, match="outside \\[joint_low, joint_high\\]"):
        DK1Config(home_pose=(10.0,) * 7, max_relative_target=0.05)


def test_dk1_accepts_valid_home_pose() -> None:
    cfg = DK1Config(home_pose=(0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0), max_relative_target=0.05)
    assert cfg.home_pose is not None and len(cfg.home_pose) == 7


def test_dk1_rejects_home_pose_without_slew_limit() -> None:
    with pytest.raises(ValueError, match="full-speed jump"):
        DK1Config(home_pose=(0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0), max_relative_target=None)


@pytest.mark.parametrize("value", [0.0, -5.0, float("nan"), float("inf")])
def test_dk1_rejects_invalid_max_relative_target(value: float) -> None:
    with pytest.raises(ValueError, match="max_relative_target must be positive and finite"):
        DK1Config(max_relative_target=value)


def test_dk1_rejects_unknown_control_mode() -> None:
    with pytest.raises(ValueError, match="control_mode must be one of"):
        DK1Config(control_mode="torque")


def test_dk1_accepts_pos_vel() -> None:
    assert DK1Config(control_mode="pos_vel").control_mode == "pos_vel"


@pytest.mark.parametrize("rotation", [0, 90, 180, -90])
def test_dk1_accepts_cv2_rotations(rotation: int) -> None:
    assert DK1Config(camera_rotation=rotation).camera_rotation == rotation


def test_dk1_rejects_bad_camera_rotation_and_fps() -> None:
    with pytest.raises(ValueError, match="camera_rotation must be one of"):
        DK1Config(camera_rotation=45)
    with pytest.raises(ValueError, match="camera_fps must be positive"):
        DK1Config(camera_fps=0)


def test_dk1_rejects_camera_devices_for_unknown_cameras() -> None:
    with pytest.raises(ValueError, match="not in cameras"):
        DK1Config(cameras=("wrist",), camera_devices="wrist=1,context=0")


def test_dk1_accepts_camera_devices_subset() -> None:
    cfg = DK1Config(cameras=("wrist", "context"), camera_devices="wrist=1")
    assert parse_camera_devices(cfg.camera_devices) == {"wrist": 1}


@pytest.mark.parametrize("spec", [None, "", "   "])
def test_parse_camera_devices_empty(spec: str | None) -> None:
    assert parse_camera_devices(spec) == {}


def test_parse_camera_devices_indices_and_paths() -> None:
    out = parse_camera_devices("wrist=1, context=/dev/video2")
    assert out == {"wrist": 1, "context": "/dev/video2"}
    assert isinstance(out["wrist"], int)


@pytest.mark.parametrize("spec", ["wrist", "wrist=", "=1", "wrist=1,"])
def test_parse_camera_devices_rejects_malformed(spec: str) -> None:
    with pytest.raises(ValueError, match="must look like name=index_or_path"):
        parse_camera_devices(spec)


def test_parse_camera_devices_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="twice"):
        parse_camera_devices("wrist=1,wrist=2")


def test_action_semantics_labels_every_dim() -> None:
    sem = action_semantics()
    assert sem.control_mode == "joint_pos"
    assert sem.dim_labels == (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    )
    assert sem.max_step is None


def test_action_semantics_declares_slew_limit_as_max_step() -> None:
    sem = action_semantics(0.06)
    assert sem.max_step == (0.06,) * 6 + (1.0,)  # gripper: full travel per step


def test_action_box_carries_limits_and_max_step() -> None:
    cfg = DK1Config()
    box = action_box(cfg.low, cfg.high, max_relative_target=cfg.max_relative_target)
    assert box.shape == (7,)
    assert box.semantics is not None and box.semantics.max_step is not None
    assert np.array_equal(box.low, cfg.low)


def test_camera_specs() -> None:
    specs = camera_specs(480, 640, ("wrist", "context"))
    assert len(specs) == 2
    assert all(isinstance(s, CameraSpec) for s in specs)
    assert specs[0].name == "wrist"
    assert specs[0].height == 480 and specs[0].width == 640
