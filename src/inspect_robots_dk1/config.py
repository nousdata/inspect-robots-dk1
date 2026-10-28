"""Configuration for the DK1 embodiment and the LeRobot policy client.

Both configs are frozen dataclasses whose defaults match a stock TRLC-DK1
follower (the ``dk1_follower`` LeRobot plugin, one wrist camera at 640x480) so
zero-arg construction "just works" for ``.info`` / preflight / ``doctor``. Each
exposes :meth:`from_kwargs` so the adapters accept flat scalar keyword
arguments: this is what lets ``inspect-robots run -P pretrained_path=... -E
port=...`` configure them, since the Inspect Robots CLI only forwards scalar
``key=value`` pairs.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
)

from inspect_robots_dk1 import packing
from inspect_robots_dk1.packing import MOTORS, NUM_JOINTS, STATE_KEY, TOTAL_DIM

_T = TypeVar("_T", bound="_FromKwargs")

# Default action limits, radians. These are the joint_pos_limits the DK1 control
# library (trlc_dk1_control/config.py) enforces in impedance mode, pulled in by
# 0.08 rad so the embodiment's clamp engages before the driver's own clamp:
#   joint_1 +-120 deg, joint_2 -5..180 deg, joint_3 -5..270 deg,
#   joint_4 -111.7..90 deg, joint_5 +-90 deg, joint_6 +-120 deg,
# intersected with the plugin's own pos_vel clips (joint_4 +-100 deg, joint_5
# +-90 deg), so the same box is safe in both control modes. pos_vel clips no
# other joint itself, so this clamp is the only joint limit there. Tighten
# further for your table setup.
_MARGIN = 0.08
JOINT_LOW: tuple[float, ...] = (
    -2.0943951023931953 + _MARGIN,
    -0.08726646259971647 + _MARGIN,
    -0.08726646259971647 + _MARGIN,
    -1.7453292519943295 + _MARGIN,  # joint_4: the pos_vel clip (-100 deg) is tighter
    -1.5707963267948966 + _MARGIN,
    -2.0943951023931953 + _MARGIN,
    0.0,
)
JOINT_HIGH: tuple[float, ...] = (
    2.0943951023931953 - _MARGIN,
    3.141592653589793 - _MARGIN,
    4.71238898038469 - _MARGIN,
    1.5707963267948966 - _MARGIN,
    1.5707963267948966 - _MARGIN,
    2.0943951023931953 - _MARGIN,
    1.0,
)

DEFAULT_CAMERAS: tuple[str, ...] = ("wrist",)

# The two control modes the DK1 plugin implements (DK1FollowerConfig.control_mode).
VALID_CONTROL_MODES: tuple[str, ...] = ("impedance", "pos_vel")

# OpenCV rotations lerobot's Cv2Rotation enum accepts.
VALID_CAMERA_ROTATIONS: tuple[int, ...] = (0, 90, 180, -90)

# Per-step slew limit for the six arm joints, radians per control step. The DK1
# plugin has no slew limit of its own (unlike LeRobot's SO follower), so the
# embodiment applies this one inside step(). 0.06 rad/step at 30 Hz is 1.8 rad/s,
# the cap the DK1's own scripted motions in this project's tooling use.
DEFAULT_MAX_RELATIVE_TARGET = 0.06


class _FromKwargs:
    """Mixin: build a frozen dataclass from flat scalar kwargs (CLI-friendly)."""

    @classmethod
    def from_kwargs(cls: type[_T], **flat: Any) -> _T:
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(flat) - names
        if unknown:
            raise TypeError(f"{cls.__name__} got unexpected config keys: {sorted(unknown)}")
        return cls(**flat)


def parse_camera_devices(spec: str | None) -> dict[str, int | str]:
    """Parse ``"wrist=1,context=0"`` into ``{"wrist": 1, "context": 0}``.

    Values that parse as integers become OpenCV camera indices; anything else is
    kept as a device path string. ``None`` or an empty string means no cameras.
    """
    if not spec or not spec.strip():
        return {}
    devices: dict[str, int | str] = {}
    for item in spec.split(","):
        name, sep, value = item.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not name or not value:
            raise ValueError(
                f"camera_devices entry {item.strip()!r} must look like name=index_or_path"
            )
        if name in devices:
            raise ValueError(f"camera_devices names camera {name!r} twice")
        devices[name] = int(value) if value.lstrip("-").isdigit() else value
    return devices


@dataclass(frozen=True)
class DK1Config(_FromKwargs):
    """Static configuration for a single TRLC-DK1 follower embodiment."""

    # The follower's USB-to-CAN adapter: a serial port (/dev/ttyACM* on Linux,
    # /dev/cu.usbmodem* on macOS). `lerobot-find-port` finds it.
    port: str = "/dev/ttyACM0"
    # "impedance": MIT mode + gravity compensation (250 Hz thread, holds position
    # between commands). "pos_vel": the motors' own position+velocity mode.
    control_mode: str = "impedance"
    max_gripper_torque: float = 1.0  # Nm
    joint_velocity_scaling: float = 0.2  # pos_vel mode only
    disable_torque_on_disconnect: bool = False  # pos_vel mode only (plugin behavior)
    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    # CLI-settable camera wiring for the default (hardware) driver factory:
    # "wrist=1" or "wrist=1,context=/dev/video2". Every name in ``cameras`` needs
    # an entry unless ``camera_configs`` is given instead.
    camera_devices: str | None = None
    camera_fps: int = 30
    camera_rotation: int = 0
    control_hz: float = 30.0
    cam_height: int = 480
    cam_width: int = 640
    joint_low: tuple[float, ...] = JOINT_LOW
    joint_high: tuple[float, ...] = JOINT_HIGH
    home_pose: tuple[float, ...] | None = None
    joints_are_delta: bool = False
    # Radians per step for the arm joints; None disables the embodiment's slew
    # limit (the CLI's default guardrails still apply their own).
    max_relative_target: float | None = DEFAULT_MAX_RELATIVE_TARGET
    # Wait for the arm joints to reach each accepted command before observing,
    # so a chunked policy plans from a converged pose. None disables the wait.
    settle_tolerance: float | None = None
    settle_timeout_s: float = 1.0
    # Timeouts per trial before settling gives up for the rest of that trial.
    settle_timeout_budget: int = 20
    # LeRobot CameraConfig objects keyed by camera name, used only by the default
    # (hardware) driver factory. Opaque to this package; not CLI-settable. Wins
    # over ``camera_devices`` when set.
    camera_configs: Any = None

    def __post_init__(self) -> None:
        if self.control_mode not in VALID_CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {VALID_CONTROL_MODES}, got {self.control_mode!r}"
            )
        for name in ("joint_low", "joint_high"):
            if len(getattr(self, name)) != TOTAL_DIM:
                raise ValueError(f"{name} must have {TOTAL_DIM} entries")
        if self.home_pose is not None:
            if len(self.home_pose) != TOTAL_DIM:
                raise ValueError(f"home_pose must have {TOTAL_DIM} entries")
            if not all(math.isfinite(v) for v in self.home_pose):
                raise ValueError(f"home_pose must contain finite numbers, got {self.home_pose}")
            hp = np.asarray(self.home_pose, dtype=np.float64)
            if np.any(hp < self.low) or np.any(hp > self.high):
                raise ValueError(
                    f"home_pose {self.home_pose} is outside [joint_low, joint_high]; "
                    "the arm cannot reach an out-of-range target"
                )
        if self.max_relative_target is not None and (
            not math.isfinite(self.max_relative_target) or self.max_relative_target <= 0
        ):
            raise ValueError(
                f"max_relative_target must be positive and finite, got {self.max_relative_target}"
            )
        if self.home_pose is not None and self.max_relative_target is None:
            raise ValueError(
                "home_pose without max_relative_target would command a full-speed "
                "jump to the home pose; set DK1Config.max_relative_target (radians "
                "per step) to slew-limit it, or unset home_pose"
            )
        if self.camera_rotation not in VALID_CAMERA_ROTATIONS:
            raise ValueError(
                f"camera_rotation must be one of {VALID_CAMERA_ROTATIONS}, "
                f"got {self.camera_rotation}"
            )
        if self.camera_fps <= 0:
            raise ValueError(f"camera_fps must be positive, got {self.camera_fps}")
        unknown = set(parse_camera_devices(self.camera_devices)) - set(self.cameras)
        if unknown:
            raise ValueError(
                f"camera_devices names cameras {sorted(unknown)} not in cameras={self.cameras}"
            )
        if self.settle_tolerance is not None and (
            not np.isfinite(self.settle_tolerance) or self.settle_tolerance <= 0
        ):
            raise ValueError("settle_tolerance must be finite and > 0")
        if not np.isfinite(self.settle_timeout_s) or self.settle_timeout_s <= 0:
            raise ValueError("settle_timeout_s must be finite and > 0")
        if (
            not isinstance(self.settle_timeout_budget, int)
            or isinstance(self.settle_timeout_budget, bool)
            or self.settle_timeout_budget < 1
        ):
            raise ValueError("settle_timeout_budget must be a positive integer")

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Return the configured lower motor limits as a float64 array."""
        return np.asarray(self.joint_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Return the configured upper motor limits as a float64 array."""
        return np.asarray(self.joint_high, dtype=np.float64)


@dataclass(frozen=True)
class LeRobotPolicyConfig(_FromKwargs):
    """Static configuration for a LeRobot policy loaded from a checkpoint."""

    # Hub id or local directory of a LeRobot checkpoint trained on DK1 data
    # (7-D "<joint>.pos" state/action, the configured cameras). Empty by
    # default: there is no public DK1 checkpoint to point at, so the real
    # inference seam refuses to load until this is set.
    pretrained_path: str = ""
    policy_type: str = "act"  # act, smolvla, pi0, pi05, diffusion, ...
    device: str = "cpu"  # "mps" on Apple silicon, "cuda" on a GPU box
    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    state_key: str = STATE_KEY
    # Max actions consumed per inference: ``act()`` truncates the model's chunk to
    # its first ``chunk_size`` actions (mirrors the async policy server's
    # ``actions_per_chunk``) and advertises it as ``PolicyConfig.action_horizon``.
    chunk_size: int = 50
    cam_height: int = 480
    cam_width: int = 640

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")


def action_semantics(max_relative_target: float | None = None) -> ActionSemantics:
    """The action semantics both components declare: absolute joint targets.

    Compatibility checking compares control_mode + rotation_repr (errors) and
    gripper + frame (warnings), so both sides build from this one function. The
    embodiment passes its slew limit so the declared ``max_step`` (what the CLI's
    default delta-limit guardrail derives from) matches what ``step()`` enforces;
    the gripper gets its full 0..1 travel per step.
    """
    max_step: tuple[float | None, ...] | None = None
    if max_relative_target is not None:
        max_step = (max_relative_target,) * NUM_JOINTS + (1.0,)
    return ActionSemantics(
        control_mode="joint_pos",
        rotation_repr="none",
        gripper="continuous",
        frame="base",
        dim_labels=MOTORS,
        max_step=max_step,
    )


def camera_specs(height: int, width: int, names: tuple[str, ...]) -> tuple[CameraSpec, ...]:
    """Build CameraSpecs for the given names at one resolution (single source of truth)."""
    return tuple(CameraSpec(name=n, height=height, width=width, channels=3) for n in names)


def action_box(
    low: npt.NDArray[np.float64] | None = None,
    high: npt.NDArray[np.float64] | None = None,
    *,
    max_relative_target: float | None = None,
) -> Box:
    """The shared 7-D joint-position action space.

    ``low``/``high`` are optional safety limits (the embodiment supplies them;
    the policy leaves them unset).
    """
    return Box(
        shape=(TOTAL_DIM,),
        low=low,
        high=high,
        semantics=action_semantics(max_relative_target),
    )


def observation_space(height: int, width: int, names: tuple[str, ...]) -> ObservationSpace:
    """Build cameras and packed motor state (radians + normalized gripper)."""
    return ObservationSpace(
        cameras=camera_specs(height, width, names),
        state_keys=frozenset({STATE_KEY}),
        state=packing.state_spec(),
    )
