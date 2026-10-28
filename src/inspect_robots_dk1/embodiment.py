"""``DK1Embodiment``: Inspect Robots embodiment for a TRLC-DK1 follower arm.

Wraps the DK1 LeRobot plugin driver (``lerobot_robot_trlc_dk1.DK1Follower``).
Ported from ``inspect-robots-so101`` and built for real-robot reality:

* **Safety backstop**: every command is clamped to the configured joint limits
  inside :meth:`step`, *independently* of any Inspect Robots ``Approver`` (so
  unclamped model outputs can never reach the motors).
* **Slew limit**: the DK1 plugin has no per-step limit of its own (LeRobot's SO
  follower has ``max_relative_target``; the DK1 driver does not), so
  :meth:`step` also limits each arm joint to ``max_relative_target`` radians
  away from the last observed position. The gripper is not slew-limited.
* **Operator-in-the-loop episode end**: there is no privileged success oracle;
  the operator's end-of-episode keypress returns
  ``StepResult(terminated=True, termination_reason="operator_end")`` and the
  human verdict (with optional grader notes) is captured afterwards by the
  framework's operator prompt and read by judgement-based scorers.
* **Self-paced**: declares ``SELF_PACED`` and sleeps to the control rate inside
  :meth:`step` (the framework does not pace for us). With ``settle_tolerance``
  set, ``control_hz`` is a floor on step duration rather than a fixed rate.
* **No calibration step**: the DK1 plugin reports ``is_calibrated`` as always
  true and keeps no calibration file (the Damiao motors are absolute), so
  there is nothing to guard. In ``pos_vel`` mode the plugin's ``connect()``
  does drive the gripper closed once to zero it; keep clear of the fingers.

The driver is injected (``driver_factory``) and so are the clock / sleep /
operator seams, so the whole embodiment runs in tests with no serial port, no
motors, no cameras, and no stdin. The real driver, a connected ``DK1Follower``
whose ``get_observation`` already returns both motor positions *and* camera
frames, is built in a pragma'd default that only executes on hardware.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.conformance import DeviceSlot
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.types import OPERATOR_END, Action, Observation, StepResult

from inspect_robots_dk1 import packing
from inspect_robots_dk1.config import (
    DK1Config,
    action_box,
    observation_space,
    parse_camera_devices,
)
from inspect_robots_dk1.operator import OperatorIO, default_poll_end

if TYPE_CHECKING:
    from types import TracebackType

ImageMap = Mapping[str, npt.NDArray[np.uint8]]
Vec = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)

#: Spacing between settle polls. The driver observation read can make the loop
#: read-bound on hardware, so this is a floor rather than a guaranteed rate.
_SETTLE_POLL_S = 0.01

#: Revolute-joint slots settling and slew-limiting apply to. The gripper is
#: excluded because one closing on an object may never reach its target.
_ARM_SLOTS = np.arange(packing.NUM_JOINTS, dtype=np.intp)

_PLUGIN_INSTALL = (
    'uv pip install "lerobot_robot_trlc_dk1 @ git+https://github.com/robot-learning-co/trlc-dk1"'
)

#: Free-form operating notes the LLM agent policy injects into its prompt.
DOCS = """\
TRLC-DK1 follower: 6 revolute joints (joint_1..joint_6, radians, absolute
targets) plus a gripper (0 = fully open, 1 = fully closed). joint_1 is the base
yaw, joint_2 the shoulder pitch, joint_3 the elbow, joint_4..joint_6 the wrist
(roll, pitch, roll). The arm ships with a fisheye wrist camera ("wrist").
Commands are slew-limited per step; hold still by repeating the current
joint_pos. Keep a hand on the e-stop."""


@runtime_checkable
class DK1Driver(Protocol):
    """The minimal LeRobot-robot surface the embodiment needs.

    Satisfied directly by ``lerobot_robot_trlc_dk1.DK1Follower`` (and any other
    LeRobot ``Robot`` with the same motor names): observations are dicts of
    ``"<motor>.pos"`` floats plus camera frames keyed by camera name; actions
    are dicts of ``"<motor>.pos"``.
    """

    def get_observation(self) -> Mapping[str, Any]:
        """Return motor positions and camera frames from the connected follower."""
        ...

    def send_action(self, action: Mapping[str, float]) -> Mapping[str, Any]:
        """Command motor positions keyed by LeRobot's ``"<motor>.pos"`` convention.

        Must echo the accepted action back with every motor key (the DK1 plugin
        echoes the goal it sent); ``_send`` parses the echo as the settle
        target, so a partial mapping raises ``KeyError``.
        """
        ...

    def disconnect(self) -> None:
        """Release the follower's hardware resources."""
        ...


DriverFactory = Callable[[DK1Config], DK1Driver]
CameraConfigFactory = Callable[..., Any]


def _default_make_camera_config(**kw: Any) -> Any:  # pragma: no cover - real lerobot
    from pathlib import Path

    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.opencv import OpenCVCameraConfig

    device = kw.pop("index_or_path")
    return OpenCVCameraConfig(
        index_or_path=device if isinstance(device, int) else Path(device),
        rotation=Cv2Rotation(kw.pop("rotation")),
        **kw,
    )


def build_camera_configs(cfg: DK1Config, make: CameraConfigFactory | None = None) -> dict[str, Any]:
    """LeRobot camera configs for the driver, from ``camera_configs`` or ``camera_devices``.

    An explicit ``camera_configs`` mapping wins. Otherwise every configured
    camera name must have a ``camera_devices`` entry, which becomes an OpenCV
    camera at the configured resolution, fps, and rotation. ``make`` is the
    config constructor seam (the real one builds ``OpenCVCameraConfig``).
    """
    if cfg.camera_configs is not None:
        return dict(cfg.camera_configs)
    devices = parse_camera_devices(cfg.camera_devices)
    missing = [name for name in cfg.cameras if name not in devices]
    if missing:
        raise ValueError(
            f"no camera device configured for {missing}; pass "
            f"camera_devices='{','.join(f'{m}=<index>' for m in missing)}' "
            "(an OpenCV index or path per camera) or DK1Config.camera_configs"
        )
    make_fn: CameraConfigFactory = make if make is not None else _default_make_camera_config
    return {
        name: make_fn(
            index_or_path=devices[name],
            width=cfg.cam_width,
            height=cfg.cam_height,
            fps=cfg.camera_fps,
            rotation=cfg.camera_rotation,
        )
        for name in cfg.cameras
    }


def _default_driver_factory(cfg: DK1Config) -> DK1Driver:  # pragma: no cover - real hardware
    from lerobot_robot_trlc_dk1 import DK1Follower, DK1FollowerConfig

    robot = DK1Follower(
        DK1FollowerConfig(
            port=cfg.port,
            control_mode=cfg.control_mode,
            max_gripper_torque=cfg.max_gripper_torque,
            disable_torque_on_disconnect=cfg.disable_torque_on_disconnect,
            cameras=build_camera_configs(cfg),
            joint_velocity_scaling=cfg.joint_velocity_scaling,
        )
    )
    # DK1Follower.connect() takes no calibrate flag: the plugin has no
    # calibration file. In pos_vel mode it zeroes the gripper by closing it.
    robot.connect()
    return cast(DK1Driver, robot)


class DK1Embodiment:
    """Inspect Robots embodiment for a single TRLC-DK1 follower (joint-position control)."""

    #: Device-shaped constructor arguments for `inspect-robots setup` / `doctor`.
    DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = (
        DeviceSlot(arg="port", kind="serial", label="DK1 follower USB-CAN serial port"),
    )
    #: Imports package metadata cannot express; `doctor` checks them without importing.
    RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
        "lerobot": 'uv pip install "inspect-robots-dk1[lerobot]"',
        # lerobot>=0.6 gates lerobot.async_inference (the policy's obs helper) on grpcio.
        "grpc": 'uv pip install "lerobot[async]"',
        "lerobot_robot_trlc_dk1": _PLUGIN_INSTALL,
        "trlc_dk1_control": _PLUGIN_INSTALL,
    }

    def __init__(
        self,
        config: DK1Config | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else DK1Config.from_kwargs(**flat)
        self._driver_factory: DriverFactory = driver_factory or _default_driver_factory
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end: Callable[[], bool] = poll_end or default_poll_end
        self._sleep: Callable[[float], None] = sleep_fn or time.sleep
        self._clock: Callable[[], float] = clock or time.perf_counter

        self._driver: DK1Driver | None = None
        self._instruction: str | None = None
        self._last_state: Vec | None = None
        self._t_last = 0.0
        self.num_steps = 0
        self.settle_timeouts = 0
        # Set when the per-trial timeout budget is exhausted. Cleared at reset.
        self._settle_disabled = False

        self.info = EmbodimentInfo(
            name="dk1",
            action_space=action_box(
                low=self._cfg.low,
                high=self._cfg.high,
                max_relative_target=self._cfg.max_relative_target,
            ),
            observation_space=observation_space(
                self._cfg.cam_height,
                self._cfg.cam_width,
                self._cfg.cameras,
            ),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
            docs=DOCS,
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect (if needed), drive to home, and block on operator readiness."""
        # Clear at entry because the home command settles below. Clearing near
        # num_steps would let one exhausted trial suppress the next reset settle.
        self.settle_timeouts = 0
        self._settle_disabled = False
        self._last_state = None
        if self._driver is None:
            self._driver = self._driver_factory(self._cfg)
        self._home()
        if self._cfg.home_pose is not None:
            self._settle(np.asarray(self._cfg.home_pose, dtype=np.float64))
        self._operator.wait_ready()
        self._instruction = scene.instruction
        self.num_steps = 0
        self._t_last = self._clock()
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Clamp, slew-limit, and command one action; pace to the rate; maybe end."""
        self._require_driver()
        self.num_steps += 1
        cmd = packing.validate_dim(action.data)
        if self._cfg.joints_are_delta:
            if self._last_state is None:  # pragma: no cover - reset() always observes first
                raise RuntimeError("step() called before reset()")
            cmd = self._last_state + cmd
        target = self._send(cmd)
        # Before _pace(), so settle time is absorbed by the control period.
        settle_info = self._settle_info(self._settle(target))
        self._pace()

        obs = self._observe(self._instruction)
        if self._poll_end():
            # The operator only signals that the episode is over. The verdict,
            # partial/skip, and grader notes belong to the framework's single
            # operator prompt, which a definitive reason here would suppress.
            return StepResult(
                observation=obs,
                terminated=True,
                termination_reason=OPERATOR_END,
                info=settle_info,
            )
        return StepResult(observation=obs, terminated=False, info=settle_info)

    def close(self) -> None:
        """Disconnect and release the driver (no-op if never connected).

        The handle is cleared even if ``disconnect()`` raises, so a failed
        shutdown can't leave a half-dead driver behind (``close()`` stays
        idempotent either way).
        """
        if self._driver is not None:
            try:
                self._driver.disconnect()
            finally:
                self._driver = None
                self._last_state = None

    def __enter__(self) -> DK1Embodiment:
        """Support ``with DK1Embodiment(...) as emb:`` for guaranteed shutdown."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _require_driver(self) -> DK1Driver:
        if self._driver is None:  # pragma: no cover - reset() always connects first
            raise RuntimeError("step() called before reset()")
        return self._driver

    def _home(self) -> None:
        """Drive to home_pose, interpolating in bounded steps of max_relative_target.

        Open-loop design: the trajectory is computed from a single initial
        observation and advanced in closed-form, so a physically stalled joint
        still terminates. ``_last_state`` tracks the interpolated pose so the
        per-command slew limit in :meth:`_send` follows the plan instead of
        pinning every command to the start pose.
        """
        if self._cfg.home_pose is None:
            return
        target = np.asarray(self._cfg.home_pose, dtype=np.float64)
        step_limit = self._cfg.max_relative_target
        if step_limit is None:  # pragma: no cover - DK1Config enforces this when home_pose is set
            raise RuntimeError("home_pose is set but max_relative_target is None")

        raw = self._require_driver().get_observation()
        current = packing.from_obs_dict(raw)
        if not np.all(np.isfinite(current)):
            raise RuntimeError(
                f"cannot home: initial motor observation contains non-finite values: {current}"
            )
        if np.any(current < self._cfg.low - step_limit) or np.any(
            current > self._cfg.high + step_limit
        ):
            raise RuntimeError(
                f"cannot home: observed start pose {current} is outside joint limits "
                f"[{self._cfg.low}, {self._cfg.high}] by more than max_relative_target "
                f"({step_limit}); check the arm's power-on pose"
            )
        self._last_state = current

        # Reset the pace clock so the very first homing step is paced like the rest.
        self._t_last = self._clock()

        max_dist = float(np.max(np.abs(target - current)))
        if max_dist <= step_limit:
            self._send(target)
            self._pace()
            return

        n_steps = int(np.ceil(max_dist / step_limit))
        for _ in range(n_steps - 1):
            diff = target - current
            step_delta = np.clip(diff, -step_limit, step_limit)
            current = current + step_delta
            self._send(current)
            self._last_state = current
            self._pace()
        self._send(target)
        self._pace()

    def _send(self, cmd: Vec) -> Vec:
        """Clamp, slew-limit, command the motors, and return the accepted action.

        The slew limit bounds each arm joint to ``max_relative_target`` radians
        from the last observed (or, mid-homing, planned) position. The DK1
        driver has no such limit of its own, so this is the only one between a
        chunked policy's jump and the motors besides the CLI guardrails.
        """
        clamped = np.clip(cmd, self._cfg.low, self._cfg.high)
        limit = self._cfg.max_relative_target
        ref = self._last_state
        if limit is not None and ref is not None:
            delta = np.clip(clamped[_ARM_SLOTS] - ref[_ARM_SLOTS], -limit, limit)
            clamped[_ARM_SLOTS] = ref[_ARM_SLOTS] + delta
        accepted = self._require_driver().send_action(packing.to_action_dict(clamped))
        return packing.from_obs_dict(accepted)

    def _settle(self, target: Vec) -> tuple[bool, float] | None:
        """Wait for arm joints to reach ``target`` and report status plus residual.

        ``None`` means settling is unconfigured or disabled for this trial.
        Reads precede sleeps so an already-converged command adds no delay.
        """
        tolerance = self._cfg.settle_tolerance
        if tolerance is None or self._settle_disabled:
            return None
        driver = self._require_driver()
        wanted = target[_ARM_SLOTS]
        started = self._clock()
        # The poll cap prevents a frozen or non-monotonic injected clock from
        # wedging a step. A positive timeout guarantees at least one poll.
        max_polls = math.ceil(self._cfg.settle_timeout_s / _SETTLE_POLL_S)
        polls = 0
        while True:
            current = packing.from_obs_dict(driver.get_observation())[_ARM_SLOTS]
            error = np.abs(current - wanted)
            residual = float(np.max(error))
            if residual <= tolerance:
                return True, residual
            polls += 1
            if polls >= max_polls:
                break
            if self._clock() - started >= self._cfg.settle_timeout_s:
                break
            self._sleep(_SETTLE_POLL_S)
        self.settle_timeouts += 1
        if self.settle_timeouts >= self._cfg.settle_timeout_budget:
            self._settle_disabled = True
            joint = int(_ARM_SLOTS[int(np.argmax(error))])
            # logging, not warnings.warn: identical residuals across trials must
            # still emit a notice instead of being deduplicated by the warnings
            # registry.
            logger.warning(
                "settle timeout budget exhausted; observations for the rest of "
                "this trial may precede the commanded pose (worst motor=%s, "
                "residual=%.4f rad)",
                packing.MOTORS[joint],
                residual,
            )
        return False, residual

    def _settle_info(self, settled: tuple[bool, float] | None) -> dict[str, Any]:
        """Report per-step settle state, or an empty mapping when unconfigured."""
        if self._cfg.settle_tolerance is None:
            return {}
        info: dict[str, Any] = {"settle_timeouts": self.settle_timeouts}
        if settled is not None:
            info["settled"], info["settle_residual"] = settled
        if self._settle_disabled:
            info["settle_disabled"] = True
        return info

    def _pace(self) -> None:
        hz = self._cfg.control_hz
        if hz and hz > 0:
            elapsed = self._clock() - self._t_last
            self._sleep(max(0.0, 1.0 / hz - elapsed))
        self._t_last = self._clock()

    def _observe(self, instruction: str | None) -> Observation:
        raw = self._require_driver().get_observation()
        observed_at = self._clock()
        state = packing.from_obs_dict(raw)
        self._last_state = state
        images = {cam: np.asarray(raw[cam], dtype=np.uint8) for cam in self._cfg.cameras}
        expected = (self._cfg.cam_height, self._cfg.cam_width, 3)
        for name, img in images.items():
            if img.shape != expected:
                raise ValueError(f"camera {name!r} returned shape {img.shape}, expected {expected}")
        return Observation(
            images=images,
            state={packing.STATE_KEY: state},
            instruction=instruction,
            image_times=dict.fromkeys(self._cfg.cameras, observed_at),
            state_time=observed_at,
        )
