"""Tests for DK1Embodiment (all hardware/IO seams injected: no serial, motors, stdin)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
from inspect_robots.embodiment import SELF_PACED
from inspect_robots.errors import EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.types import OPERATOR_END, Action

from conftest import DIM, FakeDriver, _build, _operator
from inspect_robots_dk1.config import DK1Config
from inspect_robots_dk1.embodiment import DOCS, DK1Embodiment, build_camera_configs

# Unslewed configs for the tests that check the hard clamp alone.
NO_SLEW = DK1Config(max_relative_target=None)


def test_zero_arg_info_no_hardware() -> None:
    emb = DK1Embodiment()  # nothing mocked: construction must not touch hardware
    assert emb.info.name == "dk1"
    assert emb.info.action_space.dim == 7
    assert emb.info.action_space.low is not None and emb.info.action_space.high is not None
    assert emb.info.control_hz == 30.0
    assert SELF_PACED in emb.info.capabilities
    assert emb.info.observation_space.camera_names == frozenset({"wrist"})
    assert emb.info.observation_space.state_keys == frozenset({"joint_pos"})
    assert emb.info.docs == DOCS and "gripper" in DOCS
    assert emb.info.is_simulated is False


def test_info_max_step_follows_config() -> None:
    sem = DK1Embodiment(DK1Config(max_relative_target=0.1)).info.action_space.semantics
    assert sem is not None and sem.max_step == (0.1,) * 6 + (1.0,)
    sem = DK1Embodiment(NO_SLEW).info.action_space.semantics
    assert sem is not None and sem.max_step is None


def test_flat_kwargs_build_config() -> None:
    emb = DK1Embodiment(port="/dev/cu.usbmodem1", control_hz=15.0)
    assert emb.info.control_hz == 15.0
    assert emb._cfg.port == "/dev/cu.usbmodem1"


def test_reset_returns_observation_and_homes() -> None:
    home = (0.05,) * 6 + (0.0,)
    cfg = DK1Config(home_pose=home, max_relative_target=0.1)
    emb, drv, _ = _build(cfg)
    obs = emb.reset(Scene(id="s", instruction="reach"))
    assert set(obs.images) == {"wrist"}
    assert obs.state["joint_pos"].shape == (7,)
    assert obs.instruction == "reach"
    assert len(drv.commands) == 1  # homing command issued
    assert drv.commands[0] == pytest.approx(list(home))


def test_reset_interpolates_homing_when_far() -> None:
    """Each intermediate command differs from the previous by at most max_relative_target."""
    # Mixed directions: joints 0-2 start below home, joint 3 is already at home,
    # joints 4-5 start above home (negative direction). Gripper closes 0 -> 1.
    home = (1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0)
    start = np.array([0.0, 0.0, 0.0, 0.5, 0.8, 0.8, 0.0])
    cfg = DK1Config(home_pose=home, max_relative_target=0.4)
    driver = FakeDriver(state=start)
    emb, drv, _ = _build(cfg, driver=driver)
    emb.reset(Scene(id="s", instruction="reach"))

    commands = np.stack(drv.commands)  # shape (n_steps, 7)
    for i in range(1, len(commands)):
        per_joint_delta = np.abs(commands[i] - commands[i - 1])
        assert np.all(per_joint_delta <= 0.4 + 1e-9), (
            f"step {i}: delta {per_joint_delta} exceeded max_relative_target"
        )
    assert commands[-1] == pytest.approx(list(home))
    assert len(commands) == 3  # ceil(1.0 / 0.4) = 3 commands


def test_reset_homing_raises_on_non_finite_observation() -> None:
    cfg = DK1Config(home_pose=(0.05,) * 6 + (0.0,), max_relative_target=0.2)
    driver = FakeDriver(state=np.full(DIM, np.nan))
    emb = DK1Embodiment(cfg, driver_factory=lambda _c: driver, operator=_operator())
    with pytest.raises(RuntimeError, match="non-finite values"):
        emb.reset(Scene(id="s", instruction="reach"))


def test_reset_homing_raises_when_start_out_of_limits() -> None:
    """_home() raises if the observed start pose is outside limits by more than step_limit."""
    cfg = DK1Config(home_pose=(0.05,) * 6 + (0.0,), max_relative_target=0.1)
    driver = FakeDriver(state=np.full(DIM, 15.0))  # far above every limit
    emb = DK1Embodiment(cfg, driver_factory=lambda _c: driver, operator=_operator())
    with pytest.raises(RuntimeError, match="outside joint limits"):
        emb.reset(Scene(id="s", instruction="reach"))


def test_homing_first_step_is_paced() -> None:
    """The first homing step must be paced like every other step (so101 regression)."""
    control_hz = 30.0
    period = 1.0 / control_hz
    _t = [100.0]

    def _clock() -> float:
        t = _t[0]
        _t[0] += period / 2
        return t

    cfg = DK1Config(
        home_pose=(1.0,) * 6 + (0.0,),
        max_relative_target=0.2,
        control_hz=control_hz,
        cam_height=4,
        cam_width=4,
    )
    sleeps: list[float] = []
    emb = DK1Embodiment(
        cfg,
        driver_factory=lambda _c: FakeDriver(),
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=sleeps.append,
        clock=_clock,
    )
    emb.reset(Scene(id="s", instruction="reach"))

    # Homing 0 -> 1.0 in steps of 0.2 -> 5 commands -> 5 _pace() calls.
    assert len(sleeps) >= 5, f"expected >=5 sleeps, got {len(sleeps)}"
    expected_sleep = period / 2
    for i, s in enumerate(sleeps[:5]):
        assert s == pytest.approx(expected_sleep, abs=1e-9), f"sleep[{i}] = {s:.9f}s"


def test_observation_records_monotonic_capture_times() -> None:
    times = iter([10.0, 10.25])
    emb = DK1Embodiment(
        DK1Config(cam_height=4, cam_width=4),
        driver_factory=lambda _c: FakeDriver(),
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _d: None,
        clock=lambda: next(times),
    )
    obs = emb.reset(Scene(id="s", instruction="reach"))
    assert obs.image_times == {"wrist": 10.25}
    assert obs.state_time == 10.25


def test_reset_without_home_pose_issues_no_command() -> None:
    emb, drv, _ = _build()
    emb.reset(Scene(id="s", instruction="x"))
    assert drv.commands == []


# -- start_joints glide -------------------------------------------------------

# joint_4 at -1.6 rad keeps this inside the default joint box (its floor is -1.665).
CLI_START = "0.0,1.03,1.31,-1.6,0.0,-0.02"

STEP = 1.0 / 30.0  # one control period at the default control_hz


class StuckDriver(FakeDriver):
    """Accepts and echoes every command but the motors never move."""

    def __init__(self, state: np.ndarray) -> None:
        super().__init__(state=state)
        self._stuck = np.asarray(state, dtype=float).copy()

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        accepted = FakeDriver.send_action(self, action)
        self.state = self._stuck.copy()
        return accepted


class SaggingDriver(FakeDriver):
    """Arm joints settle ``sag`` rad short of every command, in the direction they moved.

    This is the real impedance-mode wrist: a soft gain under gravity load. A
    closed-loop glide (observed + at most max_relative_target) stalls against
    it as soon as ``sag`` equals the step size, because each command re-sends
    the pose the motor just sagged back from.
    """

    def __init__(self, sag: float) -> None:
        super().__init__()
        self._sag = sag

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        before = self.state.copy()
        echo = FakeDriver.send_action(self, action)
        accepted = self.commands[-1]
        moved = np.sign(accepted[:6] - before[:6])
        self.state[:6] = accepted[:6] - self._sag * moved
        return echo


def test_reset_with_default_start_joints_does_not_glide() -> None:
    emb, drv, sleeps = _build(DK1Config(start_joints=None))
    emb.reset(Scene(id="s", instruction="x"))
    assert drv.commands == []
    assert drv.observation_reads == 1  # only the reset observation
    assert sleeps == []


def test_start_joints_glide_sends_the_planned_wrist_steps_then_the_arm_steps() -> None:
    """Phase 1 is joint_4's 3-step plan, phase 2 is joint_1's 5-step plan, exact values."""
    start = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3])  # gripper part-closed
    cfg = DK1Config(start_joints=(0.5, 0.2, 0.0, -0.25, 0.0, 0.0), max_relative_target=0.1)
    emb, drv, sleeps = _build(cfg, driver=FakeDriver(state=start))
    emb.reset(Scene(id="s", instruction="x"))

    commands = np.stack(drv.commands)
    assert len(commands) == 3 + 5  # ceil(0.25 / 0.1) wrist steps, then ceil(0.5 / 0.1) arm steps
    # Phase 1: joint_4 steps -0.1, -0.2, -0.25 while joint_1..joint_3 hold their observed pose.
    assert commands[:3, 3] == pytest.approx([-0.1, -0.2, -0.25])
    assert commands[:3, 0] == pytest.approx([0.0, 0.0, 0.0])
    assert commands[:3, 1] == pytest.approx([0.2, 0.2, 0.2])
    # Phase 2: joint_1 steps 0.1 .. 0.5 while the wrist is pinned to its target.
    assert commands[3:, 0] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert commands[3:, 3] == pytest.approx([-0.25] * 5)
    assert commands[-1][:6] == pytest.approx([0.5, 0.2, 0.0, -0.25, 0.0, 0.0])
    # Joints with no plan (joint_2, joint_3, joint_5, joint_6) never leave their observed values.
    assert commands[:, [1, 2, 4, 5]] == pytest.approx(np.tile([0.2, 0.0, 0.0, 0.0], (8, 1)))
    # One planning observation, one residual check per phase, then the reset observation.
    assert drv.observation_reads == 1 + 2 + 1
    # Paced at control_hz after every command (the fake clock never advances).
    assert sleeps == pytest.approx([STEP] * 8)


def test_start_joints_glide_holds_the_gripper_at_its_observed_value() -> None:
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7])
    cfg = DK1Config(start_joints=(0.2, 0.0, 0.0, -0.1, 0.0, 0.0), max_relative_target=0.1)
    emb, drv, _ = _build(cfg, driver=FakeDriver(state=start))
    emb.reset(Scene(id="s", instruction="x"))
    assert len(drv.commands) == 1 + 2
    assert all(cmd[6] == pytest.approx(0.7) for cmd in drv.commands)


def test_start_joints_glide_completes_against_a_motor_that_settles_short() -> None:
    """The hardware finding: a motor 0.02 rad behind every command no longer stalls the glide.

    With max_relative_target equal to the sag, the old closed-loop glide sent
    the same pose forever. The open-loop plan still sends every planned step,
    the residual (0.02) is steady-state sag, and no fault is raised.
    """
    cfg = DK1Config(start_joints=(0.1, 0.0, 0.0, -0.1, 0.0, 0.0), max_relative_target=0.02)
    drv = SaggingDriver(sag=0.02)
    emb, _, _ = _build(cfg, driver=drv)
    obs = emb.reset(Scene(id="s", instruction="x"))

    commands = np.stack(drv.commands)
    assert len(commands) == 5 + 5  # ceil(0.1 / 0.02) per phase, planned, not re-observed
    assert commands[:5, 3] == pytest.approx([-0.02, -0.04, -0.06, -0.08, -0.1])
    assert commands[:5, 0] == pytest.approx([0.0] * 5)
    assert commands[5:, 0] == pytest.approx([0.02, 0.04, 0.06, 0.08, 0.1])
    assert commands[5:, 3] == pytest.approx([-0.1] * 5)  # pinned to target despite the sag
    # The motors sit 0.02 short of both targets; that residual is accepted.
    assert obs.state["joint_pos"][0] == pytest.approx(0.08)
    assert obs.state["joint_pos"][3] == pytest.approx(-0.08)


def test_start_joints_glide_goes_through_the_hard_clamp() -> None:
    """A start pose at the joint limit arrives clamped, never above it."""
    high = float(DK1Config().high[0])
    cfg = DK1Config(start_joints=(high, 0.0, 0.0, 0.0, 0.0, 0.0), max_relative_target=5.0)
    emb, drv, _ = _build(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    assert len(drv.commands) == 1
    assert drv.commands[0][0] == pytest.approx(high)


@pytest.mark.parametrize(
    ("start", "phase", "joint"),
    [
        ((0.0, 1.0, 0.0, 0.0, 0.0, 0.0), "phase 2 (arm, joint_1..joint_3)", "joint_2"),
        ((0.0, 0.0, 0.0, -1.0, 0.0, 0.0), "phase 1 (wrist, joint_4..joint_6)", "joint_4"),
    ],
)
def test_start_joints_glide_faults_when_a_joint_does_not_move(
    start: tuple[float, ...], phase: str, joint: str
) -> None:
    """A phase whose joint stays put ends with a 1.0 rad residual and faults, naming both."""
    cfg = DK1Config(start_joints=start, max_relative_target=0.1)
    drv = StuckDriver(np.zeros(DIM))
    emb, _, _ = _build(cfg, driver=drv)
    expected = re.escape(f"start_joints glide {phase} ended with {joint} 1.000 rad from its target")
    with pytest.raises(EmbodimentFault, match=expected):
        emb.reset(Scene(id="s", instruction="x"))
    # The whole 10-step plan was sent (open-loop), then the residual check faulted
    # before any later phase could start.
    idx = int(joint.split("_")[1]) - 1
    sign = np.sign(start[idx])
    assert [c[idx] for c in drv.commands] == pytest.approx([0.1 * k * sign for k in range(1, 11)])


def test_start_joints_glide_faults_on_non_finite_observation() -> None:
    cfg = DK1Config(start_joints=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0), max_relative_target=0.1)
    emb, _, _ = _build(cfg, driver=FakeDriver(state=np.full(DIM, np.nan)))
    with pytest.raises(EmbodimentFault, match="non-finite"):
        emb.reset(Scene(id="s", instruction="x"))


def test_start_joints_from_cli_string_glides_then_homes() -> None:
    """The CLI string reaches the embodiment through from_kwargs and drives the glide."""
    emb = DK1Embodiment(
        start_joints=CLI_START,
        max_relative_target=0.5,
        cam_height=4,
        cam_width=4,
        driver_factory=lambda _c: FakeDriver(),
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )
    obs = emb.reset(Scene(id="s", instruction="x"))
    assert obs.state["joint_pos"][:6] == pytest.approx([0.0, 1.03, 1.31, -1.6, 0.0, -0.02])
    assert obs.state["joint_pos"][6] == pytest.approx(0.0)


def test_step_clamps_to_limits() -> None:
    emb, drv, _ = _build(NO_SLEW)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(DIM, 1000.0)))
    cmd = drv.commands[-1]
    assert cmd[:6] == pytest.approx(list(NO_SLEW.high[:6]))  # joints clamped
    assert cmd[6] == pytest.approx(1.0)  # gripper clamped to closed


def test_step_clamps_low_side() -> None:
    emb, drv, _ = _build(NO_SLEW)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(DIM, -1000.0)))
    cmd = drv.commands[-1]
    assert cmd[:6] == pytest.approx(list(NO_SLEW.low[:6]))
    assert cmd[6] == pytest.approx(0.0)  # gripper floor


def test_step_slew_limits_arm_joints_from_observed_pose() -> None:
    """The DK1 driver has no slew limit, so the embodiment bounds each arm joint
    to max_relative_target radians from the last observed position; the gripper
    is free to jump."""
    emb, drv, _ = _build(DK1Config(max_relative_target=0.05))
    emb.reset(Scene(id="s", instruction="x"))  # observed at zeros
    emb.step(Action(data=np.array([1.0, 0.02, 0.0, 0.0, 0.0, -1.0, 1.0])))
    assert drv.commands[-1] == pytest.approx([0.05, 0.02, 0.0, 0.0, 0.0, -0.05, 1.0])
    # Next step slews from the new observation (the fake arm reached +-0.05).
    emb.step(Action(data=np.array([1.0, 0.02, 0.0, 0.0, 0.0, -1.0, 0.0])))
    assert drv.commands[-1] == pytest.approx([0.10, 0.02, 0.0, 0.0, 0.0, -0.10, 0.0])


def test_step_slew_applies_after_the_hard_clamp() -> None:
    # Start near the joint_1 upper limit: the clamp pulls 1000 down to the limit,
    # then the slew keeps the command within 0.05 of the observed pose.
    high = float(NO_SLEW.high[0])
    start = np.zeros(DIM)
    start[0] = high - 0.02
    emb, drv, _ = _build(DK1Config(max_relative_target=0.05), driver=FakeDriver(state=start))
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
    assert drv.commands[-1][0] == pytest.approx(high)


def test_step_delta_mode_adds_current() -> None:
    drv = FakeDriver(state=np.full(DIM, 0.1))
    cfg = DK1Config(joints_are_delta=True, max_relative_target=None)
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(DIM, 0.01)))
    emb.step(Action(data=np.full(DIM, 0.01)))
    assert drv.commands[0][0] == pytest.approx(0.11)
    assert drv.commands[1][0] == pytest.approx(0.12)
    # One read at reset and one post-action read per step; no extra delta-mode read.
    assert drv.observation_reads == 3


def test_reset_twice_reuses_driver() -> None:
    calls = {"n": 0}

    def _factory(_c: DK1Config) -> FakeDriver:
        calls["n"] += 1
        return FakeDriver()

    emb = DK1Embodiment(
        DK1Config(cam_height=4, cam_width=4),
        driver_factory=_factory,
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )
    emb.reset(Scene(id="s", instruction="x"))
    emb.reset(Scene(id="s", instruction="x"))
    assert calls["n"] == 1  # driver built once, reused on the second reset


def test_step_terminates_operator_end_without_grading_prompt() -> None:
    prompts: list[str] = []
    emb, _, _ = _build(poll_end_seq=[True], operator=_operator(prompts=prompts))
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(DIM)))
    assert result.terminated is True
    assert result.termination_reason == OPERATOR_END
    assert result.info == {}
    assert prompts == ["Position the scene, then press Enter to start..."]


def test_step_continues_when_no_end_signal() -> None:
    emb, _, _ = _build(poll_end_seq=[False])
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(DIM)))
    assert result.terminated is False
    assert emb.num_steps == 1


def test_pacing_sleeps_to_control_rate() -> None:
    emb, _, sleeps = _build()  # control_hz=30 -> period ~0.0333, clock constant 0
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.zeros(DIM)))
    assert sleeps and sleeps[-1] == pytest.approx(1.0 / 30.0)


def test_pacing_skipped_when_hz_zero() -> None:
    emb, _, sleeps = _build(DK1Config(control_hz=0.0))
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.zeros(DIM)))
    assert sleeps == []


def test_close_idempotent_and_releases() -> None:
    emb, drv, _ = _build()
    emb.close()  # before connect: no error
    emb.reset(Scene(id="s", instruction="x"))
    emb.close()
    assert drv.disconnected is True
    emb.close()  # second close: no error


def test_close_releases_driver_even_if_disconnect_raises() -> None:
    class ExplodingDriver(FakeDriver):
        def disconnect(self) -> None:
            raise RuntimeError("serial port yanked")

    drv = ExplodingDriver()
    emb, _, _ = _build(driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    with pytest.raises(RuntimeError, match="serial port yanked"):
        emb.close()
    emb.close()  # handle was cleared despite the raise: now a no-op


def test_context_manager_closes_on_exit() -> None:
    emb, drv, _ = _build()
    with emb as entered:
        assert entered is emb
        emb.reset(Scene(id="s", instruction="x"))
    assert drv.disconnected is True


def test_context_manager_closes_on_exception() -> None:
    emb, drv, _ = _build()
    with pytest.raises(RuntimeError, match="boom"), emb:
        emb.reset(Scene(id="s", instruction="x"))
        raise RuntimeError("boom")
    assert drv.disconnected is True


@pytest.mark.parametrize("shape", [(2, 4, 3), (4, 4, 1)])
def test_observe_rejects_wrong_camera_shape(shape: tuple[int, ...]) -> None:
    driver = FakeDriver(frame_shape=shape)
    emb, _, _ = _build(driver=driver)
    with pytest.raises(ValueError) as exc:
        emb.reset(Scene(id="s", instruction="x"))
    assert str(exc.value) == f"camera 'wrist' returned shape {shape}, expected (4, 4, 3)"


# -- camera config wiring for the hardware driver factory ----------------------


def _record_make(made: list[dict[str, Any]]) -> Any:
    def _make(**kw: Any) -> dict[str, Any]:
        made.append(kw)
        return kw

    return _make


def test_build_camera_configs_from_devices() -> None:
    made: list[dict[str, Any]] = []
    cfg = DK1Config(
        cameras=("wrist", "context"),
        camera_devices="wrist=1,context=/dev/video2",
        camera_fps=60,
        camera_rotation=180,
    )
    out = build_camera_configs(cfg, make=_record_make(made))
    assert list(out) == ["wrist", "context"]
    assert made[0] == {
        "index_or_path": 1,
        "width": 640,
        "height": 480,
        "fps": 60,
        "rotation": 180,
    }
    assert made[1]["index_or_path"] == "/dev/video2"


def test_build_camera_configs_prefers_explicit_configs() -> None:
    sentinel = {"wrist": object()}
    cfg = DK1Config(camera_configs=sentinel, camera_devices="wrist=1")
    out = build_camera_configs(cfg, make=_record_make([]))
    assert out == sentinel and out is not sentinel


def test_build_camera_configs_requires_a_device_per_camera() -> None:
    with pytest.raises(ValueError, match=r"camera_devices='wrist=<index>'"):
        build_camera_configs(DK1Config(), make=_record_make([]))
    cfg = DK1Config(cameras=("wrist", "context"), camera_devices="wrist=1")
    with pytest.raises(ValueError, match=r"\['context'\]"):
        build_camera_configs(cfg, make=_record_make([]))


def test_build_camera_configs_with_no_cameras_is_empty() -> None:
    assert build_camera_configs(DK1Config(cameras=()), make=_record_make([])) == {}
