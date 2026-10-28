"""Goal-closing tests: the DK1 embodiment passes the Inspect Robots conformance
kit, and the DK1 + LeRobot pairing is compatible with zero errors and zero
warnings; built-in tasks are realizable."""

from __future__ import annotations

from inspect_robots.compat import check_compatibility
from inspect_robots.conformance import (
    assert_embodiment_conformant,
    check_embodiment,
    device_slots,
)
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.spaces import ActionSemantics, Box

from inspect_robots_dk1.config import DK1Config, action_box, observation_space
from inspect_robots_dk1.embodiment import DK1Embodiment
from inspect_robots_dk1.policy import LeRobotPolicy


def test_embodiment_is_conformant() -> None:
    # The adapters guide's one required test: hardware-free constructor, finite
    # bounds, dim labels, one state field of shape (action_dim,), guardrails build.
    assert_embodiment_conformant(DK1Embodiment().info)


def test_conformance_report_has_no_issues_without_slew_limit_either() -> None:
    report = check_embodiment(DK1Embodiment(DK1Config(max_relative_target=None)).info)
    assert report.issues == ()


def test_device_slots_declare_the_serial_port() -> None:
    slots = device_slots(DK1Embodiment)
    assert [(s.arg, s.kind) for s in slots] == [("port", "serial")]


def test_runtime_requirements_name_the_plugin_modules() -> None:
    reqs = DK1Embodiment.RUNTIME_REQUIREMENTS
    assert set(reqs) == {"lerobot", "grpc", "lerobot_robot_trlc_dk1", "trlc_dk1_control"}
    assert all(isinstance(v, str) and "pip install" in v for v in reqs.values())


def test_lerobot_dk1_compatible_no_errors_no_warnings() -> None:
    report = check_compatibility(LeRobotPolicy(), DK1Embodiment())
    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []


def test_pair_declares_matching_state_contract() -> None:
    policy = LeRobotPolicy()
    embodiment = DK1Embodiment()
    assert policy.info.observation_space.state == embodiment.info.observation_space.state
    assert policy.info.observation_space.state is not None
    assert policy.info.observation_space.state.fields[0].unit == "rad+normalized"


def test_builtin_task_is_realizable() -> None:
    task = resolve("task", "cubepick-reach")
    report = check_compatibility(LeRobotPolicy(), DK1Embodiment(), task)
    assert report.ok is True, [i.message for i in report.errors]
    assert report.errors == []


def test_negative_wrong_dim_policy_trips_action_dim_error() -> None:
    bad = PolicyInfo(
        name="bad",
        action_space=Box(shape=(6,), semantics=ActionSemantics(control_mode="joint_pos")),
    )

    class _Bad:
        info = bad
        config = PolicyConfig()

        def reset(self, scene: object) -> None: ...
        def act(self, obs: object) -> object: ...  # pragma: no cover

    report = check_compatibility(_Bad(), DK1Embodiment())  # type: ignore[arg-type]
    assert report.ok is False
    assert any(i.code == "action_dim" for i in report.errors)


def test_negative_policy_advertising_rate_trips_control_rate_warning() -> None:
    # Locks the load-bearing detail: leaving PolicyInfo.control_hz=None is the only
    # reason the real pairing is warning-free. A policy that advertises a rate warns.
    info = PolicyInfo(
        name="rated",
        action_space=action_box(),
        observation_space=observation_space(480, 640, ("wrist",)),
        control_hz=100.0,
    )

    class _Rated:
        config = PolicyConfig()

        def __init__(self) -> None:
            self.info = info

        def reset(self, scene: object) -> None: ...
        def act(self, obs: object) -> object: ...  # pragma: no cover

    report = check_compatibility(_Rated(), DK1Embodiment())  # type: ignore[arg-type]
    assert report.ok is True  # only a warning, not an error
    assert any(i.code == "control_rate" for i in report.warnings)
