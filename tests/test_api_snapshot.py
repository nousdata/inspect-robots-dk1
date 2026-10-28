"""Guard the public API surface so changes to __all__ are deliberate."""

from __future__ import annotations

import re

import inspect_robots_dk1

EXPECTED_API = {
    "MOTORS",
    "STATE_KEY",
    "TOTAL_DIM",
    "DK1Config",
    "DK1Embodiment",
    "LeRobotPolicy",
    "LeRobotPolicyConfig",
    "OperatorIO",
    "build",
    "from_obs_dict",
    "run_preflight",
    "to_action_dict",
}


def test_public_api_matches_all() -> None:
    assert set(inspect_robots_dk1.__all__) == EXPECTED_API


def test_all_names_are_importable() -> None:
    for name in inspect_robots_dk1.__all__:
        assert hasattr(inspect_robots_dk1, name), name


def test_version() -> None:
    assert re.match(r"\d+\.\d+", inspect_robots_dk1.__version__)


def test_entry_points_resolve_via_registry() -> None:
    from inspect_robots.registry import resolve

    pol = resolve("policy", "lerobot")
    emb = resolve("embodiment", "dk1")
    assert pol.info.name == "lerobot"
    assert emb.info.name == "dk1"
