"""inspect-robots-dk1: Inspect Robots adapter for the TRLC-DK1 follower + LeRobot policies.

Registers three Inspect Robots components via entry points:

* embodiment ``dk1``: :class:`~inspect_robots_dk1.embodiment.DK1Embodiment`
* policy ``lerobot``: :class:`~inspect_robots_dk1.policy.LeRobotPolicy`
* policy ``dream``: :class:`~inspect_robots_dk1.dream.DreamPolicy` (Dream Labs' hosted Dream-1)

so ``inspect-robots run --instruction "Put the can in the box" --policy lerobot
--embodiment dk1`` works once both packages are installed. Use
``inspect-robots doctor --embodiment dk1`` or
:func:`~inspect_robots_dk1.preflight.run_preflight` (the
``inspect-robots-dk1-preflight`` CLI) to verify compatibility before any motion.

Ported from `inspect-robots-so101 <https://github.com/robocurve/inspect-robots-so101>`_.
"""

from __future__ import annotations

from inspect_robots_dk1.config import DK1Config, LeRobotPolicyConfig
from inspect_robots_dk1.dream import DreamAPIError, DreamPolicy, DreamPolicyConfig
from inspect_robots_dk1.embodiment import DK1Embodiment
from inspect_robots_dk1.operator import OperatorIO
from inspect_robots_dk1.packing import MOTORS, STATE_KEY, TOTAL_DIM, from_obs_dict, to_action_dict
from inspect_robots_dk1.policy import LeRobotPolicy
from inspect_robots_dk1.preflight import build, run_preflight

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("inspect-robots-dk1")
except PackageNotFoundError:  # pragma: no cover - only hit in a non-installed tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "MOTORS",
    "STATE_KEY",
    "TOTAL_DIM",
    "DK1Config",
    "DK1Embodiment",
    "DreamAPIError",
    "DreamPolicy",
    "DreamPolicyConfig",
    "LeRobotPolicy",
    "LeRobotPolicyConfig",
    "OperatorIO",
    "build",
    "from_obs_dict",
    "run_preflight",
    "to_action_dict",
]
