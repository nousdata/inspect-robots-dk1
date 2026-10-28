"""End-to-end: a full eval() rollout on a mocked DK1 and LeRobot policy.

Proves the judgement-based wiring (operator end -> before_scoring -> operator
judgement -> operator scorer) and chunk replay compose. The inline task keeps
the suite self-contained without hardware, lerobot, or torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from inspect_robots import Scene, Task, TrialRecord, operator_scorer
from inspect_robots import eval as rl_eval

from inspect_robots_dk1 import packing
from inspect_robots_dk1.config import DK1Config, LeRobotPolicyConfig
from inspect_robots_dk1.embodiment import DK1Embodiment
from inspect_robots_dk1.operator import OperatorIO
from inspect_robots_dk1.policy import LeRobotPolicy


class _FakeDriver:
    def __init__(self) -> None:
        self.state = np.zeros(packing.TOTAL_DIM)

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = packing.to_action_dict(self.state)
        obs["wrist"] = np.zeros((4, 4, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.state = packing.from_obs_dict(action)
        return action

    def disconnect(self) -> None: ...


def _predict(_obs: Any) -> np.ndarray:
    return np.zeros((1, packing.TOTAL_DIM), dtype=np.float32)  # one-action chunk of zeros


def _grade_yes(record: TrialRecord, scene: Scene) -> None:
    del scene
    assert record.termination_reason == "operator_end"
    record.operator_judgement = "y"


def test_eval_scores_success_end_to_end() -> None:
    policy = LeRobotPolicy(
        LeRobotPolicyConfig(cam_height=4, cam_width=4, chunk_size=1),
        predict_fn=_predict,
    )
    embodiment = DK1Embodiment(
        DK1Config(cam_height=4, cam_width=4),
        driver_factory=lambda _c: _FakeDriver(),
        operator=OperatorIO(input_fn=lambda _p: "", output_fn=lambda _m: None),
        poll_end=lambda: True,  # operator ends every episode immediately
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )
    task = Task(
        name="dk1-operator-e2e",
        scenes=[Scene(id="operator-e2e", instruction="put the can in the box")],
        scorer=operator_scorer(),
        max_steps=1,
    )

    logs = rl_eval(task, policy, embodiment, sinks=[], seed=0, before_scoring=_grade_yes)

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.results.metrics["operator"] == 1.0
