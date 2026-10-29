"""Tests for DreamPolicy (Dream Labs' hosted Dream-1 over HTTP).

Every test injects a fake session: no network, no key, no Dream SDK. The fake
records each POST (url, body, headers, timeout) and answers canned replies in
the gateway's envelope format, so the tests pin the wire format this policy
speaks (see docs/dream-api-notes.md).
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import pytest
from inspect_robots.compat import check_compatibility
from inspect_robots.scene import Scene
from inspect_robots.types import Observation
from PIL import Image

from inspect_robots_dk1 import dream, packing
from inspect_robots_dk1.dream import (
    API_KEY_ENV,
    DreamAPIError,
    DreamPolicy,
    DreamPolicyConfig,
)
from inspect_robots_dk1.embodiment import DK1Embodiment

DIM = packing.TOTAL_DIM
H = 48


def _ok(
    actions: Any, *, session_id: str = "sess-1", code: str = "DA-0001", reason: str | None = "new"
) -> dict:
    return {
        "status": {
            "code": code,
            "name": "OK",
            "reason": reason,
            "retryable": False,
            "retry_after_s": None,
        },
        "session_id": session_id,
        "actions": actions,
        "timing": {"serve_infer_ms": 400, "gateway_ms": 450},
    }


def _refusal(
    code: str, reason: str, *, retryable: bool = False, retry_after_s: float | None = None
) -> dict:
    return {
        "status": {
            "code": code,
            "name": code.replace("-", "_"),
            "reason": reason,
            "retryable": retryable,
            "retry_after_s": retry_after_s,
            "detail": "the detail",
        },
        "session_id": None,
        "actions": None,
    }


def _per_key_actions(side: str = "right", *, rows: int = H, grip_2d: bool = False) -> dict:
    arm = np.tile(np.arange(packing.NUM_JOINTS, dtype=float) * 0.1, (rows, 1)).tolist()
    grip = [0.25] * rows
    return {
        "per_key": {
            f"action.{side}_arm": arm,
            f"action.{side}_grip": [[g] for g in grip] if grip_2d else grip,
            f"action.{side}_eef_xyz": [[0.3, 0.1, 0.2]] * rows,
        }
    }


def _eef_only_actions(side: str = "right") -> dict:
    return {
        "q_r": None,
        "per_key": {
            f"action.{side}_arm": None,
            f"action.{side}_eef_xyz": [[0.3, 0.1, 0.2]] * H,
            f"action.{side}_eef_rot6": [[1, 0, 0, 0, 1, 0]] * H,
            f"action.{side}_grip": [0.9] * H,
        },
    }


class FakeResponse:
    def __init__(self, payload: Any, *, status_code: int = 200, text: str | None = None) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else "<body>"

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Answers each POST from a queue of responses (or raises a queued exception)."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, json: Any, headers: Any, timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": dict(headers), "timeout": timeout})
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt if isinstance(nxt, FakeResponse) else FakeResponse(nxt)


def _cfg(**over: Any) -> DreamPolicyConfig:
    values: dict[str, Any] = {"api_key": "dl_test_key", "cam_height": 4, "cam_width": 4}
    values.update(over)
    return DreamPolicyConfig(**values)


def _obs(
    *,
    state: np.ndarray | None = None,
    context: bool = False,
    wrist: bool = True,
    shape: tuple[int, ...] = (4, 4, 3),
) -> Observation:
    images: dict[str, np.ndarray] = {}
    if wrist:
        images["wrist"] = np.full(shape, 200, dtype=np.uint8)
    if context:
        images["context"] = np.full(shape, 50, dtype=np.uint8)
    if state is None:
        state = np.array([0.625, 0.006, 0.036, 0.005, -0.031, -0.010, 0.0])
    return Observation(images=images, state={"joint_pos": state}, instruction="fold")


def _policy(*responses: Any, **over: Any) -> tuple[DreamPolicy, FakeSession]:
    session = FakeSession(*responses)
    pol = DreamPolicy(_cfg(**over), session=session)
    pol.reset(Scene(id="s", instruction="Fold the towel corner to corner"))
    return pol, session


def _decode_jpeg(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ------------------------------------------------------------------ contract


def test_info_config_and_compat_zero_arg() -> None:
    pol = DreamPolicy()
    assert pol.info.name == "dream"
    assert pol.info.action_space.dim == DIM
    assert pol.info.action_space.semantics is not None
    assert pol.info.action_space.semantics.dim_labels == packing.MOTORS
    assert pol.info.control_hz is None
    assert pol.info.observation_space.state_keys == frozenset({"joint_pos"})
    assert [c.name for c in pol.info.observation_space.cameras] == ["wrist"]
    assert pol.config.action_horizon == 48
    assert pol.session_id is None
    report = check_compatibility(pol, DK1Embodiment())
    assert report.ok is True and report.errors == [] and report.warnings == []


def test_flat_kwargs_build_config() -> None:
    pol = DreamPolicy(checkpoint="dream-1-ft:abcd1234-towels", side="left", chunk_size=10)
    assert pol._cfg.checkpoint == "dream-1-ft:abcd1234-towels"
    assert pol._cfg.other_side == "right"
    assert pol.config.action_horizon == 10


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"side": "middle"}, "side must be one of"),
        ({"robot": "panda"}, "robot must be one of"),
        ({"gripper_convention": "upside"}, "gripper_convention must be one of"),
        ({"checkpoint": ""}, "checkpoint must name"),
        ({"gateway_url": "api.dreamlabs.ai"}, "gateway_url must be an http"),
        ({"chunk_size": 0}, "chunk_size must be >= 1"),
        ({"frame_width": 0}, "frame_width and frame_height"),
        ({"jpeg_quality": 100}, "jpeg_quality must be in 1..95"),
        ({"timeout_s": 0.0}, "timeout_s must be positive"),
        ({"timeout_s": float("inf")}, "timeout_s must be positive"),
    ],
)
def test_config_rejects_bad_values(over: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _cfg(**over)


# ------------------------------------------------------------- observation


def test_act_builds_request_and_maps_per_key_reply() -> None:
    pol, session = _policy(_ok(_per_key_actions()))
    chunk = pol.act(_obs(context=True))

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://api.dreamlabs.ai/v1/get_action"
    assert call["headers"]["Authorization"] == "Bearer dl_test_key"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["timeout"] == 15.0
    body = call["json"]
    assert body["checkpoint"] == "dreamlabs/dream-1-beta"
    assert body["session_id"] is None
    assert body["reset"] is False
    obs = body["observation"]
    assert obs["prompt"] == "Fold the towel corner to corner"
    assert obs["embodiment"] == "other_embodiment"
    assert obs["state.right_arm"] == [0.0] * 6  # withheld for another robot's arm: zeros
    assert obs["state.right_gripper"] == [1.0]  # our 0 (open) -> Dream 1 (open)
    assert obs["state.right_grip"] == [1.0]
    assert obs["state.left_arm"] == [0.0] * 6
    assert obs["state.left_gripper"] == [1.0] and obs["state.left_grip"] == [1.0]
    for key in ("video.wrist_r", "video.wrist_l", "video.overhead"):
        assert isinstance(obs[key], list) and len(obs[key]) == 1
        img = _decode_jpeg(obs[key][0])
        assert img.format == "JPEG" and img.size == (320, 176)
    # the real wrist frame is bright, the absent left wrist is black, the context frame is dark grey
    assert np.asarray(_decode_jpeg(obs["video.wrist_r"][0])).mean() > 150
    assert np.asarray(_decode_jpeg(obs["video.wrist_l"][0])).max() == 0
    assert 30 < np.asarray(_decode_jpeg(obs["video.overhead"][0])).mean() < 80
    assert "data:" not in obs["video.wrist_r"][0]

    assert len(chunk) == H
    assert chunk.control_hz is None
    assert chunk.inference_latency_s is not None
    row = chunk.actions[0].data
    assert row.shape == (DIM,)
    assert np.allclose(row[:6], np.arange(6) * 0.1)
    assert row[6] == pytest.approx(0.75)  # Dream 0.25 (mostly closed) -> ours 0.75
    assert pol.num_inferences == 1
    assert pol.session_id == "sess-1"
    assert pol.last_timing == {"serve_infer_ms": 400, "gateway_ms": 450}


def test_missing_context_camera_is_a_black_overhead_frame() -> None:
    pol, session = _policy(_ok(_per_key_actions()))
    pol.act(_obs(context=False))
    obs = session.calls[0]["json"]["observation"]
    assert np.asarray(_decode_jpeg(obs["video.overhead"][0])).max() == 0


def test_fill_missing_off_requires_context_camera() -> None:
    pol, _ = _policy(fill_missing_cameras=False)
    with pytest.raises(ValueError, match=r"missing camera 'context' for video.overhead"):
        pol.act(_obs(context=False))


def test_fill_missing_off_refuses_the_absent_wrist() -> None:
    pol, _ = _policy(fill_missing_cameras=False)
    with pytest.raises(ValueError, match=r"no left wrist camera for video.wrist_l"):
        pol.act(_obs(context=True))


def test_missing_wrist_camera_raises() -> None:
    pol, _ = _policy()
    with pytest.raises(ValueError, match=r"missing camera 'wrist'"):
        pol.act(_obs(wrist=False))


def test_missing_state_raises() -> None:
    pol, _ = _policy()
    obs = Observation(images={"wrist": np.zeros((4, 4, 3), np.uint8)}, state={})
    with pytest.raises(ValueError, match=r"missing state key 'joint_pos'"):
        pol.act(obs)


def test_wrong_state_width_raises() -> None:
    pol, _ = _policy()
    with pytest.raises(ValueError, match=r"expected a 7-D vector"):
        pol.act(_obs(state=np.zeros(6)))


def test_bad_frame_shape_raises() -> None:
    pol, _ = _policy()
    with pytest.raises(ValueError, match=r"frame must be HWC with 3 channels"):
        pol.act(_obs(shape=(4, 4)))


def test_robot_yam_sends_real_joints_and_lerobot_convention_passes_gripper() -> None:
    state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3])
    pol, session = _policy(
        _ok(_per_key_actions("left")), robot="yam", gripper_convention="lerobot", side="left"
    )
    chunk = pol.act(_obs(state=state))
    obs = session.calls[0]["json"]["observation"]
    assert obs["embodiment"] == "yam"
    assert obs["state.left_arm"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert obs["state.left_gripper"] == [pytest.approx(0.3)]  # as is
    assert obs["state.right_arm"] == [0.0] * 6
    assert obs["state.right_gripper"] == [1.0]
    assert "video.wrist_l" in obs and "video.wrist_r" in obs
    assert chunk.actions[0].data[6] == pytest.approx(0.25)  # lerobot convention: as is


def test_robot_empty_omits_embodiment_key_and_prompt_override_wins() -> None:
    pol, session = _policy(_ok(_per_key_actions()), robot="", prompt="pick the cup")
    pol.act(_obs())
    obs = session.calls[0]["json"]["observation"]
    assert "embodiment" not in obs
    assert obs["prompt"] == "pick the cup"
    assert obs["state.right_arm"] == pytest.approx([0.625, 0.006, 0.036, 0.005, -0.031, -0.010])


def test_prompt_empty_when_no_instruction() -> None:
    session = FakeSession(_ok(_per_key_actions()))
    pol = DreamPolicy(_cfg(), session=session)
    pol.reset(Scene(id="s", instruction=None))
    pol.act(_obs())
    assert session.calls[0]["json"]["observation"]["prompt"] == ""


def test_frame_size_and_quality_are_configurable() -> None:
    pol, session = _policy(
        _ok(_per_key_actions()), frame_width=64, frame_height=32, jpeg_quality=50
    )
    pol.act(_obs())
    obs = session.calls[0]["json"]["observation"]
    assert _decode_jpeg(obs["video.wrist_r"][0]).size == (64, 32)


# ----------------------------------------------------------------- replies


def test_q_chunk_reply_maps_with_gripper_inverted() -> None:
    rows = np.zeros((3, DIM))
    rows[:, 6] = [1.0, 0.5, 0.0]  # Dream: open, half, closed
    rows[:, 0] = [0.1, 0.2, 0.3]
    pol, _ = _policy(_ok({"q_r_chunk": rows.tolist(), "q_l_chunk": rows.tolist(), "per_key": {}}))
    chunk = pol.act(_obs())
    got = np.stack([a.data for a in chunk.actions])
    assert got[:, 0].tolist() == [0.1, 0.2, 0.3]
    assert got[:, 6].tolist() == [0.0, 0.5, 1.0]  # ours: open, half, closed


def test_left_side_reads_q_l_chunk() -> None:
    rows = np.ones((2, DIM))
    pol, _ = _policy(_ok({"q_l_chunk": rows.tolist()}), side="left")
    assert len(pol.act(_obs())) == 2


def test_per_key_gripper_may_be_2d_or_named_gripper() -> None:
    pol, _ = _policy(_ok(_per_key_actions(grip_2d=True)))
    assert pol.act(_obs()).actions[0].data[6] == pytest.approx(0.75)
    actions = _per_key_actions()
    actions["per_key"]["action.right_gripper"] = actions["per_key"].pop("action.right_grip")
    pol, _ = _policy(_ok(actions))
    assert pol.act(_obs()).actions[0].data[6] == pytest.approx(0.75)


def test_chunk_truncated_to_chunk_size() -> None:
    pol, _ = _policy(_ok(_per_key_actions()), chunk_size=5)
    assert len(pol.act(_obs())) == 5


def test_eef_only_reply_raises_clearly() -> None:
    pol, _ = _policy(_ok(_eef_only_actions()))
    with pytest.raises(ValueError, match=r"no joint targets for the right arm") as exc:
        pol.act(_obs())
    msg = str(exc.value)
    assert "action.right_eef_xyz" in msg and "robot='other_embodiment'" in msg
    assert "fine-tuned on DK1 data" in msg


def test_q_chunk_wrong_width_raises() -> None:
    pol, _ = _policy(_ok({"q_r_chunk": np.zeros((2, 14)).tolist()}))
    with pytest.raises(ValueError, match=r"q_r_chunk has shape \(2, 14\); expected \(N, 7\)"):
        pol.act(_obs())


def test_per_key_arm_wrong_width_raises() -> None:
    actions = {
        "per_key": {"action.right_arm": np.zeros((2, 7)).tolist(), "action.right_grip": [0, 0]}
    }
    pol, _ = _policy(_ok(actions))
    with pytest.raises(ValueError, match=r"arm chunk has shape \(2, 7\); expected \(N, 6\)"):
        pol.act(_obs())


def test_per_key_gripper_wrong_length_raises() -> None:
    actions = {
        "per_key": {"action.right_arm": np.zeros((2, 6)).tolist(), "action.right_grip": [0.0]}
    }
    pol, _ = _policy(_ok(actions))
    with pytest.raises(ValueError, match=r"gripper chunk has shape \(1,\); expected \(2,\)"):
        pol.act(_obs())


@pytest.mark.parametrize("actions", [_per_key_actions(rows=0), {"q_r_chunk": []}])
def test_empty_chunk_raises(actions: dict) -> None:
    pol, _ = _policy(_ok(actions))
    with pytest.raises(ValueError, match=r"empty action chunk"):
        pol.act(_obs())


def test_actions_not_a_dict_raises() -> None:
    pol, _ = _policy(_ok([1, 2, 3]))
    with pytest.raises(ValueError, match=r"'actions' is not a dict: list"):
        pol.act(_obs())


# ------------------------------------------------------------------ errors


def test_http_refusal_carries_envelope() -> None:
    pol, _ = _policy(
        FakeResponse(
            _refusal("DA-2000", "pool_warming", retryable=True, retry_after_s=30), status_code=503
        )
    )
    with pytest.raises(DreamAPIError) as exc:
        pol.act(_obs())
    err = exc.value
    assert err.code == "DA-2000" and err.reason == "pool_warming"
    assert err.retryable is True and err.retry_after_s == 30.0
    assert err.http_status == 503
    assert "DA-2000 DA_2000 (pool_warming): the detail" in str(err)


def test_unauthorized_is_not_retryable() -> None:
    pol, _ = _policy(FakeResponse(_refusal("DA-1000", "unknown_key"), status_code=401))
    with pytest.raises(DreamAPIError) as exc:
        pol.act(_obs())
    assert exc.value.code == "DA-1000" and exc.value.retryable is False
    assert exc.value.retry_after_s is None


def test_refusal_without_reason_or_detail() -> None:
    status = {"code": "DA-5000", "name": "MODEL_FAILED", "reason": None, "retryable": False}
    pol, _ = _policy(FakeResponse({"status": status, "session_id": None, "actions": None}))
    with pytest.raises(DreamAPIError) as exc:
        pol.act(_obs())
    assert str(exc.value) == "DreamAPI refused the request: DA-5000 MODEL_FAILED"
    assert exc.value.code == "DA-5000" and exc.value.reason is None


def test_non_json_reply_raises() -> None:
    pol, _ = _policy(
        FakeResponse(ValueError("no json"), status_code=502, text="<html>bad gateway</html>")
    )
    with pytest.raises(
        DreamAPIError, match=r"non-JSON reply .* \(HTTP 502\): '<html>bad gateway</html>'"
    ) as exc:
        pol.act(_obs())
    assert exc.value.http_status == 502


@pytest.mark.parametrize(
    "payload", [{"detail": "not found"}, {"status": "ok"}, {"status": {"code": 7}}, "x"]
)
def test_reply_without_envelope_raises(payload: Any) -> None:
    pol, _ = _policy(FakeResponse(payload, status_code=404))
    with pytest.raises(DreamAPIError, match=r"no status envelope \(HTTP 404\)"):
        pol.act(_obs())


def test_transport_failure_raises() -> None:
    pol, _ = _policy(OSError("connection refused"))
    with pytest.raises(
        DreamAPIError, match=r"request to https://api\.dreamlabs\.ai/v1/get_action failed"
    ):
        pol.act(_obs())


def test_missing_key_raises_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    session = FakeSession()
    pol = DreamPolicy(_cfg(api_key=""), session=session)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match=rf"no Dream API key.*{API_KEY_ENV}"):
        pol.act(_obs())
    assert session.calls == []


def test_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "dl_env_key")
    pol, session = _policy(_ok(_per_key_actions()), api_key="")
    pol.act(_obs())
    assert session.calls[0]["headers"]["Authorization"] == "Bearer dl_env_key"


# ---------------------------------------------------------------- sessions


def test_session_echoed_then_released_on_reset_and_close() -> None:
    pol, session = _policy(
        _ok(_per_key_actions(), session_id="sess-1"),
        _ok(_per_key_actions(), session_id="sess-1", code="DA-0000", reason=None),
        _ok(None, session_id=None, code="DA-0000", reason=None),  # end_session
        _ok(_per_key_actions(), session_id="sess-2"),
        _ok(None, session_id=None, code="DA-0000", reason=None),  # end_session
    )
    pol.act(_obs())
    pol.act(_obs())
    assert session.calls[1]["json"]["session_id"] == "sess-1"
    assert pol.num_inferences == 2 and pol.session_replacements == 0

    pol.reset(Scene(id="t", instruction="again"))
    end = session.calls[2]
    assert end["url"] == "https://api.dreamlabs.ai/v1/end_session"
    assert end["json"] == {"checkpoint": "dreamlabs/dream-1-beta", "session_id": "sess-1"}
    assert pol.session_id is None and pol.num_inferences == 0

    pol.act(_obs())
    assert session.calls[3]["json"]["session_id"] is None
    assert pol.session_id == "sess-2"
    pol.close()
    assert session.calls[4]["json"]["session_id"] == "sess-2"
    pol.close()  # nothing to release: no request
    assert len(session.calls) == 5


def test_end_session_failure_is_swallowed() -> None:
    pol, session = _policy(_ok(_per_key_actions()), OSError("gone"))
    pol.act(_obs())
    pol.close()
    assert pol.session_id is None and len(session.calls) == 2


def test_server_side_session_replacement_is_counted_and_adopted() -> None:
    pol, _ = _policy(
        _ok(_per_key_actions(), session_id="sess-1"),
        _ok(_per_key_actions(), session_id="sess-9", code="DA-0001", reason="expired"),
    )
    pol.act(_obs())
    pol.act(_obs())
    assert pol.session_replacements == 1 and pol.session_id == "sess-9"


def test_reply_without_session_id_keeps_the_old_one() -> None:
    pol, _ = _policy(
        _ok(_per_key_actions(), session_id="sess-1"),
        _ok(_per_key_actions(), session_id=None, code="DA-0000", reason=None),
    )
    pol.act(_obs())
    pol.act(_obs())
    assert pol.session_id == "sess-1"
    assert pol.last_timing is not None
    pol2, _ = _policy({**_ok(_per_key_actions()), "timing": "n/a"})
    pol2.act(_obs())
    assert pol2.last_timing is None


# --------------------------------------------------------- default session


def test_default_session_is_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    made: list[FakeSession] = []

    def _factory() -> FakeSession:
        fs = FakeSession(_ok(_per_key_actions()), _ok(_per_key_actions()))
        made.append(fs)
        return fs

    monkeypatch.setattr(requests, "Session", _factory)
    pol = DreamPolicy(_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    pol.act(_obs())
    pol.act(_obs())
    assert len(made) == 1 and len(made[0].calls) == 2


def test_pose_fn_adds_keys_and_may_not_clash() -> None:
    seen: list[np.ndarray] = []

    def pose(state: np.ndarray) -> dict[str, Any]:
        seen.append(state)
        return {"state.right_eef_xyz": [0.3, 0.1, 0.2], "state.right_eef_rot6": [1, 0, 0, 0, 1, 0]}

    session = FakeSession(_ok(_per_key_actions()))
    pol = DreamPolicy(_cfg(), session=session, pose_fn=pose)
    pol.reset(Scene(id="s", instruction="x"))
    pol.act(_obs())
    obs = session.calls[0]["json"]["observation"]
    assert obs["state.right_eef_xyz"] == [0.3, 0.1, 0.2]
    assert obs["state.right_eef_rot6"] == [1, 0, 0, 0, 1, 0]
    assert seen[0].shape == (DIM,) and seen[0][0] == pytest.approx(0.625)

    clashing = DreamPolicy(_cfg(), session=FakeSession(), pose_fn=lambda s: {"prompt": "no"})
    clashing.reset(Scene(id="s", instruction="x"))
    with pytest.raises(
        ValueError, match=r"pose_fn returned keys the policy already sends: \['prompt'\]"
    ):
        clashing.act(_obs())


def test_blank_frame_helper() -> None:
    img = _decode_jpeg(dream.blank_frame((8, 6), 85))
    assert img.size == (8, 6) and np.asarray(img).max() == 0
