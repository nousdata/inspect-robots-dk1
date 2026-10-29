"""``DreamPolicy``: an Inspect Robots policy that asks Dream Labs' Dream-1 for actions.

Dream-1 (https://dreamlabs.ai/research/dream-1/) is a hosted robot foundation
model. There is no checkpoint to load: each ``act()`` is one HTTPS request,
``POST <gateway>/v1/get_action`` with ``Authorization: Bearer <key>``, carrying
the DK1's wrist frame (and a context frame when the embodiment provides one),
the 7-D state, and the instruction, and the reply carries a chunk of actions.
The wire format is the one the official ``dreamlabs-client`` package speaks
(``docs/dream-api-notes.md`` records the facts and their sources); this module
re-implements it with ``requests`` so the policy needs no Dream SDK, no
``websockets`` and no IK stack, and so tests can inject a fake session.

What goes on the wire (``build_observation``):

* ``state.<side>_arm`` (six floats, radians), ``state.<side>_gripper`` and
  ``state.<side>_grip`` (one float, Dream's 0 closed .. 1 open), for the side
  the DK1 occupies; the other side's slots are zeros with an open gripper,
  since Dream-1 is a bimanual model and the slots must be present. With
  ``robot="other_embodiment"`` (the default: the DK1 is not a YAM) the arm slot
  is sent as zeros, as the official client does, because the serve withholds
  another robot's joints from a base model anyway.
* ``video.wrist_<r|l>`` and ``video.overhead``: each a list holding one base64
  JPEG resized to 320x176 (what the models read). The wrist frame is required;
  the overhead frame is the embodiment's ``context`` camera when present;
  anything missing is a black frame when ``fill_missing_cameras`` is on. The
  official client sends a window of up to 9 frames per camera at 5 fps; this
  policy sends a single frame per request (a still, which the serve accepts).
* ``prompt``: the scene instruction (or the ``prompt`` config override).
* ``embodiment``: the ``robot`` config value, naming whose state this is.

What comes back and how it maps to the DK1 contract (``chunk_from_actions``):

* A fine-tuned DK1 model (one trained on DK1 recordings through Dream's
  fine-tuning service) answers ``per_key`` with ``action.<side>_arm`` rows of
  six joint targets and ``action.<side>_grip`` (or ``_gripper``) rows: these
  become the 7-D ``[joint_1..joint_6, gripper]`` chunk directly.
* A YAM-layout answer (``q_<r|l>_chunk``, rows of six joints then a gripper)
  maps the same way.
* The gripper column is converted from Dream's 0 closed .. 1 open to this
  package's 0 open .. 1 closed (``gripper_convention="dream"``); a fine-tune
  trained in LeRobot DK1 units passes it through (``"lerobot"``).
* A Dream-1 **base** model asked by a non-YAM arm answers **end-effector
  poses only** (``action.<side>_eef_xyz`` / ``_eef_rot6``; ``action.<side>_arm``
  is null). Those need the DK1's kinematics (Dream's arm file: URDF plus a
  user-measured tool centre point) to become joint targets. This policy does
  not solve IK: it raises ``ValueError`` naming the keys it got, so the
  failing assumption is visible instead of silently driving the arm.

The Dream-1 base models also **require** each hand's pose as input
(``state.<side>_eef_xyz``, ``state.<side>_eef_rot6``; the gateway refuses the
request without them). This policy has no DK1 kinematics, so it sends none;
the ``pose_fn`` constructor seam is where a forward-kinematics provider plugs
in once the tool centre point has been measured on the arm.

Construction is inert (no network, no key read); the key is read from the
``api_key`` config or the ``DREAM_API_KEY`` environment variable at the first
request and never logged.
"""

from __future__ import annotations

import base64
import contextlib
import io
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_dk1 import packing
from inspect_robots_dk1.config import _FromKwargs, action_box, observation_space
from inspect_robots_dk1.packing import GRIPPER_IDX, NUM_JOINTS, STATE_KEY, TOTAL_DIM

DEFAULT_GATEWAY_URL = "https://api.dreamlabs.ai"
GET_ACTION_PATH = "/v1/get_action"
END_SESSION_PATH = "/v1/end_session"
API_KEY_ENV = "DREAM_API_KEY"
# The public base model ``GET <platform>/v1/models`` listed for our key on
# 2026-10-28 (with ``dreamlabs/dream-1-beta-flash``); a fine-tune is
# ``dream-1-ft:<uid8>-<suffix>``. The research page's ``dreamlabs/dream-1`` is
# not an id the gateway knows.
DEFAULT_CHECKPOINT = "dreamlabs/dream-1-beta"

# ``robot`` values: what the request's observation ``embodiment`` key says about
# whose joints the state holds. "" omits the key.
ROBOT_OTHER = "other_embodiment"
ROBOT_YAM = "yam"
VALID_ROBOTS: tuple[str, ...] = (ROBOT_OTHER, ROBOT_YAM, "")
SIDES: tuple[str, ...] = ("right", "left")
GRIPPER_CONVENTIONS: tuple[str, ...] = ("dream", "lerobot")

# Envelope codes (the status block of every reply).
CODE_OK = "DA-0000"
CODE_OK_NEW_SESSION = "DA-0001"

# Dream's gripper value for the arm slot the DK1 does not occupy: open.
ABSENT_SIDE_GRIP = 1.0

USER_AGENT = "inspect-robots-dk1-dream"

# Extra observation keys from the 7-D state: the seam for a DK1 forward-kinematics
# provider that adds ``state.<side>_eef_xyz`` (metres) and ``state.<side>_eef_rot6``
# (first two columns of the rotation) in the arm's base frame, which the Dream-1
# base models require (verified 2026-10-28: without them the gateway answers
# ``DA-4000 malformed`` naming the four missing keys). Not CLI-settable.
PoseFn = Callable[[npt.NDArray[np.float64]], Mapping[str, Any]]


class ResponseLike(Protocol):
    """The slice of ``requests.Response`` this module reads."""

    @property
    def status_code(self) -> int:
        """The HTTP status code."""
        ...

    @property
    def text(self) -> str:
        """The body as text (for error messages)."""
        ...

    def json(self) -> Any:
        """Decode the body as JSON (``ValueError`` when it is not JSON)."""
        ...


class SessionLike(Protocol):
    """The slice of ``requests.Session`` this module calls (injectable for tests)."""

    def post(
        self, url: str, *, json: Any, headers: Mapping[str, str], timeout: float
    ) -> ResponseLike:
        """POST a JSON body and return the response."""
        ...


class DreamAPIError(RuntimeError):
    """The gateway could not be reached, answered junk, or refused the request.

    ``code`` / ``reason`` / ``retryable`` / ``retry_after_s`` carry the status
    envelope when there was one (``DA-1000`` unauthorized, ``DA-2000`` no
    capacity such as ``pool_warming`` or ``rate_limited``, ``DA-4000`` bad
    request, ``DA-5000`` model failure, ``DA-6000`` Dream-side fault).
    """

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        code: str | None = None,
        reason: str | None = None,
        retryable: bool = False,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.code = code
        self.reason = reason
        self.retryable = retryable
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class DreamPolicyConfig(_FromKwargs):
    """Static configuration for the Dream API policy (all scalars: CLI ``-P`` friendly)."""

    api_key: str = ""  # empty: read DREAM_API_KEY from the environment at first use
    gateway_url: str = DEFAULT_GATEWAY_URL
    checkpoint: str = DEFAULT_CHECKPOINT
    robot: str = ROBOT_OTHER  # sent as observation["embodiment"]; "" omits it
    side: str = "right"  # the Dream arm slot the DK1 occupies
    prompt: str = ""  # overrides the scene instruction when set
    wrist_camera: str = "wrist"  # required; sent as video.wrist_<r|l>
    context_camera: str = "context"  # optional; sent as video.overhead when present
    fill_missing_cameras: bool = True  # black frames for roles the observation lacks
    frame_width: int = 320
    frame_height: int = 176
    jpeg_quality: int = 85
    gripper_convention: str = "dream"  # "dream": 0 closed..1 open on the wire; "lerobot": as is
    chunk_size: int = 48  # Dream answers 48 rows (1.6 s at 30 Hz); act() truncates to this
    timeout_s: float = 15.0
    state_key: str = STATE_KEY
    cam_height: int = 480
    cam_width: int = 640

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")
        if self.robot not in VALID_ROBOTS:
            raise ValueError(f"robot must be one of {VALID_ROBOTS}, got {self.robot!r}")
        if self.gripper_convention not in GRIPPER_CONVENTIONS:
            raise ValueError(
                f"gripper_convention must be one of {GRIPPER_CONVENTIONS}, "
                f"got {self.gripper_convention!r}"
            )
        if not self.checkpoint:
            raise ValueError("checkpoint must name a Dream model id (GET /v1/models lists them)")
        if not self.gateway_url.startswith(("http://", "https://")):
            raise ValueError(f"gateway_url must be an http(s) URL, got {self.gateway_url!r}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.frame_width < 1 or self.frame_height < 1:
            raise ValueError("frame_width and frame_height must be positive")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError(f"jpeg_quality must be in 1..95, got {self.jpeg_quality}")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive and finite, got {self.timeout_s}")

    @property
    def other_side(self) -> str:
        """The arm slot the DK1 does not occupy."""
        return "left" if self.side == "right" else "right"


def encode_frame(image: npt.ArrayLike, size: tuple[int, int], quality: int) -> str:
    """Resize an HWC RGB uint8 frame to ``size`` (width, height) and return it as base64 JPEG.

    Stretching, never cropping or padding, is how Dream resized training frames.
    """
    from PIL import Image

    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"frame must be HWC with 3 channels, got shape {arr.shape}")
    im = Image.fromarray(arr).resize(size, Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def blank_frame(size: tuple[int, int], quality: int) -> str:
    """A black frame for a camera role the DK1 does not have."""
    width, height = size
    return encode_frame(np.zeros((height, width, 3), dtype=np.uint8), size, quality)


class DreamPolicy:
    """Inspect Robots policy that drives the DK1 action space from the Dream API."""

    def __init__(
        self,
        config: DreamPolicyConfig | None = None,
        *,
        session: SessionLike | None = None,
        pose_fn: PoseFn | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else DreamPolicyConfig.from_kwargs(**flat)
        self._session = session
        self._pose_fn = pose_fn
        self._instruction: str | None = None
        self._session_id: str | None = None
        self.num_inferences = 0
        self.session_replacements = 0
        self.last_timing: Mapping[str, Any] | None = None
        self.info = PolicyInfo(
            name="dream",
            action_space=action_box(),  # semantics only; the embodiment owns limits
            observation_space=observation_space(
                self._cfg.cam_height, self._cfg.cam_width, (self._cfg.wrist_camera,)
            ),
            control_hz=None,  # the embodiment paces; advertising a rate trips a compat warning
        )
        self.config = PolicyConfig(action_horizon=self._cfg.chunk_size)

    # ----------------------------------------------------------------- plumbing

    def _http(self) -> SessionLike:
        """Lazily build the real ``requests.Session`` on first use."""
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _key(self) -> str:
        key = self._cfg.api_key or os.environ.get(API_KEY_ENV, "")
        if not key:
            raise ValueError(
                f"no Dream API key: set DreamPolicyConfig.api_key or the {API_KEY_ENV} "
                "environment variable (create one at platform.dreamlabs.ai/api-keys)"
            )
        return key

    def _request(self, path: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST one JSON body and return the decoded reply, which must carry a status envelope."""
        cfg = self._cfg
        url = cfg.gateway_url.rstrip("/") + path
        headers = {
            "Authorization": f"Bearer {self._key()}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        try:
            resp = self._http().post(url, json=body, headers=headers, timeout=cfg.timeout_s)
        except OSError as exc:  # requests exceptions derive from OSError
            raise DreamAPIError(f"request to {url} failed: {exc}") from exc
        try:
            payload = resp.json()
        except ValueError as exc:
            raise DreamAPIError(
                f"non-JSON reply from {url} (HTTP {resp.status_code}): {resp.text[:200]!r}",
                http_status=resp.status_code,
            ) from exc
        status = payload.get("status") if isinstance(payload, Mapping) else None
        if not isinstance(status, Mapping) or not isinstance(status.get("code"), str):
            raise DreamAPIError(
                f"reply from {url} carries no status envelope (HTTP {resp.status_code}); "
                "is this the DreamAPI gateway?",
                http_status=resp.status_code,
            )
        if not status["code"].startswith("DA-00"):
            raise DreamAPIError(
                _refusal_message(status), http_status=resp.status_code, **_envelope(status)
            )
        reply: Mapping[str, Any] = payload
        return reply

    # ------------------------------------------------------------ observation

    def _to_dream_grip(self, value: float) -> float:
        return 1.0 - value if self._cfg.gripper_convention == "dream" else value

    def _from_dream_grip(self, column: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1.0 - column if self._cfg.gripper_convention == "dream" else column

    def build_observation(self, observation: Observation) -> dict[str, Any]:
        """Pack an Inspect Robots observation into the request's ``observation`` dict."""
        cfg = self._cfg
        if cfg.state_key not in observation.state:
            raise ValueError(f"observation missing state key {cfg.state_key!r}")
        state = packing.validate_dim(observation.state[cfg.state_key])
        if cfg.wrist_camera not in observation.images:
            raise ValueError(f"observation missing camera {cfg.wrist_camera!r} for dream policy")
        size = (cfg.frame_width, cfg.frame_height)
        wrist = encode_frame(observation.images[cfg.wrist_camera], size, cfg.jpeg_quality)
        if cfg.context_camera in observation.images:
            overhead = encode_frame(observation.images[cfg.context_camera], size, cfg.jpeg_quality)
        elif cfg.fill_missing_cameras:
            overhead = blank_frame(size, cfg.jpeg_quality)
        else:
            raise ValueError(
                f"observation missing camera {cfg.context_camera!r} for video.overhead "
                "(set fill_missing_cameras=True to send a black frame)"
            )
        if not cfg.fill_missing_cameras:
            other_role = f"video.wrist_{cfg.other_side[0]}"
            raise ValueError(
                f"the DK1 has no {cfg.other_side} wrist camera for {other_role}; "
                "set fill_missing_cameras=True to send a black frame"
            )
        other_wrist = blank_frame(size, cfg.jpeg_quality)

        grip = self._to_dream_grip(float(state[GRIPPER_IDX]))
        arm = np.zeros(NUM_JOINTS) if cfg.robot == ROBOT_OTHER else state[:NUM_JOINTS]
        side, other = cfg.side, cfg.other_side
        prompt = cfg.prompt or self._instruction or ""
        obs: dict[str, Any] = {
            f"state.{side}_arm": [float(v) for v in arm],
            f"state.{side}_gripper": [grip],
            f"state.{side}_grip": [grip],
            f"state.{other}_arm": [0.0] * NUM_JOINTS,
            f"state.{other}_gripper": [ABSENT_SIDE_GRIP],
            f"state.{other}_grip": [ABSENT_SIDE_GRIP],
            "prompt": prompt,
            f"video.wrist_{side[0]}": [wrist],
            f"video.wrist_{other[0]}": [other_wrist],
            "video.overhead": [overhead],
        }
        if cfg.robot:
            obs["embodiment"] = cfg.robot
        if self._pose_fn is not None:
            extra = dict(self._pose_fn(state))
            clash = sorted(set(extra) & set(obs))
            if clash:
                raise ValueError(f"pose_fn returned keys the policy already sends: {clash}")
            obs.update(extra)
        return obs

    # ----------------------------------------------------------------- actions

    def chunk_from_actions(self, actions: Any) -> npt.NDArray[np.float64]:
        """Convert the reply's ``actions`` dict into an ``(N, 7)`` chunk in the DK1 contract."""
        cfg = self._cfg
        if not isinstance(actions, Mapping):
            raise ValueError(f"Dream reply 'actions' is not a dict: {type(actions).__name__}")
        per_key = actions.get("per_key")
        if isinstance(per_key, Mapping):
            arm = per_key.get(f"action.{cfg.side}_arm")
            grip = per_key.get(f"action.{cfg.side}_grip")
            if grip is None:
                grip = per_key.get(f"action.{cfg.side}_gripper")
            if arm is not None and grip is not None:
                return self._stack(
                    np.asarray(arm, dtype=np.float64), np.asarray(grip, dtype=np.float64)
                )
        q_key = f"q_{cfg.side[0]}_chunk"
        q_chunk = actions.get(q_key)
        if q_chunk is not None:
            rows = np.asarray(q_chunk, dtype=np.float64)
            if rows.size == 0:
                raise ValueError("Dream returned an empty action chunk")
            if rows.ndim != 2 or rows.shape[1] != TOTAL_DIM:
                raise ValueError(f"Dream {q_key} has shape {rows.shape}; expected (N, {TOTAL_DIM})")
            return self._stack(rows[:, :NUM_JOINTS], rows[:, NUM_JOINTS])
        per_keys = sorted(per_key) if isinstance(per_key, Mapping) else []
        raise ValueError(
            f"Dream returned no joint targets for the {cfg.side} arm (action keys "
            f"{sorted(actions)}, per_key keys {per_keys}). A Dream-1 base model asked by a "
            f"non-YAM arm (robot={cfg.robot!r}) answers end-effector poses only "
            f"(action.{cfg.side}_eef_xyz / _eef_rot6); turning those into DK1 joints needs the "
            "DK1 kinematics (Dream's arm file: URDF plus a measured tool centre point) and an IK "
            "solve, which this policy does not do. Use a checkpoint fine-tuned on DK1 data."
        )

    def _stack(
        self, arm: npt.NDArray[np.float64], grip: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if arm.size == 0:  # a JSON [] decodes to shape (0,), so check emptiness first
            raise ValueError("Dream returned an empty action chunk")
        if grip.ndim == 2 and grip.shape[1] == 1:
            grip = grip[:, 0]
        if arm.ndim != 2 or arm.shape[1] != NUM_JOINTS:
            raise ValueError(f"Dream arm chunk has shape {arm.shape}; expected (N, {NUM_JOINTS})")
        if grip.ndim != 1 or grip.shape[0] != arm.shape[0]:
            raise ValueError(
                f"Dream gripper chunk has shape {grip.shape}; expected ({arm.shape[0]},)"
            )
        return np.column_stack([arm, self._from_dream_grip(grip)])

    # ---------------------------------------------------------- Policy protocol

    def reset(self, scene: Scene) -> None:
        """Release the previous episode's session and stash the new instruction."""
        self.close()
        self._instruction = scene.instruction
        self.num_inferences = 0

    def close(self) -> None:
        """End the gateway session, best effort (the seat frees on idle eviction anyway)."""
        session_id, self._session_id = self._session_id, None
        if session_id is None:
            return
        with contextlib.suppress(DreamAPIError):
            self._request(
                END_SESSION_PATH, {"checkpoint": self._cfg.checkpoint, "session_id": session_id}
            )

    @property
    def session_id(self) -> str | None:
        """The gateway session the current episode runs under (None before the first act)."""
        return self._session_id

    def act(self, observation: Observation) -> ActionChunk:
        """One ``POST /v1/get_action``: pack, send, map the reply into the 7-D contract."""
        body = {
            "checkpoint": self._cfg.checkpoint,
            "session_id": self._session_id,  # null asks the gateway to mint a session
            "reset": False,
            "observation": self.build_observation(observation),
        }
        t0 = time.perf_counter()
        payload = self._request(GET_ACTION_PATH, body)
        elapsed = time.perf_counter() - t0

        status = payload["status"]
        returned = payload.get("session_id")
        if isinstance(returned, str) and returned:
            if (
                self._session_id is not None
                and returned != self._session_id
                and status["code"] == CODE_OK_NEW_SESSION
            ):
                self.session_replacements += 1  # the gateway restarted the episode server-side
            self._session_id = returned
        timing = payload.get("timing")
        self.last_timing = timing if isinstance(timing, Mapping) else None

        actions = self.chunk_from_actions(payload.get("actions"))[: self._cfg.chunk_size]
        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in actions],
            control_hz=None,
            inference_latency_s=elapsed,
        )


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _envelope(status: Mapping[str, Any]) -> dict[str, Any]:
    retry_after = status.get("retry_after_s")
    return {
        "code": _text(status.get("code")),
        "reason": _text(status.get("reason")),
        "retryable": bool(status.get("retryable", False)),
        "retry_after_s": float(retry_after) if isinstance(retry_after, (int, float)) else None,
    }


def _refusal_message(status: Mapping[str, Any]) -> str:
    code = str(status["code"])
    message = f"DreamAPI refused the request: {code} {_text(status.get('name')) or code}"
    reason = _text(status.get("reason"))
    detail = _text(status.get("detail"))
    if reason:
        message += f" ({reason})"
    if detail:
        message += f": {detail}"
    return message
