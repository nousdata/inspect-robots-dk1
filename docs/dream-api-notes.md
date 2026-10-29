# Dream API notes (Dream Labs, Dream-1)

Collected 2026-10-28 for the `dream` policy in this repo. Facts only, with
where each came from. The platform's own API reference
(<https://platform.dreamlabs.ai/api-documentation>) is behind a login and was
not readable without one; the official Python client on PyPI documents the
wire format in its README and source, and that is the primary source here.

## Sources

- Research post: <https://dreamlabs.ai/research/dream-1/> (checkpoint
  `dreamlabs/dream-1`, 1.6 s action chunks, EEF default output, pricing).
- Launch coverage: <https://www.businesswire.com/news/home/20261019230613/en/Dream-Labs-Raises-$90M-Seed-to-Build-the-Brain-for-Everyday-Robots>
  (API opened 2026-10-20, $90M seed, ex-NVIDIA team).
- Official client: <https://pypi.org/project/dreamlabs-client/> (1.0.9,
  released 2026-10-23; README is the public reference for `/v1/get_action`),
  source read from the wheel (`dream/transport.py`, `dream/client.py`,
  `dream/credentials.py`, `dream/errors.py`, `dream/models.py`).
- Library with the control loop, IK and station tooling:
  <https://pypi.org/project/dreamlabs/> (1.0.18, Python 3.11 to 3.13; pulls
  jax, mujoco, mink, viser). Repository link on PyPI:
  <https://github.com/dreamlabs-co/dreamlabs>.
- Platform pages: <https://platform.dreamlabs.ai/api-keys> (create a key, shown
  once), <https://platform.dreamlabs.ai/setup-api>,
  <https://platform.dreamlabs.ai/fine-tuning>.

## Endpoints and auth

| What | Value |
| --- | --- |
| Gateway (inference) | `https://api.dreamlabs.ai` (`DREAM_GATEWAY_URL`) |
| Platform (models, keys) | `https://platform.dreamlabs.ai` (`DREAM_API_URL`) |
| Data service (embodiments, datasets, fine-tunes) | `https://data.dreamlabs.ai` (`DREAM_DATA_URL`) |
| Inference | `POST /v1/get_action` on the gateway |
| Release a session | `POST /v1/end_session` on the gateway, body `{"checkpoint", "session_id"}` |
| List runnable models | `GET /v1/models` on the platform |
| Auth | `Authorization: Bearer <key>` on every request; keys look like `dl_live_...` |
| Key sources (official client) | `api_key` argument, else `DREAM_API_KEY`, else `~/.dreamlabs/credentials` |

The official client also sends `Dream-Client-Version: <version>` and
`User-Agent: dream-client/<version> <8 hex>`; the gateway refuses an outdated
library only for checkpoints whose robot has `parts` (fine-tunes registered
with slot plans). HTTP status is not where the outcome is: every reply carries
a `status` envelope.

## Request (`POST /v1/get_action`)

```json
{
  "checkpoint": "dreamlabs/dream-1-beta",
  "session_id": null,
  "reset": false,
  "observation": { ... }
}
```

- `session_id: null` asks the gateway to mint a session; every later request
  echoes the id the reply returned. One session per episode.
- `observation` for the Dream-1 base models (YAM station layout):
  - `state.right_arm`, `state.left_arm`: float32 `[6]`, radians.
  - `state.right_gripper`, `state.left_gripper`: float32 `[1]`, **0 closed to
    1 open** (the opposite of this package's 0 open, 1 closed).
  - `state.<side>_grip`: same convention, `[1]`; models that read hand poses
    use this name; the official client sends both.
  - `state.<side>_eef_xyz` `[3]` metres and `state.<side>_eef_rot6` `[6]`
    (first two columns of the rotation matrix), the tool centre point in the
    arm's own base frame (YAM convention: x forward, y to the arm's left, z
    up). The official client computes these from the arm's URDF; the model
    anchors its answer on them.
  - `video.overhead`, `video.wrist_l`, `video.wrist_r`: each a **list** of
    base64 JPEG strings (no `data:` prefix), oldest first, up to 9 frames
    sampled at 5 fps; a single frame is accepted as a still. Any size is
    accepted; every frame is **stretched to 320x176** (width x height) server
    side, so send 320x176 at JPEG quality 85 for the fastest upload. The
    older keys `image_window.top/left/right` are deprecated.
  - `prompt`: the task string.
  - `embodiment`: `"yam"` (the state is the model's own) or
    `"other_embodiment"` (any other arm: a base model then reads none of the
    joint state, is given zeros, and acts from the cameras, the prompt and
    the hand pose).
- A **fine-tuned** model reads the keys of the embodiment config it was
  trained on instead: `state.<part>` per part of its `state` spans,
  `video.<role>` per camera role, `prompt`.

## Reply

```json
{
  "status": {"code": "DA-0000", "name": "OK", "reason": null, "retryable": false, "retry_after_s": null, "detail": null},
  "session_id": "...",
  "actions": { ... },
  "timing": {"serve_infer_ms": 412, "gateway_ms": 460}
}
```

- Codes: `DA-0000` OK; `DA-0001` OK with a new session (`reason: "new"` on a
  first call, otherwise the gateway replaced the session mid-episode);
  `DA-1000` unauthorized (`missing`, `unknown_key`); `DA-2000` no capacity
  (`pool_warming` while a model starts, `rate_limited`), with `retryable` and
  `retry_after_s`; `DA-4000` bad request (`unknown_checkpoint`, `too_large`,
  `malformed`, `invalid_session_id`, `client_outdated`); `DA-5000` the model
  failed; `DA-6000` Dream-side fault (`cannot_scale`).
- `actions` for a YAM-layout model: `q_r_chunk` and `q_l_chunk`, `[48, 7]`
  rows of six joint targets (radians) then the gripper (0 closed, 1 open);
  `q_r`, `q_l` (first rows); `per_key`.
- `actions` for a fine-tuned model: `per_key` with one `action.<part>` per
  part of its `action` spans, `[48, width]` (or `[48]` for a one-value part),
  in the dataset's units, one row per step at the dataset's frame rate.
- `actions` for a Dream-1 base model on `other_embodiment`: `per_key` carries
  absolute tool poses (`action.<side>_eef_xyz`, `action.<side>_eef_rot6`) and
  `action.<side>_grip`; `action.<side>_arm` is **null**. No joint targets.
  The official client solves them to joints with its own IK from an "arm
  file": `urdf`, `base` link, `flange` link, `tcp` (metres in the flange
  frame, the point between the closed fingertips, which the user measures;
  the README says a coding agent must never pick or estimate it), `approach`
  and `jaw` directions, `rest` joints.
- Chunk: 48 rows per call, 1.6 s of motion (the research page), so 30 rows/s.
  The client's default loop runs 20 rows/s and refills after 30 rows (async)
  or after all 48 (sync, when round trips exceed 700 ms).
- Latency (README, fine-tunes): 250 to 450 ms through the gateway once a model
  is running; a cold model is `pool_warming` for 25 to 50 minutes.

## Rate limits and pricing

- Per session: 3 calls in quick succession, then one every 0.7 s
  (`DA-2000 rate_limited`, `retry_after_s` 0.7).
- Pricing (research page): Dream-1 $3/hour, about $0.0013 per action (call);
  Dream-1-flash $2/hour, about $0.0009 per call; both quoted at 2,250 calls
  per hour. Research beta.

## Fine-tuning a DK1 body (the path to joint targets)

Register an embodiment (`dream.DreamClient().add_embodiment(name, config)`)
with spans tiling the 7-D vector, e.g.
`{"state": {"arm": [0, 6], "gripper": [6, 7]}, "action": {"arm": [0, 6], "gripper": [6, 7]}, "video": {"wrist": "observation.images.wrist", "overhead": "observation.images.context"}, "head_camera": "overhead"}`,
upload a LeRobot v2.1 dataset (`meta/info.json` with `codebase_version`
`"v2.1"`, `fps`, `features` incl. `action` and `observation.state`, videos at
`videos/chunk-NNN/<key>/<episode>.mp4`), `finetune(dataset, suffix)` (4 to
5.5 hours for 2 to 3 cameras), then run `dream-1-ft:<uid8>-<suffix>`. Its
reply is `per_key["action.arm"]` `[48, 6]` radians and
`per_key["action.gripper"]` `[48]` in the dataset's units, which map onto
this package's 7-D contract directly. Needs `pip install dreamlabs-client[datasets]`
(Python 3.10+).

## Verified against the live gateway, 2026-10-28

Through `DreamPolicy` (`requests`, no SDK), with the key from Doppler:

- `GET platform.dreamlabs.ai/v1/models`: HTTP 200, 84 ms. Runnable ids for
  this key: `dreamlabs/dream-1-beta`, `dreamlabs/dream-1-beta-flash` (owner
  `public`, `embodiment` `yam`, `servable` true, `cameras` `overhead`,
  `wrist_l`, `wrist_r`). `dreamlabs/dream-1` (the research page's id) is not
  listed.
- `POST api.dreamlabs.ai/v1/get_action` with the YAM-layout observation but
  no hand poses: HTTP 400, body `{"status": {"code": "DA-4000", "name":
  "BAD_REQUEST", "reason": "malformed", "retryable": false, "trace_id": ...,
  "detail": "inference error: observation is missing ['state.left_eef_xyz',
  'state.left_eef_rot6', 'state.right_eef_xyz', 'state.right_eef_rot6']. This
  checkpoint reads state ['state.right_arm (6 floats)', 'state.right_gripper
  (1 floats)', 'state.left_arm (6 floats)', 'state.left_gripper (1 floats)',
  'state.left_eef_xyz (3 floats)', 'state.left_eef_rot6 (6 floats)',
  'state.left_grip (1 floats)', 'state.right_eef_xyz (3 floats)',
  'state.right_eef_rot6 (6 floats)', 'state.right_grip (1 floats)'] and
  cameras ['video.overhead', 'video.wrist_l', 'video.wrist_r']: send each
  state.<part> as a list of floats, each video.<role> as a list of base64
  JPEG frames (oldest first), and prompt."}, "session_id": "<uuid>",
  "actions": null, "timing": null}`. So the base model **requires** both
  hands' poses; a session is minted even for a refused request. 3.7 s round
  trip (three 320x176 JPEGs uploaded).
- The same request with placeholder poses (the official client's example
  constants; not a DK1 pose): HTTP 200, `DA-0001 OK_NEW_SESSION` (`reason`
  `new`), `timing {"serve_infer_ms": 311, "gateway_ms": 1776}`, 1.8 s round
  trip. `actions` = `{"per_key": {"action.right_arm": null,
  "action.right_eef_xyz": [48, 3], "action.right_eef_rot6": [48, 6],
  "action.right_grip": [48], same for left}}`; no `q_r_chunk`, no `q_r`.
  Grip values ran slightly negative (-0.07 .. 0.22 on 0 closed .. 1 open).
- `POST /v1/end_session` `{"checkpoint", "session_id"}`: HTTP 200,
  `DA-0000`. Against a session that was refused: HTTP 400 `DA-4000`.

Consequence for the DK1: with the public models there is no joint output
for a non-YAM arm. The paths to joint targets are (a) a DK1 fine-tune through
the fine-tuning service (reply in `per_key` in the dataset's units), or (b)
the official client's arm-file IK (URDF at
`robotics/dk1/trlc-dk1/urdf/follower/TRLC-DK1-Follower.urdf`, plus a tool
centre point measured on the arm, which Dream's own docs say must not be
estimated by an agent).
