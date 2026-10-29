# inspect-robots-dk1: agent guide

Inspect Robots adapter for the **TRLC-DK1** follower arm (6 joints + gripper,
LeRobot plugin `dk1_follower`) driven by **LeRobot policies**. A plugin package
in the Inspect Robots ecosystem; the framework lives in
[inspect-robots](https://github.com/robocurve/inspect-robots). This is a port
of [inspect-robots-so101](https://github.com/robocurve/inspect-robots-so101)
(RoboCurve, MIT): same module map, same seams, 7-D packing instead of 6-D,
radians instead of degrees, and a slew limit the SO adapter gets from its
driver but the DK1 driver does not provide.

## Module map (`src/inspect_robots_dk1/`)

| Module | Responsibility |
|--------|----------------|
| `packing.py` | Pure 7-D packing: motor order (`joint_1..joint_6, gripper`), `"<motor>.pos"` dict conversions, `state_spec()` (unit `rad+normalized`). No optional deps. |
| `config.py` | `DK1Config` / `LeRobotPolicyConfig` (frozen, `from_kwargs` for CLI scalars), default joint limits (TRLC control-library limits intersected with the plugin's pos_vel clips, minus 0.08 rad), `parse_camera_devices`, and the shared `action_semantics()` / `action_box()` / `observation_space()` so both components declare an identical contract. `action_semantics` carries `dim_labels` and `max_step` (the conformance kit requires labels; `max_step` is what the CLI's delta-limit guardrail derives from). `parse_start_joints` turns the CLI's `start_joints="0.0,1.03,1.31,-1.83,0.0,-0.02"` (six arm joints, radians) into the tuple the embodiment glides to at reset; the config rejects a wrong length, a non-finite entry, a target outside the joint limits, or a start pose without `max_relative_target`. |
| `operator.py` | `OperatorIO` (injectable stdin/stdout) for the readiness prompt; `default_poll_end` (real TTY poll, pragma'd). |
| `policy.py` | `LeRobotPolicy`: builds the RAW robot payload, runs the injectable `predict_fn`, truncates to `chunk_size`. `_default_predict` is the lazy torch + lerobot seam (not pragma'd; covered with `sys.modules` fakes). Refuses an empty `pretrained_path` before importing anything heavy. |
| `dream.py` | `DreamPolicy` / `DreamPolicyConfig`: Dream Labs' hosted Dream-1 over `POST /v1/get_action` with `requests` (injectable `session`), one base64 JPEG (320x176) per camera role, Dream's 0-closed..1-open gripper converted to this package's 0-open..1-closed, `per_key` / `q_<r|l>_chunk` replies mapped to the 7-D contract, and a clear `ValueError` when a base model answers end-effector poses only (no IK here). Facts and sources in `docs/dream-api-notes.md`. |
| `embodiment.py` | `DK1Embodiment`: clamp + slew backstop in `_send`, optional delta->abs, optional settle-before-observe, `SELF_PACED` pacing, operator-keypress episode end, context manager. Declares `DEVICE_SLOTS` (the serial port), `RUNTIME_REQUIREMENTS` (`lerobot`, `grpc`, `lerobot_robot_trlc_dk1`, `trlc_dk1_control`) and `info.docs`. `build_camera_configs` turns `camera_devices` into LeRobot `OpenCVCameraConfig`s (constructor injected for tests). `_default_driver_factory` (pragma'd) builds `DK1Follower(DK1FollowerConfig(...))` and calls `connect()`; there is no calibration guard because the plugin has no calibration file. |
| `preflight.py` | `build` / `run_preflight` + the `inspect-robots-dk1-preflight` CLI. |

## Invariants (do not weaken)

- `step()` always clamps to `joint_low/high` and then slew-limits the six arm
  joints to `max_relative_target` from the last observed pose, independent of
  any `Approver`. The DK1 driver has no slew limit of its own; this is the
  last line of defense.
- Policy and embodiment build their spaces from the same `config.py` helpers.
  Change dims, labels, semantics, camera names, or the state key there once.
- Construction is inert: `__init__` touches no hardware, model, or stdin.
  `assert_embodiment_conformant(DK1Embodiment().info)` and
  `inspect-robots doctor --embodiment dk1` must stay green.
- `check_compatibility(LeRobotPolicy(), DK1Embodiment())` stays at zero errors
  and zero warnings (`PolicyInfo.control_hz=None` is load-bearing).
- Coverage 100%; only hardware/TTY I/O is `# pragma: no cover`.

## Working here

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest --cov && uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Real-arm commands and what has been tested on hardware are in `README.md`
("Lab run", "Tested"). Keep "Tested" honest and dated.

## Writing style (public text)

As in the so101 repo: no em dashes in prose, bold only for definition-list
lead-ins, no decorative emoji, no slogans. Never claim a hardware result
that did not happen.
