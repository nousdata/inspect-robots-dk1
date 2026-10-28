# inspect-robots-dk1

Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals on a real
[TRLC-DK1](https://github.com/robot-learning-co/trlc-dk1) follower arm driven by
[LeRobot](https://github.com/huggingface/lerobot) policies.

Status: alpha. Ported from
[inspect-robots-so101](https://github.com/robocurve/inspect-robots-so101)
(RoboCurve, MIT), which is the reference for the structure, the safety design,
the injected test seams, and most of the code here. Everything below the
"Tested" heading says what has and has not run on the arm.

Inspect Robots has two swappable inputs: a `Policy` (the model) and an
`Embodiment` (the robot body + world). This package provides both for the
DK1 + LeRobot stack:

- **`dk1` embodiment**: TRLC's LeRobot plugin driver (`dk1_follower`), with a
  hard joint clamp and a per-step slew limit inside `step()`, camera frames
  validated against the configured resolution, operator-in-the-loop episode
  end, and self-paced control.
- **`lerobot` policy**: wraps a LeRobot checkpoint (ACT, SmolVLA, pi0, ...)
  and runs it in process, returning an action chunk per inference. Same
  design as the so101 policy, repacked for 7 dims.

Both declare the same 7-D joint-position contract: `joint_1 .. joint_6` in
radians, `gripper` in 0..1 (0 open, 1 closed), the cameras you configure
(default: one `wrist` camera at 640x480), packed `joint_pos` state. The
Inspect Robots compatibility check passes with zero errors and zero
warnings, and the embodiment passes the adapter conformance kit
(`inspect-robots doctor --embodiment dk1`).

## Install (on the machine that drives the arm)

The DK1 LeRobot plugin is not on PyPI; it installs from TRLC's repo and
needs Python 3.12 (LeRobot 0.6's floor). LeRobot 0.6 also gates the
observation helper this policy uses on `grpcio`, hence `lerobot[async]`.

```bash
uv pip install "inspect-robots-dk1[lerobot] @ git+https://github.com/nousdata/inspect-robots-dk1"
```

Into an existing LeRobot + plugin environment (the layout used for the first
lab run below), only the two framework pieces are needed:

```bash
uv pip install inspect-robots "lerobot[async]==0.6.1"
uv pip install -e /path/to/inspect-robots-dk1 --no-deps
```

The policy entry point is named `lerobot`, the same name inspect-robots-so101
registers. Install one of the two adapters per environment.

## Preflight: prove compatibility before any motion

```bash
inspect-robots doctor --embodiment dk1          # conformance kit + runtime requirements
inspect-robots-dk1-preflight                    # policy/embodiment compat (dims, cameras, state)
inspect-robots-dk1-preflight --task cubepick-reach --dry-run
```

`doctor` also lists any missing runtime modules (`lerobot`, `grpc`,
`lerobot_robot_trlc_dk1`, `trlc_dk1_control`) with the install command for
each. A green preflight means action dim (7), control mode (`joint_pos`),
cameras, and state keys line up. It does not prove the joint values are
interpreted the same way; see Safety.

## Lab run: ACT can-to-box on the real DK1

The checkpoint `policies/dk1_can_box_20261019_172323-20000` under
`~/Documents/Work/robotics/dk1` (ACT, 29 demos, wrist camera only at
640x480, 30 Hz, chunk 100) was trained from `lerobot-record` data in the
plugin's default `impedance` mode. Its training demos start in the folded
rest pose with the base turned toward the table (median first pose, radians:
`joint_1 0.63`, the other joints within 0.03 of zero, gripper open).

Hand on the E-stop for every command that moves the arm. `Unable to read
from joint_1 (DM4340)` on connect means wiring, or the E-stop is pressed.

```bash
cd ~/Documents/Work/robotics/dk1 && source trlc-dk1/.venv/bin/activate
python dk1_doctor.py --no-wiggle        # ports + camera indices -> dk1_session.json; never moves the arm
inspect-robots doctor --embodiment dk1  # no hardware touched
```

Take the follower port and the wrist camera index from `dk1_session.json`
(they were `/dev/cu.usbmodem<serial>` and `1` on 2026-10-23; the camera
index can change between plug-ins). Power the arm up in the folded rest pose,
E-stop released, then run the hold check (next section) once, and then:

```bash
inspect-robots run --policy lerobot --embodiment dk1 \
  --instruction "Put the can in the box" --max-steps 900 \
  -E port=/dev/cu.usbmodem<serial> -E camera_devices=wrist=1 \
  -E control_mode=impedance -E max_relative_target=0.06 \
  -P pretrained_path=$PWD/policies/dk1_can_box_20261019_172323-20000 \
  -P policy_type=act -P device=mps -P chunk_size=100
```

What happens: the embodiment connects (the impedance thread starts and holds
the current pose), prompts `Position the scene, then press Enter to start...`,
then the policy drives at 30 Hz for up to 900 steps (30 s). Press Enter
again to end the episode (`termination_reason="operator_end"`), then answer
the framework's `[y/n/partial/skip]` verdict prompt. The first inference on
`mps` takes about 6 s (model warm-up); the arm holds meanwhile. Logs land in
the default log dir; `inspect-robots view <log>` renders them.

For a first contact with the policy, run once with `-E max_relative_target=0.02`
(0.6 rad/s at 30 Hz) and `--max-steps 150` and watch the direction of the
first moves before using the defaults above. `--disable-guardrails` is never
needed; the CLI's clamp and delta-limit guardrails derive from the same
bounds and `max_step` the embodiment declares.

### Hold check (the adapters guide's step 4)

Slow policies leave seconds between action chunks. Verify once, in the
control mode you will use, that the motors hold position with no commands
arriving. This uses the plugin directly, no policy, no Inspect Robots:

```python
import time, numpy as np
from lerobot_robot_trlc_dk1 import DK1Follower, DK1FollowerConfig

robot = DK1Follower(DK1FollowerConfig(port="/dev/cu.usbmodem<serial>", control_mode="impedance"))
robot.connect()                        # impedance: holds the current pose from here on
obs0 = robot.get_observation()
robot.send_action({k: v for k, v in obs0.items() if k.endswith(".pos")})  # command "stay here"
time.sleep(5)                          # no commands for 5 s, as between policy chunks
obs1 = robot.get_observation()
drift = {k: abs(obs1[k] - obs0[k]) for k in obs0 if k.endswith(".pos")}
print("max drift (rad / gripper units):", max(drift.values()), drift)
robot.disconnect()
```

A few hundredths of a radian or less means the mode holds. Then nudge one
joint by hand against the hold to feel the stiffness before trusting it
with a policy.

## Safety

- **Hard clamp backstop.** Every command is clipped to `DK1Config.joint_low/high`
  inside `step()`, independent of any Inspect Robots `Approver`. The
  defaults are the limits TRLC's control library enforces in impedance mode
  (from `trlc_dk1_control/config.py`, derived from their URDF), intersected
  with the plugin's own `pos_vel` clips (joint_4 +-100 deg, joint_5 +-90
  deg), pulled in by 0.08 rad. Tighten them for your table.
- **Slew limit, because the driver has none.** LeRobot's SO follower limits
  each step to `max_relative_target`; the DK1 plugin does not. So the
  embodiment limits each arm joint to `max_relative_target` radians (default
  0.06, i.e. 1.8 rad/s at 30 Hz) from the last observed position on every
  command, and declares the same number as `ActionSemantics.max_step` so the
  CLI's default delta-limit guardrail agrees with it. The gripper is not
  slew-limited. `max_relative_target=None` turns the embodiment's limit off.
- **Homing is slew-limited or refused.** `home_pose` sends an interpolated
  trajectory in steps of `max_relative_target`; the config refuses a
  `home_pose` without a limit. `home_pose` is a 7-tuple, so it is Python-only
  (the CLI forwards scalars); on the CLI, pose the arm by hand first.
- **Native units, no renormalization.** Radians and 0..1 straight from
  LeRobot's postprocessor to the motors, after the clamp and slew. The
  compat check compares state keys, not units.
- **Absolute vs. delta joints: verify first.** Actions are absolute joint
  targets by default. If your checkpoint emits deltas, set
  `DK1Config(joints_are_delta=True)`. The compat check cannot tell these
  apart; confirm with a single slow jog.
- **Control modes.** `impedance` (default; MIT mode + gravity compensation
  on a 250 Hz thread, holds position between commands, "experimental" in
  TRLC's docs) or `pos_vel` (the motors' own position + velocity mode,
  stiffer). The plugin's `pos_vel` connect zeroes the gripper by closing it
  under torque: keep fingers out of the gripper on connect in that mode.
  Match the mode your checkpoint was trained in.
- **E-stop in hand** for the first run of anything, and for every change of
  `max_relative_target`, control mode, or checkpoint.

## Configuration

`DK1Config`: `port`, `control_mode` (`impedance` / `pos_vel`),
`max_gripper_torque`, `joint_velocity_scaling` (pos_vel only),
`disable_torque_on_disconnect` (pos_vel only), `cameras`, `camera_devices`
(`"wrist=1"` or `"wrist=1,context=/dev/video2"`; OpenCV index or path per
camera), `camera_fps`, `camera_rotation` (0, 90, 180, -90), `control_hz`,
`cam_height/width`, `joint_low/high`, `home_pose` (requires
`max_relative_target`), `joints_are_delta`, `max_relative_target`,
`settle_tolerance` (radians, default `None`), `settle_timeout_s`,
`settle_timeout_budget`, `camera_configs` (LeRobot `CameraConfig` objects,
Python-only, wins over `camera_devices`).

`LeRobotPolicyConfig`: `pretrained_path` (required for a real run; empty by
default because there is no public DK1 checkpoint), `policy_type`, `device`
(`cpu` default; `mps` on Apple silicon, `cuda` on a GPU box), `cameras`,
`state_key`, `chunk_size`, `cam_height/width`.

Scalar knobs are settable from the CLI with `-E key=value` / `-P key=value`.
Settling (`settle_tolerance`) works as in the so101 adapter: `step()` polls
the driver until the arm joints are within tolerance of the accepted
command, timeouts are reported in `StepResult.info`, and after
`settle_timeout_budget` timeouts settling disables itself for the trial.

### Python

```python
from inspect_robots import Scene, Task, eval, operator_scorer
from inspect_robots.approver import ClampApprover
from inspect_robots_dk1 import DK1Config, DK1Embodiment, LeRobotPolicy, LeRobotPolicyConfig


def grade_trial(record, _scene):
    record.operator_judgement = input("Outcome? [y/n/partial/skip]: ")
    record.operator_note = input("Grader note (optional): ")


task = Task(
    name="dk1-can-box",
    scenes=[Scene(id="can-box", instruction="Put the can in the box")],
    scorer=operator_scorer(),
    max_steps=900,
)
emb = DK1Embodiment(
    DK1Config(
        port="/dev/cu.usbmodem<serial>",
        camera_devices="wrist=1",
        home_pose=(0.63, 0.0, 0.03, 0.02, -0.01, -0.03, 0.03),  # the demos' median start pose
        max_relative_target=0.06,
    )
)
pol = LeRobotPolicy(
    LeRobotPolicyConfig(
        pretrained_path="policies/dk1_can_box_20261019_172323-20000",
        policy_type="act",
        device="mps",
        chunk_size=100,
    )
)
with emb:  # guarantees disconnect even if the eval raises
    (log,) = eval(task, pol, emb, approver=ClampApprover(emb.info.action_space), before_scoring=grade_trial)
print(log.status, log.results.metrics)
```

## Tested

As of 2026-10-28, without the arm:

- `uv run pytest --cov`: 159 tests, 100% coverage, no hardware, no torch, no
  lerobot, no stdin (driver, cameras, model, clock, and operator I/O are
  injected). `ruff` and `mypy --strict` clean.
- `assert_embodiment_conformant(DK1Embodiment().info)` passes;
  `inspect-robots doctor --embodiment dk1` reports "conformant (no issues)"
  in the DK1 venv (LeRobot 0.6.1, plugin 0.3.0, Python 3.12.13).
- `check_compatibility(LeRobotPolicy(), DK1Embodiment())`: no errors, no warnings.
- The real ACT can-box checkpoint loads through the policy's default
  inference seam on `mps` and on `cpu` with a synthetic observation and
  returns a (100, 7) chunk in radians + 0..1 gripper, identical on both
  devices. 5.6 s first call on mps, 16 ms after; 60 ms per call on cpu.
- `inspect-robots run ... -E port=/dev/cu.usbmodemDOESNOTEXIST` fails cleanly
  at connect with `EmbodimentFault: could not open port ...` and a log.

Not yet tested, all of it needs the arm:

- The real driver factory (`DK1Follower(...).connect()` with the
  `camera_devices` wiring), observation shapes from the real wrist camera,
  and the plugin's action echo.
- That the clamp, slew limit, and units are right on the motors: a slow jog
  first.
- The hold-between-chunks behavior in `impedance` mode (the plugin's thread
  should hold; unverified here).
- Any policy success.

## Development

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest --cov            # 100% coverage required
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Every public module, class, and function needs a docstring (Ruff D1). The
real driver and camera-config constructors are the only `# pragma: no cover`
seams besides TTY I/O; the model seam is covered with `sys.modules` fakes.

## License

[MIT](LICENSE). Derived from inspect-robots-so101, also MIT, by RoboCurve.
