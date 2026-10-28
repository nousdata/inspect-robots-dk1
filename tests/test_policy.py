"""Tests for LeRobotPolicy (7-D DK1 packing).

The injected-``predict_fn`` tests need no torch and no lerobot. The
``_default_predict`` seam tests install ``sys.modules`` fakes for torch and the
lerobot modules the seam imports, and assert the *wiring* (which lerobot helper
is called with which arguments); real tensor behaviour is lerobot's contract.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from inspect_robots_dk1 import packing
from inspect_robots_dk1.config import LeRobotPolicyConfig
from inspect_robots_dk1.policy import LeRobotPolicy

DIM = packing.TOTAL_DIM
CKPT = "example/dk1-act-can-box"


def _policy_cfg(**over: Any) -> LeRobotPolicyConfig:
    values: dict[str, Any] = {"cam_height": 4, "cam_width": 4}
    values.update(over)
    return LeRobotPolicyConfig(**values)


def _obs(
    instruction: str | None = "do it",
    *,
    image_shape: tuple[int, ...] = (4, 4, 3),
) -> Observation:
    img = np.zeros(image_shape, dtype=np.uint8)
    return Observation(
        images={"wrist": img},
        state={"joint_pos": np.zeros(DIM)},
        instruction=instruction,
    )


def _fake_predict(actions: np.ndarray):
    captured: dict[str, Any] = {}

    def _predict(obs: Any) -> np.ndarray:
        captured["obs"] = obs
        return actions

    return _predict, captured


def test_info_and_config_zero_arg() -> None:
    pol = LeRobotPolicy()
    assert pol.info.name == "lerobot"
    assert pol.info.action_space.dim == 7
    assert pol.info.action_space.semantics is not None
    assert pol.info.action_space.semantics.control_mode == "joint_pos"
    assert pol.info.action_space.semantics.dim_labels == packing.MOTORS
    assert pol.info.control_hz is None  # load-bearing: keeps compat warning-free
    assert pol.info.observation_space.state_keys == frozenset({"joint_pos"})
    assert pol.config.action_horizon == 50


def test_act_builds_raw_lerobot_obs_and_chunk() -> None:
    actions = np.arange(2 * DIM, dtype=float).reshape(2, DIM)
    predict, captured = _fake_predict(actions)
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="put the can in the box"))
    chunk = pol.act(_obs())

    assert len(chunk) == 2
    assert np.array_equal(chunk.actions[0].data, actions[0])
    assert chunk.control_hz is None  # in-process: embodiment paces
    assert chunk.inference_latency_s is not None
    obs = captured["obs"]
    assert obs["task"] == "put the can in the box"
    assert [obs[key] for key in packing.motor_keys()] == [0.0] * DIM
    assert isinstance(obs["joint_1.pos"], float)
    assert obs["wrist"].shape == (4, 4, 3)
    assert "observation.state" not in obs  # packing is lerobot's job, not ours
    assert pol.num_inferences == 1


def test_act_uses_empty_instruction_when_none() -> None:
    predict, captured = _fake_predict(np.zeros((1, DIM)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction=None))
    pol.act(_obs(instruction=None))
    assert captured["obs"]["task"] == ""


def test_act_truncates_chunk_to_chunk_size() -> None:
    predict, _ = _fake_predict(np.arange(5 * DIM, dtype=float).reshape(5, DIM))
    pol = LeRobotPolicy(_policy_cfg(chunk_size=2), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    chunk = pol.act(_obs())
    assert len(chunk) == 2
    assert np.array_equal(chunk.actions[1].data, np.arange(DIM, 2 * DIM, dtype=float))


def test_act_empty_actions_raises() -> None:
    predict, _ = _fake_predict(np.zeros((0, DIM)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="empty action chunk"):
        pol.act(_obs())


def test_act_wrong_action_width_raises() -> None:
    predict, _ = _fake_predict(np.zeros((2, 6)))  # an SO-ARM-shaped chunk
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match=r"expected \(N, 7\)"):
        pol.act(_obs())


def test_act_missing_camera_raises() -> None:
    predict, _ = _fake_predict(np.zeros((1, DIM)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    obs = Observation(images={}, state={"joint_pos": np.zeros(DIM)})
    with pytest.raises(ValueError, match="missing camera"):
        pol.act(obs)


def test_act_missing_state_raises() -> None:
    predict, _ = _fake_predict(np.zeros((1, DIM)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    obs = Observation(images={"wrist": np.zeros((4, 4, 3), np.uint8)}, state={})
    with pytest.raises(ValueError, match="missing state key"):
        pol.act(obs)


def test_config_object_overrides_flat() -> None:
    pol = LeRobotPolicy(_policy_cfg(chunk_size=3))
    assert pol.config.action_horizon == 3


def test_flat_kwargs_build_config() -> None:
    pol = LeRobotPolicy(pretrained_path=CKPT, device="mps")
    assert pol._cfg.pretrained_path == CKPT and pol._cfg.device == "mps"


@pytest.mark.parametrize("shape", [(2, 4, 3), (4, 4, 1)])
def test_act_rejects_wrong_camera_shape(shape: tuple[int, ...]) -> None:
    predict, _ = _fake_predict(np.zeros((1, DIM)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError) as exc:
        pol.act(_obs(image_shape=shape))
    assert str(exc.value) == f"camera 'wrist' returned shape {shape}, expected (4, 4, 3)"


def test_default_predict_refuses_empty_pretrained_path() -> None:
    # No torch/lerobot needed: the check runs before any heavy import.
    pol = LeRobotPolicy(_policy_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="pretrained_path is empty"):
        pol.act(_obs())


# ---------------------------------------------------------------------------
# _default_predict seam: sys.modules fakes for torch + the lerobot modules.
# ---------------------------------------------------------------------------


class FakeTensor:
    """Minimal torch.Tensor stand-in: ndim/shape/slicing/unsqueeze/squeeze/numpy."""

    def __init__(self, arr: Any) -> None:
        self.arr = np.asarray(arr, dtype=np.float64)

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    def unsqueeze(self, dim: int) -> FakeTensor:
        return FakeTensor(np.expand_dims(self.arr, dim))

    def squeeze(self, dim: int) -> FakeTensor:
        return FakeTensor(np.squeeze(self.arr, dim))

    def __getitem__(self, idx: Any) -> FakeTensor:
        return FakeTensor(self.arr[idx])

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.arr


def _install_lerobot_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chunk: np.ndarray,
    image_features: dict[str, Any] | None = None,
    feature_layout: str = "v060",
) -> dict[str, Any]:
    """Install fakes for the v0.6.0, v0.5.1, or v0.5.0 feature layout."""
    calls: dict[str, Any] = {}
    if image_features is None:
        image_features = {"observation.images.wrist": SimpleNamespace(shape=(3, 4, 4))}

    class _FakePolicy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(image_features=image_features)
            calls["policy_config"] = self.config

        def to(self, device: str) -> None:
            calls["to_device"] = device

        def eval(self) -> None:
            calls["eval"] = True

        def predict_action_chunk(self, batch: Any) -> FakeTensor:
            calls["predict_batch"] = batch
            return FakeTensor(chunk)

    class _FakePolicyClass:
        @staticmethod
        def from_pretrained(path: str) -> _FakePolicy:
            calls["from_pretrained"] = path
            return _FakePolicy()

    def get_policy_class(name: str) -> type[_FakePolicyClass]:
        calls.setdefault("policy_types", []).append(name)
        return _FakePolicyClass

    def make_pre_post_processors(policy_cfg: Any, pretrained_path: Any = None, **kw: Any) -> Any:
        calls["processors_cfg"] = policy_cfg
        calls["processors_path"] = pretrained_path
        calls["processors_kwargs"] = kw

        def preprocessor(obs: Any) -> Any:
            calls["preprocessed"] = obs
            return {"batch": obs}

        def postprocessor(tensor: FakeTensor) -> FakeTensor:
            calls.setdefault("postprocessed", []).append(tensor)
            return tensor

        return preprocessor, postprocessor

    def hw_to_dataset_features(hw: Any, prefix: str, use_video: bool = True) -> Any:
        calls["hw_features"] = hw
        calls["hw_prefix"] = prefix
        calls["hw_use_video"] = use_video
        return {"observation.state": {"names": [k for k in hw if k.endswith(".pos")]}}

    def raw_observation_to_observation(raw: Any, feats: Any, img_feats: Any) -> Any:
        calls["raw_obs"] = raw
        calls["raw_features"] = feats
        calls["raw_image_features"] = img_feats
        return {"prepared": True}

    torch_mod = ModuleType("torch")
    torch_mod.no_grad = nullcontext  # type: ignore[attr-defined]

    def _stack(tensors: list[FakeTensor], dim: int = 0) -> FakeTensor:
        return FakeTensor(np.stack([t.arr for t in tensors], axis=dim))

    torch_mod.stack = _stack  # type: ignore[attr-defined]

    lerobot = ModuleType("lerobot")
    policies = ModuleType("lerobot.policies")
    factory = ModuleType("lerobot.policies.factory")
    factory.get_policy_class = get_policy_class  # type: ignore[attr-defined]
    factory.make_pre_post_processors = make_pre_post_processors  # type: ignore[attr-defined]
    async_inference = ModuleType("lerobot.async_inference")
    helpers = ModuleType("lerobot.async_inference.helpers")
    helpers.raw_observation_to_observation = raw_observation_to_observation  # type: ignore[attr-defined]
    datasets = ModuleType("lerobot.datasets")
    lerobot.policies = policies  # type: ignore[attr-defined]
    policies.factory = factory  # type: ignore[attr-defined]
    lerobot.async_inference = async_inference  # type: ignore[attr-defined]
    async_inference.helpers = helpers  # type: ignore[attr-defined]
    lerobot.datasets = datasets  # type: ignore[attr-defined]

    modules = {
        "torch": torch_mod,
        "lerobot": lerobot,
        "lerobot.policies": policies,
        "lerobot.policies.factory": factory,
        "lerobot.async_inference": async_inference,
        "lerobot.async_inference.helpers": helpers,
        "lerobot.datasets": datasets,
    }
    if feature_layout == "v060":
        monkeypatch.delitem(sys.modules, "lerobot.datasets.feature_utils", raising=False)
        monkeypatch.delitem(sys.modules, "lerobot.datasets.utils", raising=False)
        utils = ModuleType("lerobot.utils")
        feature_utils_mod = ModuleType("lerobot.utils.feature_utils")
        feature_utils_mod.hw_to_dataset_features = hw_to_dataset_features  # type: ignore[attr-defined]
        utils.feature_utils = feature_utils_mod  # type: ignore[attr-defined]
        lerobot.utils = utils  # type: ignore[attr-defined]
        modules["lerobot.utils"] = utils
        modules["lerobot.utils.feature_utils"] = feature_utils_mod
    elif feature_layout == "v051":
        monkeypatch.delitem(sys.modules, "lerobot.utils.feature_utils", raising=False)
        monkeypatch.delitem(sys.modules, "lerobot.utils", raising=False)
        feature_utils_mod = ModuleType("lerobot.datasets.feature_utils")
        feature_utils_mod.hw_to_dataset_features = hw_to_dataset_features  # type: ignore[attr-defined]
        datasets.feature_utils = feature_utils_mod  # type: ignore[attr-defined]
        modules["lerobot.datasets.feature_utils"] = feature_utils_mod
    elif feature_layout == "v050":
        monkeypatch.delitem(sys.modules, "lerobot.utils.feature_utils", raising=False)
        monkeypatch.delitem(sys.modules, "lerobot.utils", raising=False)
        monkeypatch.delitem(sys.modules, "lerobot.datasets.feature_utils", raising=False)
        utils_mod = ModuleType("lerobot.datasets.utils")
        utils_mod.hw_to_dataset_features = hw_to_dataset_features  # type: ignore[attr-defined]
        datasets.utils = utils_mod  # type: ignore[attr-defined]
        modules["lerobot.datasets.utils"] = utils_mod
    else:
        raise ValueError(f"unknown feature layout: {feature_layout}")

    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _seam_cfg(**over: Any) -> LeRobotPolicyConfig:
    return _policy_cfg(device="cpu", pretrained_path=CKPT, **over)


def test_default_predict_wires_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = np.arange(1 * 2 * DIM, dtype=float).reshape(1, 2, DIM)
    calls = _install_lerobot_fakes(monkeypatch, chunk=chunk)
    pol = LeRobotPolicy(_seam_cfg())  # no predict_fn: exercises the real seam
    pol.reset(Scene(id="s", instruction="grab it"))
    out = pol.act(_obs(instruction="grab it"))

    assert calls["policy_types"] == ["act"]
    assert calls["from_pretrained"] == CKPT
    assert calls["to_device"] == "cpu"
    assert calls["eval"] is True
    assert calls["processors_cfg"] is calls["policy_config"]
    assert calls["processors_path"] == CKPT
    assert calls["processors_kwargs"] == {
        "preprocessor_overrides": {"device_processor": {"device": "cpu"}},
        "postprocessor_overrides": {"device_processor": {"device": "cpu"}},
    }
    hw = calls["hw_features"]
    assert [k for k in hw if k.endswith(".pos")] == list(packing.motor_keys())
    assert all(hw[k] is float for k in packing.motor_keys())
    assert hw["wrist"] == (4, 4, 3)
    assert calls["hw_prefix"] == "observation"
    assert calls["hw_use_video"] is False
    raw = calls["raw_obs"]
    assert raw["task"] == "grab it"
    assert [raw[k] for k in packing.motor_keys()] == [0.0] * DIM
    assert raw["wrist"].shape == (4, 4, 3)
    assert calls["raw_features"] == {"observation.state": {"names": list(packing.motor_keys())}}
    assert calls["raw_image_features"] == calls["policy_config"].image_features
    assert calls["preprocessed"] == {"prepared": True}
    assert calls["predict_batch"] == {"batch": {"prepared": True}}
    assert len(calls["postprocessed"]) == 2
    assert len(out) == 2
    assert np.array_equal(out.actions[0].data, chunk[0, 0])
    assert np.array_equal(out.actions[1].data, chunk[0, 1])


def test_default_predict_unsqueezes_2d_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = np.arange(2 * DIM, dtype=float).reshape(2, DIM)  # no batch dim from the model
    _install_lerobot_fakes(monkeypatch, chunk=chunk)
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    out = pol.act(_obs())
    assert len(out) == 2
    assert np.array_equal(out.actions[1].data, chunk[1])


def test_default_predict_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, DIM)))
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    pol.act(_obs())
    pol.act(_obs())
    assert calls["policy_types"] == ["act"]  # lazy closure built exactly once
    assert pol.num_inferences == 2


def test_default_predict_rejects_camera_feature_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lerobot_fakes(
        monkeypatch,
        chunk=np.ones((1, 2, DIM)),
        image_features={"observation.images.context": SimpleNamespace(shape=(3, 4, 4))},
    )
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match=r"do not cover configured camera\(s\) \['wrist'\]"):
        pol.act(_obs())


@pytest.mark.parametrize("layout", ["v050", "v051"])
def test_default_predict_hw_features_fallback_imports(
    monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    calls = _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, DIM)), feature_layout=layout)
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    out = pol.act(_obs())
    assert len(out) == 2
    assert calls["hw_prefix"] == "observation"
