"""CPU-only baseline policy and config/graph wiring with dependency stubs."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_source(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def policy(monkeypatch, tmp_path):
    module = load_source(monkeypatch, "prep_experiment", "python/freetoken/engine/experiment.py")
    env = {"FREETOKEN_EXPERIMENT_BASELINE": "1", "FREETOKEN_FUSED_COPY": "0",
           "FREETOKEN_DISABLE_KERNEL_CACHE": "1"}
    env.update({name: str(tmp_path / name) for name in module._CACHE_VARS})
    module.os = SimpleNamespace(environ=env)
    monkeypatch.delitem(sys.modules, "freetoken.moe.offload_cache", raising=False)
    config = SimpleNamespace(cuda_graph_max_bs=0, cuda_graph_bs=None, use_dummy_weight=False,
                             moe_strategy="offload", model_path="model.gguf", dtype="bfloat16")
    return module, config, env


def test_baseline_manifest_is_config_only_and_does_not_mutate(policy, tmp_path):
    module, config, env = policy
    before = dict(env)
    result = module.validate_baseline(config)
    assert result["hardware_validation"] == "not_performed"
    assert result["fused_copy_at_config"] == "module_not_loaded"
    assert result["cuda_graph_max_bs"] == 0
    assert env == before
    assert not list(tmp_path.iterdir())
    json.dumps(result)


@pytest.mark.parametrize("value", [None, "0", "false", "off", "no"])
def test_normal_mode_preserves_existing_configuration(policy, value):
    module, config, env = policy
    if value is None:
        env.pop(module.BASELINE_ENV)
    else:
        env[module.BASELINE_ENV] = value
    config.cuda_graph_bs = [1, 2]
    env["FREETOKEN_FUSED_COPY"] = "1"
    assert module.validate_baseline(config) is None


@pytest.mark.parametrize("name", ["FREETOKEN_FUSED_COPY", "FREETOKEN_SKIP_FAST_INDEX_COPY",
    "FREETOKEN_SKIP_BANK_PIN", "FREETOKEN_DISABLE_KERNEL_CACHE_VERSION_CHECK", "FREETOKEN_DISABLE_JIT"])
def test_unsafe_flags_are_rejected(policy, name):
    module, config, env = policy
    env[name] = "1"
    with pytest.raises(ValueError, match=name):
        module.validate_baseline(config)


@pytest.mark.parametrize("value", ["", " ", "maybe", "2"])
@pytest.mark.parametrize("name", ["FREETOKEN_EXPERIMENT_BASELINE", "FREETOKEN_FUSED_COPY",
                                  "FREETOKEN_SKIP_FAST_INDEX_COPY", "FREETOKEN_SKIP_BANK_PIN"])
def test_unknown_booleans_are_not_silently_accepted(policy, name, value):
    module, config, env = policy
    env[name] = value
    with pytest.raises(ValueError, match=name):
        module.validate_baseline(config)


def test_missing_explicit_fused_off_is_rejected(policy):
    module, config, env = policy
    env.pop("FREETOKEN_FUSED_COPY")
    with pytest.raises(ValueError, match="FREETOKEN_FUSED_COPY"):
        module.validate_baseline(config)


@pytest.mark.parametrize("maximum,sizes", [(None, None), (4, None), (0, [1]), (0, [0])])
def test_graph_config_cannot_override_baseline(policy, maximum, sizes):
    module, config, _ = policy
    config.cuda_graph_max_bs, config.cuda_graph_bs = maximum, sizes
    with pytest.raises(ValueError, match="graph"):
        module.validate_baseline(config)


@pytest.mark.parametrize("sizes", [None, []])
def test_empty_graph_config_is_accepted(policy, sizes):
    module, config, _ = policy
    config.cuda_graph_bs = sizes
    assert module.validate_baseline(config) is not None


@pytest.mark.parametrize("value", [True, None])
def test_already_imported_fused_flag_requires_restart(policy, monkeypatch, value):
    module, config, _ = policy
    monkeypatch.setitem(sys.modules, "freetoken.moe.offload_cache", SimpleNamespace(_FUSED_COPY=value))
    with pytest.raises(ValueError, match="restart Python"):
        module.validate_baseline(config)


def test_imported_per_bank_flag_is_accepted(policy, monkeypatch):
    module, config, _ = policy
    monkeypatch.setitem(sys.modules, "freetoken.moe.offload_cache", SimpleNamespace(_FUSED_COPY=False))
    assert module.validate_baseline(config)["fused_copy_at_config"] is False


@pytest.mark.parametrize("name", ["TVM_FFI_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME"])
def test_relative_cache_path_is_rejected(policy, name):
    module, config, env = policy
    env[name] = "relative/cache"
    with pytest.raises(ValueError, match=name):
        module.validate_baseline(config)


def test_shared_cache_path_is_rejected(policy):
    module, config, env = policy
    env["TRITON_CACHE_DIR"] = env["TVM_FFI_CACHE_DIR"]
    with pytest.raises(ValueError, match="distinct"):
        module.validate_baseline(config)


@pytest.mark.parametrize("name,value", [("moe_strategy", "auto"), ("use_dummy_weight", True)])
def test_noncomparable_workload_is_rejected(policy, name, value):
    module, config, _ = policy
    setattr(config, name, value)
    with pytest.raises(ValueError):
        module.validate_baseline(config)


def stub_package(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__path__ = []
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def wired_config(policy, monkeypatch):
    module, _, _ = policy
    logger = Mock()
    for name in ("freetoken", "freetoken.engine", "freetoken.models", "freetoken.layers"):
        stub_package(monkeypatch, name)
    stub_package(monkeypatch, "torch")
    stub_package(monkeypatch, "freetoken.distributed", DistributedInfo=object, get_tp_info=Mock())
    stub_package(monkeypatch, "freetoken.layers.quantization", set_quant_config=Mock())
    stub_package(monkeypatch, "freetoken.models.register", _load_attr=Mock(),
                 checkpoint_quant_config=Mock(), get_model_spec=Mock())
    stub_package(monkeypatch, "freetoken.utils", cached_load_hf_config=Mock(),
                 init_logger=lambda _: logger, mem_GB=Mock())
    monkeypatch.setitem(sys.modules, "freetoken.engine.experiment", module)
    config_module = load_source(monkeypatch, "freetoken.engine.config", "python/freetoken/engine/config.py")
    return config_module.EngineConfig, logger


def test_engine_config_enforces_baseline_before_model_loading(wired_config):
    cls, _ = wired_config
    with pytest.raises(ValueError, match="graph"):
        cls(model_path="model.gguf", tp_info=object(), dtype="bfloat16", moe_strategy="offload")


def test_engine_config_emits_manifest_and_keeps_legacy_alias(wired_config):
    cls, logger = wired_config
    config = cls(model_path="model.gguf", tp_info=object(), dtype="bfloat16",
                 moe_backend="offload", cuda_graph_max_bs=0)
    assert config.moe_strategy == "offload" and config.moe_backend is None
    assert logger.info.call_args.args[0] == "FREETOKEN_BASELINE_CONFIG=%s"
    assert json.loads(logger.info.call_args.args[1])["hardware_validation"] == "not_performed"


def test_real_graph_runner_takes_no_capture_path(wired_config, monkeypatch):
    cls, _ = wired_config
    config = cls(model_path="model.gguf", tp_info=object(), dtype="bfloat16",
                 moe_strategy="offload", cuda_graph_max_bs=0)
    stub_package(monkeypatch, "freetoken.core", Batch=object, Req=object, get_global_ctx=Mock())
    stub_package(monkeypatch, "freetoken.utils.progress", emit_progress=Mock())
    stub_package(monkeypatch, "tqdm", tqdm=Mock())
    graph = load_source(monkeypatch, "freetoken.engine.graph", "python/freetoken/engine/graph.py")
    # Stub torch has no cuda methods: any allocation/capture call would fail this test.
    runner = graph.GraphRunner(stream=None, device=None, model=None, attn_backend=None,
        cuda_graph_bs=config.cuda_graph_bs, cuda_graph_max_bs=config.cuda_graph_max_bs,
        free_memory=0, max_seq_len=1, vocab_size=1, dummy_req=None)
    assert runner.graph_bs_list == [] and runner.graph_map == {}
