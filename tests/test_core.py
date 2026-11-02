import json

import pytest

from aerosynapse import DISCLAIMER, PLATFORM_TIER, ValidationTier
from aerosynapse.config import ExperimentConfig, dump_config, load_config
from aerosynapse.device import hardware_report, select_device
from aerosynapse.exceptions import ConfigurationError
from aerosynapse.experiments.record import ExperimentRecord, ExperimentRun, file_sha256
from aerosynapse.registry import SOLVERS, Registry
from aerosynapse.safety import banner

pytestmark = pytest.mark.unit


def test_config_roundtrip(tmp_path):
    cfg = ExperimentConfig(experiment_id="x1", seed=3)
    p = tmp_path / "c.yaml"
    dump_config(cfg, p)
    assert load_config(p) == cfg


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("experiment_id: a\nbogus: 1\n")
    with pytest.raises(ConfigurationError):
        load_config(p)
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "missing.yaml")


def test_record_requires_hypothesis():
    with pytest.raises(ConfigurationError):
        ExperimentRecord("id", "rq", "", "math", 0).validate()
    with pytest.raises(ConfigurationError):
        ExperimentRecord("bad id", "rq", "h", "math", 0).validate()


def test_run_writes_all_artifacts(tmp_path):
    rec = ExperimentRecord("e1", "rq", "h", "u_t = u_xx", 0)
    with ExperimentRun(rec, tmp_path, config=ExperimentConfig(experiment_id="e1")) as run:
        run.log(it=0, loss=1.0)
        run.record.results["rel_l2"] = 0.1
    d = tmp_path / "e1"
    for f in ("record.json", "metrics.json", "history.csv", "config.yaml", "git_hash.txt"):
        assert (d / f).exists()
    data = json.loads((d / "record.json").read_text())
    from aerosynapse.experiments.record import REQUIRED_FIELDS

    assert all(k in data for k in REQUIRED_FIELDS)
    assert data["status"] == "completed"
    assert len(file_sha256(d / "record.json")) == 64


def test_run_records_failure(tmp_path):
    rec = ExperimentRecord("e2", "rq", "h", "m", 0)
    with pytest.raises(RuntimeError), ExperimentRun(rec, tmp_path):
        raise RuntimeError("boom")
    assert json.loads((tmp_path / "e2" / "record.json").read_text())["status"] == "failed"


def test_registry():
    r: Registry[int] = Registry("thing")
    r.register("a")(1)
    assert r.get("a") == 1 and "a" in r and r.names() == ["a"] and list(r) == ["a"]
    with pytest.raises(KeyError):
        r.register("a")(2)
    with pytest.raises(KeyError):
        r.get("zzz")
    import aerosynapse.numerics  # noqa: F401

    assert "convdiff_fv" in SOLVERS


def test_safety_and_device():
    assert PLATFORM_TIER == ValidationTier.RESEARCH_SIMULATION
    assert "RESEARCH USE ONLY" in DISCLAIMER and "RESEARCH" in banner()
    assert "certified" not in DISCLAIMER.replace("not certified", "")
    hw = hardware_report()
    assert hw["cpu_count"] and select_device("cpu") == "cpu"
    assert select_device() in ("cpu", "cuda", "mps")
    with pytest.raises(ValueError):
        select_device("tpu")
