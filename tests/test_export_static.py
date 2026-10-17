import json

from experiments.export_static import build_bundle, export
from tests.test_ablation import row


def _write(d, name, rows):
    (d / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def _full(r):
    r.update({"model": "m", "usage": {"input_tokens": 1, "output_tokens": 1, "llm_calls": 1}, "latency_s": 1.0,
              "cost_usd": 0, "fault_injected": True, "legitimate": True, "false_refusal": False,
              "task_failure": False, "crashes": 0, "errors": [], "details": {}})
    return r


def test_bundle_has_everything_the_dashboard_requests(tmp_path):
    _write(tmp_path, "main", [_full(row("F2", "baseline", success=False, viol=["duplicate_action"])),
                              _full(row("F2", "idempotency+checkpoint+authz+verification+approval"))])
    _write(tmp_path, "ablation", [_full(row("F2", "idempotency"))])
    (tmp_path / "main.traces.json").write_text(json.dumps({"F2baseline0": [{"seq": 1, "t": 0, "kind": "llm"}]}))
    b = build_bundle(results_dir=tmp_path)
    assert [d["name"] for d in b["datasets"]] == ["main", "ablation"]
    assert set(b) == {"datasets", "summary", "records", "ablation", "traces"}
    assert b["summary"]["main"]["trials_total"] == 2 and len(b["records"]["main"]) == 2
    assert b["ablation"]["scenarios"][0]["id"] == "F2"
    assert b["traces"]["main"]["F2baseline0"][0]["kind"] == "llm"


def test_export_writes_site_and_space_readme(tmp_path, monkeypatch):
    res = tmp_path / "res"; res.mkdir()
    _write(res, "main", [_full(row("F2", "baseline"))])
    monkeypatch.setattr("experiments.export_static.RESULTS_DIR", res)
    import experiments.export_static as ex
    monkeypatch.setattr(ex, "build_bundle", lambda **k: build_bundle(results_dir=res))
    out = export(tmp_path / "site", space_readme=True)
    assert (out / "index.html").exists() and json.loads((out / "data.json").read_text())["datasets"]
    assert "sdk: static" in (out / "README.md").read_text()
