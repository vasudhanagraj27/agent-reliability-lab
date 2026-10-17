from experiments.analyze import ABLATION_CONFIGS, HARDENED_FULL, ablation_matrix, outcome_code


def row(sid, cfg, trial=0, success=True, viol=(), esc=False, domain="finance"):
    return {"run_id": f"{sid}{cfg}{trial}", "domain": domain, "scenario_id": sid, "config": cfg, "trial": trial,
            "title": f"title {sid}", "tags": [], "task_success": success, "safety_violations": list(viol),
            "escalated": esc}


def test_outcome_codes():
    assert outcome_code(row("F1", "x")) == "ok"
    assert outcome_code(row("F1", "x", success=False, viol=["duplicate_action"])) == "duplicate"
    assert outcome_code(row("F1", "x", success=False, viol=["unauthorized_action"])) == "unauthorized"
    assert outcome_code(row("F1", "x", success=False, viol=["unapproved_high_value_action"])) == "no_approval"
    assert outcome_code(row("F1", "x", success=False, esc=True)) == "escalated"
    assert outcome_code(row("F1", "x", success=False)) == "failed"


def test_matrix_uses_trial_zero_of_main_for_baseline_and_hardened_only():
    main = [row("F2", "baseline", 0, False, ["duplicate_action"]), row("F2", "baseline", 1, True),
            row("F2", HARDENED_FULL, 0), row("F9", "baseline", 0),          # F9 not in the ablation
            row("F2", "idempotency", 0, False)]                            # ignored: comes from ablation
    abl = [row("F2", "idempotency"), row("F2", "checkpoint", success=False, viol=["duplicate_action"]),
           row("F2", "authz", success=False, viol=["duplicate_action"]), row("F2", "verification")]
    m = ablation_matrix(main, abl)
    assert [s["id"] for s in m["scenarios"]] == ["F2"] and m["configs"] == ABLATION_CONFIGS
    c = m["cells"]
    assert c["F2|baseline"]["outcome"] == "duplicate"          # trial 0, not the later success
    assert c["F2|idempotency"]["outcome"] == "ok"              # ablation row wins over main's
    assert c["F2|checkpoint"]["outcome"] == "duplicate" and c[f"F2|{HARDENED_FULL}"]["outcome"] == "ok"
    assert m["totals"]["baseline"] == {"n": 1, "violations": 1, "success": 0}
    assert m["totals"]["checkpoint"]["violations"] == 1 and m["totals"]["verification"]["success"] == 1


def test_scenarios_ordered_finance_before_devops_numerically():
    abl = [row("D10", "authz", domain="devops"), row("F11", "authz"), row("F2", "authz"), row("D2", "authz", domain="devops")]
    assert [s["id"] for s in ablation_matrix([], abl)["scenarios"]] == ["F2", "F11", "D2", "D10"]
