"""Metric definitions, checked on hand-built results where the right answer is known by inspection."""

import numpy as np

from ease.eval.harness import DISP, Results, compare, metric_parts, summarise

R, B, N = DISP["READY"], DISP["BLOCKED"], DISP["NEEDS_INFO"]


def results(name, pred, p_success=None):
    # two episodes, one action each, four scored steps per episode
    true = np.array([R, B, B, N, N, R, R, R])
    true_changed = np.array([True, True, False, True, False, True, False, False])
    ep = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array(pred)
    prev = np.concatenate([[N], pred[:3], [N], pred[4:7]])
    actions = {"ep": ep, "t": np.tile(np.arange(4), 2), "true": true, "pred": pred, "true_changed": true_changed,
               "pred_changed": pred != prev,
               "p_success": np.full(8, 0.5) if p_success is None else np.array(p_success),
               "success": true == R}
    events = {"ep": ep, "t": np.tile(np.arange(4), 2), "type": np.array(["revise"] * 8),
              "ledger_changed": np.array([True, True, False, True, False, True, True, False]),
              "tokens": np.array([100, 200, 0, 300, 0, 400, 100, 0]), "encodes": np.array([2, 4, 0, 6, 0, 8, 2, 0]),
              "nodes": np.zeros(8), "n_active": np.zeros(8), "n_pred": np.ones(8)}
    preds = {"ep": ep, "t": np.tile(np.arange(4), 2), "true": np.zeros(8, int), "pred": np.zeros(8, int),
             "p_true": np.full(8, 0.5), "true_changed": true_changed}
    return Results(name, 2, actions, preds, events)


def test_metrics_on_a_perfect_system():
    s = summarise(results("perfect", [R, B, B, N, N, R, R, R], p_success=[1, 0, 0, 0, 0, 1, 1, 1]), n_boot=50)
    assert s["disposition_accuracy"]["value"] == 1.0
    assert s["stale_decision_rate"]["value"] == 0.0 and s["stale_decision_rate"]["n"] == 4
    assert s["spurious_change_rate"]["value"] == 0.0
    assert s["false_ready_rate"]["value"] == 0.0 and s["missed_ready_rate"]["value"] == 0.0
    assert s["success_brier"]["value"] == 0.0
    assert s["tokens_per_changed_event"]["value"] == (100 + 200 + 300 + 400 + 100) / 5
    assert s["tokens_per_unchanged_event"]["value"] == 0.0
    assert abs(s["status_nll"]["value"] - np.log(2)) < 1e-12


def test_a_system_that_never_updates_is_entirely_stale():
    """It keeps saying NEEDS_INFO. Every change of the truth is missed."""
    r = results("frozen", [N] * 8)
    s = summarise(r, n_boot=50)
    # truth changes at 4 steps: to R, to B, to N, to R. Only "to N" is matched by a constant NEEDS_INFO.
    assert s["stale_decision_rate"]["value"] == 3 / 4
    assert s["spurious_change_rate"]["value"] == 0.0
    assert s["false_ready_rate"]["value"] == 0.0
    assert s["missed_ready_rate"]["value"] == 1.0
    assert s["disposition_accuracy"]["value"] == 2 / 8


def test_spurious_change_and_false_ready_are_counted_where_truth_did_not_move():
    #                 true: R  B  B  N | N  R  R  R       changed: T T F T | F T F F
    r = results("jumpy", [R, B, R, N, R, R, B, R])
    s = summarise(r, n_boot=50)
    # unchanged-truth steps: idx 2, 4, 6, 7. Predictions changed at 2 (B->R), 4 (N->R), 6 (R->B), 7 (B->R).
    assert s["spurious_change_rate"]["value"] == 4 / 4
    # truth not READY at idx 1, 2, 3, 4; predicted READY at 2 and 4
    assert s["false_ready_rate"]["value"] == 2 / 4
    assert s["stale_decision_rate"]["value"] == 0.0


def test_paired_comparison_and_cost_ratio():
    a = results("a", [R, B, B, N, N, R, R, R])
    b = results("b", [N] * 8)
    b.events["tokens"] = a.events["tokens"] * 10
    c = compare(a, b, n_boot=200)
    assert abs(c["disposition_accuracy"]["difference"] - (1.0 - 0.25)) < 1e-12
    assert c["stale_decision_rate"]["difference"] == -0.75
    assert abs(c["tokens_per_changed_event"]["ratio"] - 0.1) < 1e-12
    assert abs(c["tokens_per_changed_event"]["ratio_lo"] - 0.1) < 1e-9, "the ratio is the same in every resample"
    lo, hi = c["disposition_accuracy"]["lo"], c["disposition_accuracy"]["hi"]
    assert lo <= c["disposition_accuracy"]["difference"] <= hi


def test_metrics_are_ratios_of_sums_over_episodes():
    r = results("x", [R, B, B, N, R, R, R, R])
    num, den = metric_parts(r)["disposition_accuracy"]
    assert num.tolist() == [4.0, 3.0] and den.tolist() == [4.0, 4.0]
