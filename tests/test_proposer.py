"""The conformal threshold rule is checked on simulated tasks where the truth is known."""

import numpy as np

from ease.proposer import crc_threshold


def simulate_tasks(rng, n_tasks, n_links=6, mu=0.55, sigma=0.15):
    """Similarity of true links in each task; tasks differ in how easy their links are."""
    out = []
    for _ in range(n_tasks):
        centre = rng.normal(mu, 0.08)
        out.append(rng.normal(centre, sigma, size=rng.integers(1, n_links + 1)))
    return out


def miss_matrix(tasks, thresholds):
    return np.asarray([[(s < t).mean() for t in thresholds] for s in tasks])


def test_expected_miss_rate_is_controlled_over_repeated_draws():
    """The guarantee is about the average over draws of calibration set and new task. Repeat the
    whole procedure many times and check the average miss rate on the new task."""
    rng = np.random.default_rng(0)
    thresholds = np.linspace(-0.5, 1.0, 301)
    alpha, n_cal, trials = 0.10, 50, 1500
    losses = []
    for _ in range(trials):
        tasks = simulate_tasks(rng, n_cal + 1)
        res = crc_threshold(miss_matrix(tasks[:n_cal], thresholds), thresholds, alpha)
        assert res["feasible"]
        losses.append((tasks[-1] < res["threshold"]).mean())
    mean = float(np.mean(losses))
    se = float(np.std(losses) / np.sqrt(trials))
    assert mean <= alpha + 3 * se, (mean, se)
    assert mean >= alpha - 0.05, "the rule should not be needlessly conservative either"


def test_too_few_calibration_tasks_is_reported_as_infeasible():
    thresholds = np.linspace(0, 1, 11)
    res = crc_threshold(np.zeros((5, 11)), thresholds, alpha=0.05)  # 1/(n+1) = 0.167 > alpha
    assert not res["feasible"] and res["threshold"] == float("-inf")


def test_threshold_is_monotone_in_alpha():
    rng = np.random.default_rng(1)
    thresholds = np.linspace(-0.5, 1.0, 301)
    m = miss_matrix(simulate_tasks(rng, 200), thresholds)
    t = [crc_threshold(m, thresholds, a)["threshold"] for a in (0.02, 0.05, 0.10, 0.20)]
    assert t == sorted(t), "tolerating more misses can only raise the threshold"


def test_correction_term_makes_the_rule_stricter_than_the_empirical_rate():
    rng = np.random.default_rng(2)
    thresholds = np.linspace(-0.5, 1.0, 301)
    m = miss_matrix(simulate_tasks(rng, 40), thresholds)
    res = crc_threshold(m, thresholds, 0.10)
    naive = thresholds[np.where(m.mean(0) <= 0.10)[0].max()]
    assert res["threshold"] <= naive
    assert res["corrected_risk"] >= res["empirical_risk"]
