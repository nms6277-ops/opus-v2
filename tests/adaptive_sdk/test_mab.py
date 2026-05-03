"""Tests for ThompsonSamplingMAB (Gaussian / Normal-Inverse-Gamma)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.adaptive_sdk.mab import NIGArm, ThompsonSamplingMAB

# ---------------------------------------------------------------------------
# NIGArm update math (closed-form check)
# ---------------------------------------------------------------------------


def test_nig_update_matches_closed_form_single_observation():
    rng = np.random.default_rng(0)
    arm = NIGArm(rng, mu=0.0, kappa=1.0, alpha=2.0, beta=1.0)
    # Feed reward = 4.0
    arm.update(4.0)
    # From Murphy 2007:
    #   mu'    = (kappa*mu + x) / (kappa + 1) = (1*0 + 4) / 2 = 2
    #   kappa' = 2
    #   alpha' = 2.5
    #   beta'  = 1 + 0.5 * (1 / 2) * (4 - 0)^2 = 1 + 4 = 5
    assert arm.mu == pytest.approx(2.0)
    assert arm.kappa == pytest.approx(2.0)
    assert arm.alpha == pytest.approx(2.5)
    assert arm.beta == pytest.approx(5.0)


def test_nig_update_sequence_matches_iterated_closed_form():
    rng = np.random.default_rng(0)
    arm = NIGArm(rng, mu=0.0, kappa=1.0, alpha=2.0, beta=1.0)
    rewards = [1.0, 2.0, 3.0, -1.0, 0.5]

    # Manual iterated update.
    mu, kappa, alpha, beta = 0.0, 1.0, 2.0, 1.0
    for r in rewards:
        new_mu = (kappa * mu + r) / (kappa + 1.0)
        new_beta = beta + 0.5 * (kappa / (kappa + 1.0)) * (r - mu) ** 2
        mu = new_mu
        beta = new_beta
        kappa += 1.0
        alpha += 0.5

    for r in rewards:
        arm.update(r)

    assert arm.mu == pytest.approx(mu)
    assert arm.kappa == pytest.approx(kappa)
    assert arm.alpha == pytest.approx(alpha)
    assert arm.beta == pytest.approx(beta)


def test_nig_posterior_mean_converges_to_true_mean():
    """After many observations drawn from N(m, s^2), posterior mu ~= m."""
    rng = np.random.default_rng(123)
    arm = NIGArm(rng, mu=0.0, kappa=1.0, alpha=2.0, beta=1.0)
    true_mu = 3.0
    data = rng.normal(loc=true_mu, scale=0.5, size=5000)
    for r in data:
        arm.update(float(r))
    assert arm.mu == pytest.approx(true_mu, abs=0.05)


# ---------------------------------------------------------------------------
# MAB selection
# ---------------------------------------------------------------------------


def test_mab_select_arm_returns_valid_index():
    mab = ThompsonSamplingMAB(n_arms=3, seed=0)
    for _ in range(100):
        assert 0 <= mab.select_arm() < 3


def test_mab_converges_to_best_arm():
    """If arm 2 yields higher rewards, TS should exploit it."""
    mab = ThompsonSamplingMAB(n_arms=3, seed=42)
    rng = np.random.default_rng(42)

    true_means = [0.0, 1.0, 3.0]
    true_std = 0.5

    counts = [0, 0, 0]
    # Train: 2000 rounds.
    for _ in range(2000):
        arm_id = mab.select_arm()
        counts[arm_id] += 1
        reward = float(rng.normal(loc=true_means[arm_id], scale=true_std))
        mab.update(arm_id, reward)

    # Best arm should dominate (at least majority of pulls in the back half).
    late_counts = [0, 0, 0]
    for _ in range(500):
        arm_id = mab.select_arm()
        late_counts[arm_id] += 1
        reward = float(rng.normal(loc=true_means[arm_id], scale=true_std))
        mab.update(arm_id, reward)

    assert late_counts[2] > late_counts[0]
    assert late_counts[2] > late_counts[1]
    # Posterior means roughly ordered.
    assert mab.arms[2].mu > mab.arms[0].mu


def test_mab_bad_arm_index_raises():
    mab = ThompsonSamplingMAB(n_arms=3, seed=0)
    with pytest.raises(IndexError):
        mab.update(5, 1.0)


def test_sample_is_finite_after_many_updates():
    rng = np.random.default_rng(0)
    arm = NIGArm(rng, mu=0.0, kappa=1.0, alpha=2.0, beta=1.0)
    for _ in range(10_000):
        arm.update(float(rng.normal()))
    s = arm.sample()
    assert math.isfinite(s)
