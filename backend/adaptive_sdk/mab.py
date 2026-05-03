"""Gaussian Thompson Sampling over a Normal-Inverse-Gamma (NIG) posterior.

Motivation
----------
Classical Thompson Sampling assumes Beta-Bernoulli (binary reward). Our reward
is continuous ``reward = pnl - fees - slippage`` (or a user-provided
transform). The conjugate prior for an unknown mean AND unknown variance of a
normal likelihood is the Normal-Inverse-Gamma, parametrized ``(mu, kappa,
alpha, beta)``:

    mu in R, kappa > 0, alpha > 0, beta > 0.

Sampling
--------
For each arm, one draw::

    sigma^2 ~ InverseGamma(alpha, beta)
    mu_s    ~ Normal(mu, sigma^2 / kappa)

We pick ``argmax_arm(mu_s)``. Numerically, ``InverseGamma(alpha, beta)``
corresponds to ``1 / Gamma(shape=alpha, rate=beta)`` i.e.
``1 / numpy.random.gamma(shape=alpha, scale=1/beta)``.

Update (conjugate, Murphy 2007, eqs. 86-89)
-------------------------------------------
Given a single scalar observation ``x``::

    mu_new    = (kappa * mu + x) / (kappa + 1)
    kappa_new = kappa + 1
    alpha_new = alpha + 1/2
    beta_new  = beta + (kappa / (2 * (kappa + 1))) * (x - mu)^2

Priors (v1)
-----------
``mu_0 = 0, kappa_0 = 1, alpha_0 = 2, beta_0 = 1`` -- weakly informative and
well-behaved numerically (alpha > 1 ensures finite posterior variance for
sigma^2 immediately; kappa_0 = 1 means the prior carries "one effective
observation").
"""

from __future__ import annotations

import math

import numpy as np


class NIGArm:
    """Normal-Inverse-Gamma posterior for a single bandit arm."""

    __slots__ = ("mu", "kappa", "alpha", "beta", "_rng")

    def __init__(
        self,
        rng: np.random.Generator,
        mu: float = 0.0,
        kappa: float = 1.0,
        alpha: float = 2.0,
        beta: float = 1.0,
    ) -> None:
        if kappa <= 0.0 or alpha <= 0.0 or beta <= 0.0:
            raise ValueError("kappa, alpha, beta must all be > 0")
        self.mu = mu
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self._rng = rng

    def sample(self) -> float:
        """One Thompson draw of ``mu`` from the current posterior."""
        # Gamma(shape=alpha, scale=1/beta) ~ Gamma(alpha, rate=beta).
        # Its reciprocal is InverseGamma(alpha, beta).
        g = self._rng.gamma(shape=self.alpha, scale=1.0 / self.beta)
        g = max(g, 1e-24)  # guard against pathological small draws
        sigma2 = 1.0 / g
        scale = math.sqrt(sigma2 / self.kappa)
        return float(self._rng.normal(loc=self.mu, scale=scale))

    def update(self, reward: float) -> None:
        """Conjugate update with a single scalar reward."""
        new_mu = (self.kappa * self.mu + reward) / (self.kappa + 1.0)
        new_kappa = self.kappa + 1.0
        new_alpha = self.alpha + 0.5
        new_beta = self.beta + 0.5 * (self.kappa / (self.kappa + 1.0)) * (reward - self.mu) ** 2
        self.mu = new_mu
        self.kappa = new_kappa
        self.alpha = new_alpha
        self.beta = new_beta

    # Handy read-only views (no setters -- arms mutate only via update()).

    @property
    def posterior_mean(self) -> float:
        return self.mu

    @property
    def posterior_variance(self) -> float:
        """E[sigma^2] under the InverseGamma(alpha, beta) marginal."""
        if self.alpha <= 1.0:
            return float("inf")
        return self.beta / (self.alpha - 1.0)


class ThompsonSamplingMAB:
    """Gaussian Thompson Sampling MAB with ``n_arms`` independent NIG arms."""

    __slots__ = ("_arms",)

    def __init__(self, n_arms: int = 3, seed: int | None = None) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        rng = np.random.default_rng(seed)
        self._arms = [NIGArm(rng) for _ in range(n_arms)]

    def select_arm(self) -> int:
        """Draw one sample per arm and return ``argmax``."""
        best_idx = 0
        best_val = self._arms[0].sample()
        for i in range(1, len(self._arms)):
            s = self._arms[i].sample()
            if s > best_val:
                best_val = s
                best_idx = i
        return best_idx

    def update(self, arm_id: int, reward: float) -> None:
        if not 0 <= arm_id < len(self._arms):
            raise IndexError(f"arm_id {arm_id} out of range (0..{len(self._arms) - 1})")
        self._arms[arm_id].update(reward)

    @property
    def arms(self) -> list[NIGArm]:
        return self._arms

    @property
    def n_arms(self) -> int:
        return len(self._arms)
