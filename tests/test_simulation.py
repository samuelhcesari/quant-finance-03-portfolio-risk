"""Bout en bout : dans des mondes sans alpha, le signal daté n'en trouve pas, le signal naïf si."""

import numpy as np

from quant_portfolio import config, run
from quant_portfolio.simulate import simulate

CFG = config.load()


def test_simulation_is_reproducible_and_matches_its_declared_covariance():
    a, b = simulate(CFG, seed=5), simulate(CFG, seed=5)
    assert a.prices.equals(b.prices) and a.fund.equals(b.fund)
    emp, true = a.returns.cov().values, a.truth["sigma"]
    assert np.linalg.norm(emp - true) / np.linalg.norm(true) < 0.25


def test_point_in_time_alpha_is_zero_and_naive_alpha_is_not():
    w = run.many_worlds(CFG, 40)
    s = w["scalars"]
    assert abs(s["mean_alpha_pit"]) < 2.6 * s["se_of_mean"]           # compatible avec la vérité (0)
    gap = w["alpha_naive"] - w["alpha_pit"]                           # test apparié : même monde, deux datations
    assert gap.mean() / (gap.std(ddof=1) / np.sqrt(len(gap))) > 3


def test_a_true_premium_is_recovered():
    """Puissance : si Quality rapporte vraiment 6 %/an, le backtest daté le retrouve. Face à un
    univers qui contient lui-même les titres Quality, l'alpha attendu est 6 % x (1 - part de Quality)."""
    cfg = config.load()
    cfg["simulation"]["quality_premium"] = 0.06
    shares, alphas = [], []
    for seed in range(25):
        p = run.part01(simulate(cfg, seed=seed), cfg)
        held = (p["flags"]["pit"] == 1) & p["tradable"]
        shares.append(held.sum(axis=1).mean() / p["tradable"].sum(axis=1).mean())
        alphas.append(p["active"]["pit"].mean() * 252)
    expected = 0.06 * (1 - np.mean(shares))
    assert abs(np.mean(alphas) - expected) < 3 * np.std(alphas, ddof=1) / np.sqrt(len(alphas))
    assert np.mean(alphas) > 0.03
