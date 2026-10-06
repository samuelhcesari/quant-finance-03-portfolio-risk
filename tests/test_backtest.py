"""Partie 03 — le moteur ne triche pas sur le calendrier, et les statistiques valent leurs formules."""

import numpy as np
import pandas as pd

from quant_portfolio import backtest as bt

RNG = np.random.default_rng(0)
DAYS = pd.bdate_range("2020-01-01", periods=60)


def _returns():
    return pd.DataFrame(RNG.normal(0.0005, 0.01, (60, 3)), index=DAYS, columns=list("ABC"))


def test_weights_earn_only_from_the_next_day():
    R = _returns() * 0
    R.loc[DAYS[10], "A"] = 0.50                              # saut le jour même de la décision
    r, _ = bt.run(pd.DataFrame({"A": [1.0]}, index=[DAYS[10]]), R)
    assert r.index[0] == DAYS[11] and np.allclose(r, 0)


def test_buy_and_hold_between_rebalances_is_exact():
    R = _returns()
    w = pd.DataFrame([[0.5, 0.3, 0.2]], index=[DAYS[0]], columns=list("ABC"))
    r, _ = bt.run(w, R)
    expected = ((1 + R.iloc[1:]).prod() - 1) @ w.iloc[0]
    assert np.isclose((1 + r).prod() - 1, expected)


def test_cash_and_costs():
    R = _returns()
    w = pd.DataFrame([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], index=[DAYS[0], DAYS[30]], columns=list("ABC"))
    gross, turn = bt.run(w, R)
    net, _ = bt.run(w, R, cost_bps=20)
    assert np.isclose(turn.iloc[0], 0.5)                     # entrée : 50 % investis, 50 % liquidités
    assert turn.iloc[1] > 0.9                                # vente de A (dérivé) + achat de B
    assert np.isclose((gross - net).sum(), turn.sum() * 20e-4)


def test_psr_is_one_half_for_a_zero_mean_sample_and_grows_with_length():
    x = RNG.normal(0, 0.01, 2000)
    assert np.isclose(bt.psr(x - x.mean()), 0.5)
    y = RNG.normal(0.0005, 0.01, 4000)
    assert bt.psr(np.tile(y, 4)) > bt.psr(y) > 0.5


def test_expected_max_sharpe_matches_simulation():
    """Le maximum de N Sharpe sans talent : formule de Bailey & López de Prado contre tirage direct."""
    sims = RNG.normal(0, 0.05, (4000, 200))
    formula = np.mean([bt.expected_max_sharpe(row) for row in sims[:200]])
    assert abs(formula / sims.max(axis=1).mean() - 1) < 0.03


def test_deflated_sharpe_penalises_the_number_of_trials():
    x = RNG.normal(0.0008, 0.01, 2500)
    few, many = RNG.normal(0, 0.02, 3), RNG.normal(0, 0.02, 500)
    assert bt.deflated_sharpe(x, many) < bt.deflated_sharpe(x, few) <= 1


def test_newey_west_recovers_betas_and_matches_ols_point_estimates():
    X = RNG.normal(0, 1, (3000, 2))
    y = 0.1 + X @ np.array([0.7, -0.3]) + RNG.normal(0, 0.5, 3000)
    fit = bt.newey_west_ols(y, X)
    ols = np.linalg.lstsq(np.column_stack([np.ones(3000), X]), y, rcond=None)[0]
    assert np.allclose(fit["coef"], ols)
    assert np.all(np.abs(fit["coef"] - [0.1, 0.7, -0.3]) < 4 * fit["se"])
    assert np.allclose(fit["se"], 0.5 / np.sqrt(3000), rtol=0.15)      # bruit i.i.d. : HAC ~ MCO classique


def test_stationary_bootstrap_blocks():
    idx = bt.stationary_bootstrap(500, block=20, n_boot=200, rng=RNG)
    assert idx.min() >= 0 and idx.max() < 500
    consecutive = ((idx[:, 1:] - idx[:, :-1]) % 500 == 1).mean()
    assert abs(consecutive - (1 - 1 / 20)) < 0.01           # longueur moyenne de bloc = 20
