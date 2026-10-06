"""Partie 04 — GARCH retrouve des paramètres connus ; les tests de VaR rejettent ce qu'il faut."""

import numpy as np
import pandas as pd

from quant_portfolio import risk as rk
from quant_portfolio.simulate import garch_path

RNG = np.random.default_rng(2)


def test_garch_recovers_known_parameters():
    fit = rk.garch_fit(garch_path(12_000, 2e-6, 0.08, 0.90, RNG))
    assert abs(fit["alpha"] - 0.08) < 0.02 and abs(fit["beta"] - 0.90) < 0.025
    assert abs(np.var(fit["z"]) - 1) < 0.05                   # résidus standardisés de variance 1


def test_kupiec_accepts_the_right_rate_and_rejects_twice_too_many():
    hits = np.zeros(2000, bool)
    hits[:20] = True
    assert rk.kupiec(hits, 0.01)["p_value"] > 0.99
    hits[:40] = True
    assert rk.kupiec(hits, 0.01)["p_value"] < 0.001
    assert rk.kupiec(np.zeros(500, bool), 0.01)["p_value"] < 0.01 + 0.05     # zéro dépassement : pas de log(0)


def test_christoffersen_rejects_clustered_exceptions_only():
    spread = np.zeros(2000, bool)
    spread[::50] = True
    clustered = np.zeros(2000, bool)
    clustered[100:140] = True
    assert rk.christoffersen(spread)["p_value"] > 0.05
    assert rk.christoffersen(clustered)["p_value"] < 1e-6


def _series(n=1600):
    return pd.Series(garch_path(n, 2e-6, 0.08, 0.90, RNG), index=pd.bdate_range("2015-01-01", periods=n))


def test_var_forecast_never_uses_the_day_it_forecasts():
    r = _series()
    kw = dict(alphas=[0.01], min_window=750, refit_every=63, hs_window=500)
    base = rk.rolling_var(r, **kw)
    shocked = r.copy()
    shocked.iloc[1000:] = -0.2                                # on réécrit tout le futur à partir du jour 1000
    after = rk.rolling_var(shocked, **kw)
    for m in ("normal", "hist", "fhs"):
        cut = base[m].index.get_loc(r.index[1000])
        pd.testing.assert_frame_equal(base[m].iloc[: cut + 1], after[m].iloc[: cut + 1])


def test_filtered_var_is_calibrated_on_garch_data_and_es_exceeds_var():
    r = _series(4000)
    fc = rk.rolling_var(r, alphas=[0.05], min_window=750, refit_every=125, hs_window=500)["fhs"]
    hits = (-r.loc[fc.index] > fc["var_0.05"]).values
    assert rk.kupiec(hits, 0.05)["p_value"] > 0.01
    assert (fc["es_0.05"] > fc["var_0.05"]).all()
