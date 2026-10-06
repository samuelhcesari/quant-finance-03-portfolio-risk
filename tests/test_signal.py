"""Partie 01 — les ratios reproduisent le Projet 1, et le signal daté ne voit jamais le futur."""

import numpy as np
import pandas as pd

from quant_portfolio import config, signal as sg
from quant_portfolio.simulate import simulate

CFG = config.load()
RULES = CFG["quality_rules"]


def _fund(years=(2019, 2020, 2021), **override) -> pd.DataFrame:
    n = len(years)
    base = dict(
        ticker="AAA", fiscal_year=list(years),
        period_end_date=pd.to_datetime([f"{y}-12-31" for y in years]),
        filed=pd.to_datetime([f"{y + 1}-02-20" for y in years]),
        revenue=[100.0, 110.0, 121.0][:n], ebitda=[30.0, 33.0, 36.3][:n], ebit=[25.0, 27.5, 30.25][:n],
        pretax_income=[20.0, 22.0, 24.2][:n], tax_expense=[4.0, 4.4, 4.84][:n],
        cash_and_equivalents=[10.0] * n, short_term_debt=[5.0] * n, long_term_debt=[35.0] * n, total_equity=[70.0] * n,
    )
    return pd.DataFrame(base | override)


def test_ratios_match_hand_computation():
    m = sg.metrics(_fund()).iloc[-1]
    assert np.isclose(m.revenue_growth, 0.10)
    assert np.isclose(m.ebitda_margin, 0.30)
    assert np.isclose(m.roic, 30.25 * (1 - 0.2) / (40 + 70 - 10))       # NOPAT / capital investi
    assert np.isclose(m.net_debt_to_ebitda, (40 - 10) / 36.3)
    assert np.isclose(m.revenue_growth_3y_avg, 0.10) and m.years_available_for_avg == 3


def test_growth_needs_consecutive_years_and_positive_base():
    assert np.isnan(sg.metrics(_fund(years=(2018, 2020, 2021))).revenue_growth.iloc[1])     # trou d'un exercice
    assert np.isnan(sg.metrics(_fund(revenue=[-5.0, 110.0, 121.0])).revenue_growth.iloc[1])  # base négative


def test_quality_needs_three_years_and_fails_on_missing_metric():
    q = sg.quality(sg.metrics(_fund()), RULES)
    assert list(q.quality) == [0.0, 0.0, 1.0]            # 3 exercices complets exigés
    q = sg.quality(sg.metrics(_fund(pretax_income=[20.0, 22.0, -1.0])), RULES)
    assert q.quality.iloc[-1] == 0.0                      # ROIC indéfini -> règle échouée, pas ignorée


def _panel(fund, dates, avail):
    return sg.asof_panel(sg.quality(sg.metrics(fund), RULES), dates, avail)


def test_pit_signal_is_unchanged_when_the_future_is_deleted():
    """Le test central : le signal à la date d, calculé sur toute la base, doit être
    identique au signal calculé sur une base amputée de tout dépôt postérieur à d."""
    w = simulate(CFG, seed=3)
    dates = pd.bdate_range("2012-01-02", "2024-12-31", freq="61B")
    full = _panel(w.fund, dates, "avail_pit")
    for d in dates:
        past = _panel(w.fund[w.fund.filed < d], pd.DatetimeIndex([d]), "avail_pit")
        pd.testing.assert_series_equal(full.loc[d].dropna(), past.iloc[0].dropna(), check_names=False)


def test_naive_signal_fails_the_same_check():
    """Garde-fou du test précédent : il doit détecter la fuite quand elle existe."""
    w = simulate(CFG, seed=3)
    dates = pd.bdate_range("2012-01-02", "2024-12-31", freq="61B")
    full = _panel(w.fund, dates, "avail_naive")
    leaks = 0
    for d in dates:
        past = _panel(w.fund[w.fund.filed < d], pd.DatetimeIndex([d]), "avail_naive").iloc[0]
        leaks += int((full.loc[d].fillna(-1) != past.reindex(full.columns).fillna(-1)).sum())
    assert leaks > 10


def test_signal_expires_when_stale():
    dates = pd.to_datetime(["2022-03-01", "2023-06-01", "2024-06-03"])
    p = sg.asof_panel(sg.quality(sg.metrics(_fund()), RULES), dates, "avail_pit", max_age_days=550)
    assert p["AAA"].iloc[0] == 1.0 and p["AAA"].iloc[1] == 1.0 and np.isnan(p["AAA"].iloc[2])
