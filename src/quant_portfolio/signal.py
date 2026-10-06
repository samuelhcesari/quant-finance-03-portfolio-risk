"""Partie 01 — le signal Quality, daté au jour où il devient public.

Les ratios reproduisent les vues SQL du Projet 1 (`v_growth`, `v_margins`,
`v_returns`, `v_leverage`, `v_trailing_trends`) et les règles son profil
`quality.yaml`. La seule nouveauté est la DATE à laquelle on a le droit de s'en
servir : le Projet 1 rattache un ratio à la clôture de l'exercice ; un 10-K n'est
déposé que 5 à 10 semaines plus tard.

    naïf  : disponible à `period_end_date`           (regarde dans le futur)
    daté  : disponible à `filed` + 1 jour de bourse  (point-in-time)
"""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd

OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq}


def metrics(fund: pd.DataFrame) -> pd.DataFrame:
    """Ratios par exercice + les deux dates de disponibilité."""
    f = fund.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True).copy()
    g = f.groupby("ticker")
    consecutive = g.fiscal_year.shift() == f.fiscal_year - 1
    prior_rev = g.revenue.shift()
    f["revenue_growth"] = (f.revenue / prior_rev - 1).where(consecutive & (prior_rev > 0))
    f["ebitda_margin"] = f.ebitda / f.revenue.replace(0, np.nan)

    debt = f[["short_term_debt", "long_term_debt"]].sum(axis=1, min_count=1)
    invested = debt + f.total_equity - f.cash_and_equivalents
    nopat = f.ebit * (1 - f.tax_expense / f.pretax_income)
    f["roic"] = (nopat / invested).where((f.pretax_income > 0) & (invested != 0))
    f["net_debt_to_ebitda"] = (debt - f.cash_and_equivalents) / f.ebitda.replace(0, np.nan)

    g = f.groupby("ticker")
    for c in ("revenue_growth", "ebitda_margin"):       # AVG(...) OVER (ROWS 2 PRECEDING) ignore les NULL
        f[f"{c}_3y_avg"] = g[c].transform(lambda s: s.rolling(3, min_periods=1).mean())
    f["years_available_for_avg"] = g.cumcount().clip(upper=2) + 1

    f["avail_naive"] = f.period_end_date
    f["avail_pit"] = f.filed + pd.offsets.BDay(1)
    return f


def quality(m: pd.DataFrame, rules: list) -> pd.DataFrame:
    """Ajoute `rules_passed` (0..5) et `quality` (1.0 si toutes les règles passent).
    Une métrique manquante échoue (`allow_null: false` dans le Projet 1)."""
    passed = sum(OPS[op](m[col], thr).fillna(False).astype(int) for col, op, thr in rules)
    return m.assign(rules_passed=passed, quality=(passed == len(rules)).astype(float))


def asof_panel(m: pd.DataFrame, dates: pd.DatetimeIndex, avail: str, value: str = "quality",
               max_age_days: int = 550) -> pd.DataFrame:
    """Tableau dates x tickers : dernière valeur de `value` DISPONIBLE à chaque date.
    NaN tant que rien n'est disponible, ou si l'information a plus de `max_age_days`."""
    left = pd.DataFrame({"date": pd.DatetimeIndex(dates).as_unit("ns")})
    out = {}
    for tkr, d in m.dropna(subset=[avail]).groupby("ticker"):
        d = d.assign(**{avail: d[avail].dt.as_unit("ns")}).sort_values(avail)
        out[tkr] = pd.merge_asof(left, d[[avail, value]], left_on="date", right_on=avail,
                                 tolerance=pd.Timedelta(days=max_age_days))[value].values
    return pd.DataFrame(out, index=dates)


def event_study(m: pd.DataFrame, abnormal: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rendement anormal cumulé moyen autour du dépôt, selon que l'entreprise
    ENTRE dans le profil Quality, en SORT, ou ne change pas. Jour 0 = dépôt."""
    prev = m.groupby("ticker").quality.shift()
    kind = np.select([(m.quality == 1) & (prev == 0), (m.quality == 0) & (prev == 1)], ["entre", "sort"], "inchangé")
    kind = pd.Series(kind, index=m.index).where(prev.notna())
    paths: dict[str, list] = {"entre": [], "sort": [], "inchangé": []}
    idx = abnormal.index
    for tkr, filed, k in zip(m.ticker, m.filed, kind):
        if k is None or pd.isna(k) or tkr not in abnormal:
            continue
        p = idx.searchsorted(filed)
        seg = abnormal[tkr].values[p - window: p + window + 1] if p - window >= 0 else []
        if len(seg) == 2 * window + 1 and not np.isnan(seg).any():
            paths[k].append(np.cumsum(seg) - np.cumsum(seg)[window - 1])   # 0 la veille du dépôt
    out = pd.DataFrame({k: np.mean(v, axis=0) for k, v in paths.items() if v}, index=range(-window, window + 1))
    out.attrs["n"] = {k: len(v) for k, v in paths.items()}
    return out
