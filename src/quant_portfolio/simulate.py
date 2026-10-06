"""Monde simulé : mêmes tables que le monde réel, mais la vérité est connue.

Ce que l'on sait par construction, et que chaque partie doit retrouver :

* le profil Quality n'a AUCUN alpha (sauf `quality_premium` > 0) ;
* le cours saute le jour du dépôt du 10-K, proportionnellement à la surprise de
  marge — un signal daté à la clôture d'exercice « voit » donc ce saut d'avance ;
* la covariance inconditionnelle des rendements est une matrice connue ;
* le marché suit un GARCH(1,1) de paramètres connus, à queues épaisses (Student).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.world import World

TAX, DA_RATE, CASH_RATE, DEBT_COST = 0.21, 0.05, 0.08, 0.05


def garch_path(n: int, omega: float, alpha: float, beta: float, rng: np.random.Generator, df: float = 0) -> np.ndarray:
    """Innovations GARCH(1,1) de moyenne nulle ; gaussiennes, ou Student(df) réduites si df > 2."""
    z = rng.standard_t(df, n) * np.sqrt((df - 2) / df) if df else rng.standard_normal(n)
    e, h = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        e[t] = np.sqrt(h) * z[t]
        h = omega + alpha * e[t] ** 2 + beta * h
    return e


def _fundamentals(sim: dict, tickers: list[str], rng: np.random.Generator) -> pd.DataFrame:
    years = np.arange(sim["first_fiscal_year"], pd.Timestamp(sim["end"]).year)
    rows = []
    for tkr in tickers:
        margin_lvl = np.clip(rng.normal(0.22, 0.08), 0.06, 0.50)
        growth_lvl, turnover = rng.normal(0.05, 0.03), rng.uniform(0.6, 2.0)   # CA / capital investi
        leverage = rng.uniform(-0.5, 3.5)                                       # dette nette / EBITDA
        fye_month = rng.choice([12, 12, 12, 12, 6, 9])
        revenue, x = rng.uniform(2e9, 5e10), 0.0
        for fy in years:
            shock = rng.standard_normal()          # surprise de marge, en écarts-types
            x = 0.7 * x + 0.03 * shock
            revenue *= 1 + rng.normal(growth_lvl, 0.06)
            ebitda = max(margin_lvl + x, 0.01) * revenue
            ebit = ebitda - DA_RATE * revenue
            cash = CASH_RATE * revenue
            debt = max((leverage + rng.normal(0, 0.3)) * ebitda + cash, 0.0)
            pretax = ebit - DEBT_COST * debt
            end = pd.Timestamp(year=int(fy), month=int(fye_month), day=1) + pd.offsets.MonthEnd(0)
            rows.append(dict(
                ticker=tkr, fiscal_year=int(fy), period_end_date=end,
                filed=end + pd.Timedelta(days=int(rng.integers(35, 76))),
                revenue=revenue, ebitda=ebitda, ebit=ebit, pretax_income=pretax, tax_expense=TAX * pretax,
                cash_and_equivalents=cash, short_term_debt=0.0, long_term_debt=debt,
                total_equity=revenue / turnover - debt + cash, surprise=shock,
            ))
    return pd.DataFrame(rows)


def simulate(cfg: dict, seed: int | None = None) -> World:
    from quant_portfolio import signal as sg   # import local : signal ne dépend pas de simulate

    sim = cfg["simulation"]
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    sectors = pd.Series({f"{s[:3].upper()}{i:02d}": s for s in sim["sectors"] for i in range(1, sim["n_per_sector"] + 1)})
    tickers, n = list(sectors.index), len(sectors)
    dates = pd.bdate_range(sim["start"], sim["end"])
    T, dt = len(dates), 1 / 252

    fund = _fundamentals(sim, tickers, rng)

    # --- rendements : rf + beta * marché + facteur sectoriel + bruit propre + saut de dépôt
    m = sim["market"]
    var_m = m["vol"] ** 2 * dt
    mkt = m["mu"] * dt + garch_path(T, var_m * (1 - m["alpha"] - m["beta"]), m["alpha"], m["beta"], rng, m["df"])
    rf = sim["rf"] * dt
    beta = rng.uniform(0.7, 1.3, n)
    idio_vol = rng.uniform(0.18, 0.35, n) * np.sqrt(dt)
    S = pd.get_dummies(sectors).reindex(columns=sim["sectors"]).values.astype(float)   # n x secteurs
    sec = rng.standard_normal((T, S.shape[1])) * sim["sector_vol"] * np.sqrt(dt)
    r = rf + np.outer(mkt - rf, beta) + sec @ S.T + rng.standard_normal((T, n)) * idio_vol

    col = {t: i for i, t in enumerate(tickers)}
    day = np.searchsorted(dates.values, fund.filed.values)       # 1er jour de bourse >= dépôt
    ok = day < T
    np.add.at(r, (day[ok], fund.ticker.map(col).values[ok]), sim["filing_jump"] * fund.surprise.values[ok])

    if sim["quality_premium"]:      # alpha vrai, versé seulement quand le signal est PUBLIC
        flag = sg.asof_panel(sg.quality(sg.metrics(fund), cfg["quality_rules"]), dates, "avail_pit",
                             max_age_days=cfg["backtest"]["max_signal_age_days"])
        r += flag[tickers].shift(1).fillna(0).values * sim["quality_premium"] * dt

    sigma = (np.outer(beta, beta) * var_m + S @ S.T * sim["sector_vol"] ** 2 * dt
             + np.diag(idio_vol ** 2 + sim["filing_jump"] ** 2 * dt))    # ~1 dépôt par an
    prices = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=tickers)
    factors = pd.DataFrame({"MKT": mkt - rf, "RF": rf} | {f"SEC_{s}": sec[:, j] for j, s in enumerate(sim["sectors"])}, index=dates)
    truth = dict(sigma=sigma, beta=beta, garch=dict(alpha=m["alpha"], beta=m["beta"]),
                 quality_premium=sim["quality_premium"], filing_jump=sim["filing_jump"])
    return World("simulated", fund, prices, sectors, factors, truth)
