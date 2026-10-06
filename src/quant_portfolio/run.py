"""Exécute les quatre parties sur un monde, écrit `results/benchmark_<monde>.json`
et les figures `results/figures/<monde>/*.png`.

    python -m quant_portfolio.run --world simulated     # aucune donnée externe
    python -m quant_portfolio.run --world real          # après `make fetch`

Chaque fonction `partNN` renvoie un dictionnaire : les scalaires partent dans le
JSON, les séries servent aux figures.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from quant_portfolio import backtest as bt
from quant_portfolio import config
from quant_portfolio import covariance as cv
from quant_portfolio import risk as rk
from quant_portfolio import signal as sg
from quant_portfolio.simulate import garch_path, simulate
from quant_portfolio.world import World, load_real

VARIANTS = ["Équipondéré", "Inverse volatilité", "Variance minimale"]   # déclarées avant le run
PRIMARY = VARIANTS[0]


def _tstat(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(len(x)))


# --------------------------------------------------------------------------- #
# 01 — Signal daté                                                            #
# --------------------------------------------------------------------------- #
def part01(w: World, cfg: dict) -> dict:
    R, age = w.returns, cfg["backtest"]["max_signal_age_days"]
    m = sg.quality(sg.metrics(w.fund), cfg["quality_rules"])
    reb = bt.month_ends(R.index, cfg["backtest"]["start"])
    tradable = w.prices.loc[reb].notna()
    bench, _ = bt.run(bt.equal_weight(tradable), R)
    flags = {k: sg.asof_panel(m, reb, f"avail_{k}", max_age_days=age) for k in ("naive", "pit")}
    rets = {k: bt.run(bt.equal_weight((f == 1) & tradable), R)[0] for k, f in flags.items()}
    active = {k: r - bench for k, r in rets.items()}
    gap = active["naive"] - active["pit"]
    lag = (m.filed - m.period_end_date).dt.days
    events = sg.event_study(m, R.sub(R.mean(axis=1), axis=0))
    known = flags["pit"].notna() & flags["naive"].notna()
    return dict(
        metrics=m, rebalance=reb, tradable=tradable, bench=bench, flags=flags, rets=rets, active=active, events=events, lag=lag,
        scalars=dict(
            firm_years=int(len(m)), quality_share=float(m.quality.mean()),
            filing_lag_days=dict(median=float(lag.median()), p10=float(lag.quantile(0.1)), p90=float(lag.quantile(0.9))),
            decisions_using_unpublished_info=float((flags["pit"] != flags["naive"])[known].sum().sum() / known.sum().sum()),
            active_return=dict(naive=float(active["naive"].mean() * bt.ANN), pit=float(active["pit"].mean() * bt.ANN)),
            look_ahead_bias=dict(per_year=float(gap.mean() * bt.ANN), t_stat=_tstat(gap)),
            event_study_day0=dict(n=events.attrs["n"], **{k: float(events[k].loc[0]) for k in events}),
        ),
    )


def many_worlds(cfg: dict, n: int) -> dict:
    """Vérité connue : dans `n` mondes où Quality n'a pas d'alpha, le signal daté
    doit donner un alpha centré sur zéro, et le signal naïf un alpha positif."""
    rows = []
    for seed in range(n):
        p = part01(simulate(cfg, seed=seed), cfg)
        rows.append([p["active"]["naive"].mean() * bt.ANN, p["active"]["pit"].mean() * bt.ANN,
                     _tstat(p["active"]["naive"]), _tstat(p["active"]["pit"])])
    a = np.array(rows)
    return dict(alpha_naive=a[:, 0], alpha_pit=a[:, 1], scalars=dict(
        n_worlds=n, mean_alpha_naive=float(a[:, 0].mean()), mean_alpha_pit=float(a[:, 1].mean()),
        se_of_mean=float(a[:, 1].std(ddof=1) / np.sqrt(n)),
        share_worlds_naive_above_pit=float((a[:, 0] > a[:, 1]).mean()),
        # calibrage du test : sous alpha nul, |t| > 1,96 doit arriver dans ~5 % des mondes
        rejection_rate_5pct_pit=float((np.abs(a[:, 3]) > 1.96).mean()),
        rejection_rate_5pct_naive=float((np.abs(a[:, 2]) > 1.96).mean()),
    ))


# --------------------------------------------------------------------------- #
# 02 — Covariance                                                             #
# --------------------------------------------------------------------------- #
def part02(w: World, cfg: dict, p1: dict, rng: np.random.Generator) -> dict:
    c, R = cfg["covariance"], w.returns
    full = R.dropna(axis=1, thresh=int(0.95 * len(R))).dropna()
    reference = w.truth.get("sigma", cv.ledoit_wolf(full.values))     # monde réel : vérité de substitution
    known = cv.known_truth_experiment(reference, c["q_grid"], c["n_rep"], rng)

    order = w.sectors[full.columns].sort_values(kind="stable").index
    X = full[order].values[-c["window_days"]:]
    lam = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
    q = X.shape[1] / X.shape[0]
    noise, sigma2 = cv.noise_mask(lam, q)
    corr = {name: (lambda s: s / np.sqrt(np.outer(np.diag(s), np.diag(s))))(est(X)) for name, est in cv.ESTIMATORS.items()}

    walk = {}
    for win in c["walk_windows"]:
        walk[win] = {}
        for name, est in cv.ESTIMATORS.items():
            W = pd.DataFrame(0.0, index=p1["rebalance"], columns=R.columns)
            for d in p1["rebalance"]:
                block = R.loc[:d].iloc[-win:].dropna(axis=1)
                if len(block) == win:
                    W.loc[d, block.columns] = cv.gmv(est(block.values))
            r, turn = bt.run(W[W.abs().sum(axis=1) > 0], R)
            walk[win][name] = dict(vol=float(r.std() * np.sqrt(bt.ANN)), turnover=float(turn.iloc[1:].mean()))
        walk[win]["q"] = len(block.columns) / win
    return dict(known=known, eigenvalues=lam, q=q, sigma2=sigma2, noise=noise, corr=corr, order=order, walk=walk,
                scalars=dict(
                    known_truth=dict(q_grid=c["q_grid"], gmv_vol_over_oracle=known,
                                     reference="vraie covariance du monde simulé" if w.truth else "Ledoit-Wolf plein échantillon"),
                    spectrum=dict(q=q, noise_variance=sigma2, mp_edge=float(lam[noise].max()) if noise.any() else None, mp_edge_theory=cv.mp_edge(q, sigma2), largest_eigenvalue=float(lam[-1]),
                                  n_signal=int((~noise).sum()), n=len(lam)),
                    walk_forward_gmv={str(k): v for k, v in walk.items()},
                ))


# --------------------------------------------------------------------------- #
# 03 — Backtest et inférence                                                  #
# --------------------------------------------------------------------------- #
def _weights(w: World, p1: dict, variant: str) -> pd.DataFrame:
    R, mask = w.returns, (p1["flags"]["pit"] == 1) & p1["tradable"]
    if variant == "Équipondéré":
        return bt.equal_weight(mask)
    W = pd.DataFrame(0.0, index=mask.index, columns=mask.columns)
    for d in mask.index:
        block = R.loc[:d, mask.columns[mask.loc[d]]].iloc[-252:].dropna(axis=1)
        if block.shape[1] >= 2 and len(block) == 252:
            x = 1 / block.std().values if variant == "Inverse volatilité" else cv.gmv(cv.ledoit_wolf(block.values))
            W.loc[d, block.columns] = x / x.sum()
    return W


def part03(w: World, cfg: dict, p1: dict, rng: np.random.Generator) -> dict:
    R, rf, inf = w.returns, w.factors["RF"], cfg["inference"]
    cost = cfg["backtest"]["cost_bps"]
    bench_w = bt.equal_weight(p1["tradable"])
    bench, bench_turn = bt.run(bench_w, R, cost)
    strat, table = {}, {}
    for v in VARIANTS:
        r, turn = bt.run(_weights(w, p1, v), R, cost)
        strat[v] = r
        table[v] = bt.summary(r, rf, bench) | dict(turnover_per_year=float(turn.iloc[1:].sum() / (len(r) / bt.ANN)))
    table["Univers équipondéré"] = bt.summary(bench, rf) | dict(turnover_per_year=float(bench_turn.iloc[1:].sum() / (len(bench) / bt.ANN)))
    r = strat[PRIMARY]
    a = (r - bench).values
    years = len(a) / bt.ANN

    # bootstrap stationnaire : intervalle sur l'alpha, cône du hasard sur la trajectoire
    idx = bt.stationary_bootstrap(len(a), inf["block_days"], inf["n_boot"], rng)
    boot = a[idx]
    ci = np.percentile(boot.mean(axis=1) * bt.ANN, [2.5, 97.5])
    cone = np.percentile(np.cumsum(boot - a.mean(), axis=1), [2.5, 97.5], axis=0)
    p_boot = float((boot.mean(axis=1) <= 0).mean())

    trials = np.array([np.mean(strat[v] - bench) / np.std(strat[v] - bench, ddof=1) for v in VARIANTS])

    # expositions factorielles du rendement ACTIF
    fac = w.factors.drop(columns="RF").reindex(r.index).dropna()
    ols = bt.newey_west_ols((r - bench).loc[fac.index].values, fac.values)

    # laboratoire du hasard : portefeuilles tirés au sort, même nombre de titres que Quality
    Rm = (1 + R.loc[r.index].fillna(0)).groupby([r.index.year, r.index.month]).prod() - 1
    tradable = p1["tradable"].values[: len(Rm)]
    k = ((p1["flags"]["pit"] == 1) & p1["tradable"]).sum(axis=1).values[: len(Rm)]
    score = np.where(tradable, rng.random((inf["n_random"], *tradable.shape)), np.inf)
    picked = score <= np.sort(score, axis=2)[np.arange(inf["n_random"])[:, None], np.arange(len(k))[None, :], np.maximum(k - 1, 0)][:, :, None]
    picked &= (k > 0)[None, :, None]
    bench_m = (Rm.values * tradable).sum(axis=1) / tradable.sum(axis=1)
    rand = (picked * Rm.values).sum(axis=2) / np.maximum(picked.sum(axis=2), 1) - bench_m
    rand_ir = rand.mean(axis=1) / rand.std(axis=1, ddof=1) * np.sqrt(12)
    rand_psr = np.array([bt.psr(x) for x in rand])
    best = int(rand_ir.argmax())
    monthly = lambda x: ((1 + x.loc[r.index]).groupby(month_of(r)).prod() - 1).values      # brut, comme les tirages
    quality_m = monthly(p1["rets"]["pit"]) - monthly(p1["bench"])
    quality_ir = bt.sharpe(quality_m, 12)

    costs = {c: float((bt.run(_weights(w, p1, PRIMARY), R, c)[0] - bt.run(bench_w, R, c)[0]).mean() * bt.ANN)
             for c in inf["cost_grid_bps"]}
    return dict(
        strat=strat, bench=bench, cone=cone, rand_ir=rand_ir, quality_ir=quality_ir, ols=ols, factor_names=list(fac.columns), costs=costs,
        scalars=dict(
            performance=table, primary=PRIMARY, years=years,
            active_return=dict(estimate=float(a.mean() * bt.ANN), ci95=[float(ci[0]), float(ci[1])], p_value_bootstrap=p_boot),
            psr=bt.psr(a), deflated_sharpe=bt.deflated_sharpe(a, trials), n_trials=len(trials),
            min_detectable_alpha=bt.min_detectable_alpha(table[PRIMARY]["tracking_error"], years),
            factor_regression=dict(alpha_per_year=float(ols["coef"][0] * bt.ANN), alpha_t=float(ols["t"][0]), r2=ols["r2"], lags=ols["lags"],
                                   betas={n: dict(beta=float(b), t=float(t)) for n, b, t in zip(fac.columns, ols["coef"][1:], ols["t"][1:])}),
            random_strategies=dict(
                n=inf["n_random"], n_with_psr_above_95=int((rand_psr > 0.95).sum()),
                best_ir=float(rand_ir[best]), best_psr=float(rand_psr[best]),
                best_deflated_sharpe=bt.deflated_sharpe(rand[best], rand.mean(axis=1) / rand.std(axis=1, ddof=1)),
                quality_ir=quality_ir, quality_percentile=float((rand_ir < quality_ir).mean()),
            ),
            cost_sensitivity_bps=costs,
        ),
    )


def month_of(s: pd.Series) -> list:
    return [s.index.year, s.index.month]


# --------------------------------------------------------------------------- #
# 04 — Risque                                                                 #
# --------------------------------------------------------------------------- #
def part04(w: World, cfg: dict, p3: dict, rng: np.random.Generator) -> dict:
    c, r = cfg["risk"], p3["strat"][PRIMARY]
    true = dict(omega=2e-6, alpha=0.08, beta=0.90)
    fits = [rk.garch_fit(garch_path(5000, rng=rng, **true)) for _ in range(c["n_recovery"])]
    recovery = {k: dict(true=true[k], mean=float(np.mean([f[k] for f in fits])), sd=float(np.std([f[k] for f in fits])))
                for k in true}

    fc = rk.rolling_var(r, c["alphas"], c["min_window"], c["refit_every"], c["hs_window"])
    loss = -r.loc[fc["fhs"].index]
    tests = {}
    for method in ("normal", "hist", "fhs"):
        for a in c["alphas"]:
            hits = (loss > fc[method][f"var_{a}"]).values
            tests[f"{method}_{a}"] = dict(
                kupiec=rk.kupiec(hits, a), christoffersen=rk.christoffersen(hits),
                es_realised_over_forecast=float(loss[hits].mean() / fc[method][f"es_{a}"][hits].mean()) if hits.any() else None,
            )
    fit = rk.garch_fit(r.values)
    return dict(forecast=fc, loss=loss, fit=fit, returns=r, scalars=dict(
        garch_recovery=recovery,
        garch_strategy=dict(alpha=fit["alpha"], beta=fit["beta"], persistence=fit["alpha"] + fit["beta"],
                            long_run_vol=float(np.sqrt(fit["omega"] / (1 - fit["alpha"] - fit["beta"]) * bt.ANN))),
        var_backtests=tests,
    ))


# --------------------------------------------------------------------------- #
def run_all(world: str, cfg: dict | None = None, figures: bool = True) -> dict:
    cfg = cfg or config.load()
    rng = np.random.default_rng(cfg["seed"])
    w = simulate(cfg) if world == "simulated" else load_real(cfg)
    t0 = time.time()
    out = {"world": w}
    out["p1"] = part01(w, cfg)
    if world == "simulated":
        out["worlds"] = many_worlds(cfg, cfg["simulation"]["n_worlds"])
    out["p2"] = part02(w, cfg, out["p1"], rng)
    out["p3"] = part03(w, cfg, out["p1"], rng)
    out["p4"] = part04(w, cfg, out["p3"], rng)

    R = w.returns
    scalars = dict(world=world, seed=cfg["seed"],
                   universe=dict(n_assets=int(R.shape[1]), first_day=str(R.index[0].date()), last_day=str(R.index[-1].date())),
                   **{k: v["scalars"] for k, v in out.items() if k != "world"})
    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / f"benchmark_{world}.json"
    path.write_text(json.dumps(scalars, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"{path.name} écrit ({time.time() - t0:.0f} s)")
    if figures:
        from quant_portfolio import figures as fg
        fg.draw_all(out, cfg)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", choices=["simulated", "real"], default="simulated")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    run_all(args.world, figures=not args.no_figures)
