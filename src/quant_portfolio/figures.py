"""Figures des quatre parties + fiche de synthèse, dans results/figures/<monde>/.

Chaque figure porte en titre ce qu'elle montre (la conclusion), en sous-titre ce
qui est tracé. Les panneaux sont des fonctions `_p_*(ax, ...)` pour que la fiche
de synthèse réutilise exactement les mêmes tracés que les figures détaillées.
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from scipy.stats import norm

from quant_portfolio import backtest as bt
from quant_portfolio import covariance as cv
from quant_portfolio.config import FIGURES_DIR

# Palette catégorielle validée (daltonisme, contraste) — ordre fixe, jamais recyclé.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb", "#a3a29c"
SEQ = ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#2a78d6", "#0d366b"]          # 0..5 règles
EST_COLORS = {"Empirique": ORANGE, "Ledoit-Wolf": AQUA, "Marchenko-Pastur": BLUE}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False, "figure.dpi": 110,
})


def _new(title: str, subtitle: str, ncols: int = 1, nrows: int = 1, size=(9.5, 5), **kw):
    """Figure avec titre (la conclusion) et sous-titre (ce qui est tracé), replié à la largeur."""
    fig, axes = plt.subplots(nrows, ncols, figsize=size, **kw)
    lines = textwrap.wrap(subtitle, int(size[0] * 12.5))
    fig.suptitle(title, x=0.02, y=1 - 0.2 / size[1], ha="left", va="top", fontsize=13.5, fontweight="bold")
    fig.text(0.02, 1 - 0.56 / size[1], "\n".join(lines), ha="left", va="top", fontsize=10, color=MUTED, linespacing=1.35)
    fig._top = 1 - (0.72 + 0.2 * len(lines)) / size[1]
    return fig, axes


def _save(fig, world: str, name: str, layout: bool = True) -> None:
    out = FIGURES_DIR / world
    out.mkdir(parents=True, exist_ok=True)
    if layout:
        fig.tight_layout(rect=(0, 0, 1, fig._top))
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)


def _pct(ax, axis: str = "y", digits: int = 0) -> None:
    getattr(ax, f"{axis}axis").set_major_formatter(lambda v, _: f"{v * 100:.{digits}f} %".replace(".", ","))


def _fr(x: float, digits: int = 1, sign: bool = False) -> str:
    return f"{x:{'+' if sign else ''}.{digits}f}".replace(".", ",").replace("-", "−")


def _end_label(ax, x, y, text: str, dy: float = 0) -> None:
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=9, color=INK)


# --------------------------------------------------------------------------- #
# 01 — Signal daté                                                            #
# --------------------------------------------------------------------------- #
def fig_filing_gap(o: dict, world: str) -> None:
    p, w = o["p1"], o["world"]
    m, R = p["metrics"], w.returns
    prev = m.groupby("ticker").quality.shift()
    enters = m[(m.quality == 1) & (prev == 0) & (m.filed < R.index[-40]) & (m.period_end_date > R.index[40])]
    ab = R.sub(R.mean(axis=1), axis=0)
    move = [ab[t].iloc[ab.index.searchsorted(f)] for t, f in zip(enters.ticker, enters.filed)]
    ex = enters.iloc[int(np.nanargmax(move))]
    i0, i1 = R.index.searchsorted(ex.period_end_date), R.index.searchsorted(ex.filed)
    px = w.prices[ex.ticker].iloc[i0 - 30: i1 + 31]
    px = px / px.iloc[30] * 100

    fig, (a, b) = _new("Entre la clôture de l'exercice et le dépôt du 10-K, l'information n'existe pas encore",
                       f"À gauche : exemple, {ex.ticker} exercice {ex.fiscal_year}, cours en base 100 à la clôture. "
                       "À droite : délai clôture → dépôt sur tous les exercices.", ncols=2, size=(11.5, 4.8))
    a.axvspan(ex.period_end_date, ex.filed, color=ORANGE, alpha=0.10, lw=0)
    a.plot(px.index, px.values, color=INK, lw=1.6)
    for d, lab, ha in ((ex.period_end_date, "clôture (signal naïf)", "right"), (ex.filed, "dépôt (signal daté)", "left")):
        a.axvline(d, color=MUTED, lw=1)
        a.annotate(lab, (d, 1.0), xycoords=("data", "axes fraction"), xytext=(-4 if ha == "right" else 4, 4),
                   textcoords="offset points", ha=ha, va="bottom", fontsize=9, color=INK)
    mid = ex.period_end_date + (ex.filed - ex.period_end_date) / 2
    a.annotate(f"← {(ex.filed - ex.period_end_date).days} jours d'avance →", (mid, 1.0), xycoords=("data", "axes fraction"),
               xytext=(0, 18), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=MUTED)
    a.set_ylabel("Cours (base 100)")
    a.tick_params(axis="x", rotation=30)

    b.hist(p["lag"], bins=np.arange(p["lag"].min() - 0.5, p["lag"].max() + 2.5, 3), color=BLUE, edgecolor=SURFACE, lw=1)
    b.axvline(p["lag"].median(), color=INK, lw=1)
    b.annotate(f"médiane : {p['lag'].median():.0f} jours", (p["lag"].median(), 0.96), xycoords=("data", "axes fraction"),
               xytext=(6, 0), textcoords="offset points", va="top", fontsize=9)
    b.set_xlabel("Jours entre la clôture et le dépôt")
    b.set_ylabel("Nombre d'exercices")
    _save(fig, world, "01_filing_gap")


def fig_quality_heatmap(o: dict, world: str) -> None:
    p, w = o["p1"], o["world"]
    order = w.sectors.sort_values(kind="stable").index
    grid = p["metrics"].pivot(index="ticker", columns="fiscal_year", values="rules_passed").reindex(order)
    grid = grid.loc[:, grid.columns >= grid.columns.max() - 17]
    m = p["metrics"]
    flips = (m.quality != m.groupby("ticker").quality.shift())[m.groupby("ticker").cumcount() > 0].mean()
    fig, ax = _new(f"Le profil Quality change de statut dans {_fr(flips * 100, 0)} % des exercices seulement",
                   "Nombre de règles Quality satisfaites (sur 5) par entreprise et par exercice ; les 5 = profil validé.",
                   size=(10.5, 0.2 * len(grid) + 2.0))
    im = ax.imshow(np.ma.masked_invalid(grid.values), aspect="auto", cmap=ListedColormap(SEQ), norm=BoundaryNorm(np.arange(-0.5, 6), 6))
    ax.set_xticks(range(len(grid.columns)), grid.columns, fontsize=8)
    ax.set_yticks(range(len(grid)), grid.index, fontsize=7)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(grid.columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.5)
    ax.tick_params(which="both", length=0)
    bounds = np.flatnonzero(w.sectors[order].values[1:] != w.sectors[order].values[:-1])
    for y in bounds:
        ax.axhline(y + 0.5, color=INK, lw=1)
    for y0, y1 in zip(np.r_[0, bounds + 1], np.r_[bounds + 1, len(grid)]):
        ax.text(1.01, (y0 + y1 - 1) / 2, w.sectors[order].iloc[y0], transform=ax.get_yaxis_transform(), rotation=270,
                va="center", fontsize=9, color=MUTED)
    cb = fig.colorbar(im, ax=ax, ticks=range(6), pad=0.05, fraction=0.03)
    cb.outline.set_visible(False)
    cb.set_label("Règles satisfaites")
    _save(fig, world, "01_quality_heatmap")


def _p_events(ax, o: dict) -> None:
    ev = o["p1"]["events"]
    ax.axvspan(ev.index[0], 0, color=ORANGE, alpha=0.08, lw=0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axhline(0, color=MUTED, lw=0.8)
    for k, c in (("entre", BLUE), ("sort", ORANGE), ("inchangé", NEUTRAL)):
        if k in ev:
            ax.plot(ev.index, ev[k], color=c)
            _end_label(ax, ev.index[-1], ev[k].iloc[-1], f"{k} ({ev.attrs['n'][k]})")
    ax.set_xlim(ev.index[0], ev.index[-1] + 0.24 * len(ev) / 2)
    ax.set_xlabel("Jours de bourse autour du dépôt (0 = jour du dépôt)")
    ax.set_ylabel("Rendement anormal cumulé")
    _pct(ax, digits=1)


def fig_event_study(o: dict, world: str) -> None:
    d0 = o["p1"]["scalars"]["event_study_day0"]
    fig, ax = _new(f"Le jour du dépôt : {_fr(d0.get('entre', 0) * 100, 1, True)} % pour les entrants dans Quality, "
                   f"{_fr(d0.get('sort', 0) * 100, 1, True)} % pour les sortants",
                   "Rendement anormal cumulé moyen (titre moins univers équipondéré), ramené à zéro la veille du dépôt, par changement "
                   "de statut Quality (nombre d'événements entre parenthèses). Zone ombrée : ce que seul le signal naïf « sait » déjà.")
    _p_events(ax, o)
    _save(fig, world, "01_event_study")


def _p_worlds(ax, o: dict) -> None:
    mw = o["worlds"]
    bins = np.linspace(min(mw["alpha_naive"].min(), mw["alpha_pit"].min()), max(mw["alpha_naive"].max(), mw["alpha_pit"].max()), 32)
    for key, c, lab in (("alpha_pit", BLUE, "daté"), ("alpha_naive", ORANGE, "naïf")):
        ax.hist(mw[key], bins=bins, color=c, alpha=0.22, lw=0)
        ax.hist(mw[key], bins=bins, color=c, histtype="step", lw=1.8, label=f"{lab} : moyenne {_fr(mw[key].mean() * 100, 2, True)} %/an")
        ax.axvline(mw[key].mean(), color=c, lw=2.2)
    ax.axvline(0, color=INK, lw=1, label="vérité : alpha nul")
    ax.set_xlabel("Alpha mesuré (rendement actif annuel)")
    ax.set_ylabel(f"Nombre de mondes (sur {len(mw['alpha_pit'])})")
    _pct(ax, "x")
    ax.legend(loc="upper right", fontsize=8.5)


def fig_naive_vs_pit(o: dict, world: str) -> None:
    p = o["p1"]
    sim = "worlds" in o
    bias = p["scalars"]["look_ahead_bias"]["per_year"]
    title = ("Dater le signal à la clôture fabrique un alpha qui n'existe pas" if sim
             else f"Dater le signal à la clôture déplace le rendement actif de {_fr(bias * 100, 2, True)} %/an")
    fig, axes = _new(title,
                     "Rendement actif cumulé face à l'univers équipondéré, avant coûts"
                     + (" (à gauche, un monde). À droite : alpha mesuré dans des mondes simulés où le vrai alpha est nul." if sim else "."),
                     ncols=2 if sim else 1, size=(11.5, 4.8) if sim else (9.5, 5))
    ax = axes[0] if sim else axes
    for k, c, lab in (("naive", ORANGE, "naïf (clôture)"), ("pit", BLUE, "daté (dépôt)")):
        cum = p["active"][k].cumsum()
        ax.plot(cum.index, cum.values, color=c, lw=1.6, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Rendement actif cumulé")
    _pct(ax)
    ax.legend(loc="best", fontsize=9)
    if sim:
        _p_worlds(axes[1], o)
    _save(fig, world, "01_naive_vs_pit")


# --------------------------------------------------------------------------- #
# 02 — Covariance                                                             #
# --------------------------------------------------------------------------- #
def fig_spectrum(o: dict, world: str) -> None:
    p = o["p2"]
    lam, q, s2 = p["eigenvalues"], p["q"], p["sigma2"]
    edge, n_noise = cv.mp_edge(q, s2), int(p["noise"].sum())
    xmax = max(2.6 * edge, 1.0)
    fig, ax = _new(f"{n_noise} valeurs propres sur {len(lam)} sont indiscernables du bruit",
                   f"Valeurs propres de la matrice de corrélation ({len(lam)} titres, fenêtre de {round(len(lam) / q)} jours, q = {_fr(q, 2)}) "
                   "et densité de Marchenko-Pastur d'une matrice de pur bruit.")
    ax.hist(lam[lam <= xmax], bins=np.linspace(0, xmax, 42), density=True, color=NEUTRAL, edgecolor=SURFACE, lw=1, label="valeurs propres observées")
    x = np.linspace(1e-3, edge, 400)
    ax.plot(x, cv.mp_density(x, q, s2) * n_noise / len(lam), color=BLUE, label="Marchenko-Pastur (bruit)")
    ax.axvline(edge, color=INK, lw=1)
    ax.annotate(f"bord du bruit : {_fr(edge, 2)}", (edge, 0.97), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", va="top", fontsize=9)
    out = lam[lam > xmax]
    if len(out):
        ax.annotate("hors cadre →\n" + ", ".join(_fr(v) for v in out[::-1]) + f"\n(la plus grande = marché, {lam[-1] / len(lam) * 100:.0f} % de la variance)",
                    (0.985, 0.60), xycoords="axes fraction", ha="right", va="top", fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Valeur propre")
    ax.set_ylabel("Densité")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88), fontsize=9)
    _save(fig, world, "02_spectrum")


def fig_correlations(o: dict, world: str) -> None:
    p, sectors = o["p2"], o["world"].sectors
    fig, axes = _new("Corrélations avant et après nettoyage : la structure reste, le grain disparaît",
                     "Corrélations hors diagonale sur la même fenêtre, titres regroupés par secteur (blocs).", ncols=2, size=(11.5, 5.9))
    bounds = np.flatnonzero(sectors[p["order"]].values[1:] != sectors[p["order"]].values[:-1]) + 0.5
    off = ~np.eye(len(p["order"]), dtype=bool)
    lo, hi = np.percentile(p["corr"]["Empirique"][off], [1, 99])
    for ax, name in zip(axes, ("Empirique", "Marchenko-Pastur")):
        im = ax.imshow(np.ma.masked_where(~off, p["corr"][name]), cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=lo, vmax=hi)
        ax.set_title(name, loc="left", fontsize=10.5, color=INK)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for b in bounds:
            ax.axhline(b, color=SURFACE, lw=2), ax.axvline(b, color=SURFACE, lw=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(top=fig._top - 0.05, left=0.02, right=0.90, bottom=0.03, wspace=0.06)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Corrélation")
    _save(fig, world, "02_correlations", layout=False)


def _p_known(ax, o: dict, cfg: dict) -> None:
    q = cfg["covariance"]["q_grid"]
    for name, vals in o["p2"]["known"].items():
        ax.plot(q, vals, color=EST_COLORS[name], marker="o", ms=6, mec=SURFACE, mew=1.5, label=name)
        _end_label(ax, q[-1], vals[-1], _fr(vals[-1], 2), dy={"Ledoit-Wolf": 6, "Marchenko-Pastur": -6}.get(name, 0))
    ax.axhline(1, color=INK, lw=1)
    ax.annotate("oracle (covariance de référence)", (q[-1], 1), xytext=(0, -11), textcoords="offset points", ha="right", fontsize=8.5, color=MUTED)
    ax.set_ylim(bottom=1 - 0.09 * (max(max(v) for v in o["p2"]["known"].values()) - 1))
    ax.set_xlim(q[0] - 0.03, q[-1] + 0.09)
    ax.set_xlabel("q = nombre de titres / nombre d'observations")
    ax.set_ylabel("Volatilité vraie du portefeuille / oracle")
    ax.legend(loc="upper left", fontsize=9)


def fig_known_truth(o: dict, world: str, cfg: dict) -> None:
    ref = o["p2"]["scalars"]["known_truth"]["reference"]
    fig, ax = _new("Moins on a de données par titre, plus la covariance empirique coûte cher",
                   f"Volatilité vraie du portefeuille à variance minimale, rapportée à l'oracle (1 = parfait). Référence : {ref}.")
    _p_known(ax, o, cfg)
    _save(fig, world, "02_known_truth")


def fig_walk_forward(o: dict, world: str) -> None:
    walk = o["p2"]["walk"]
    wins = list(walk)
    short = walk[wins[0]]
    best = min(cv.ESTIMATORS, key=lambda n: short[n]["vol"])
    fig, axes = _new(f"Fenêtre de {wins[0]} jours : {_fr(short['Empirique']['vol'] * 100)} % de volatilité avec la covariance empirique, "
                     f"{_fr(short[best]['vol'] * 100)} % avec {best}",
                     "Portefeuille à variance minimale rebalancé chaque mois, par longueur de fenêtre d'estimation.", ncols=2, size=(11.5, 4.8))
    x, width = np.arange(len(wins)), 0.24
    for ax, key, lab, fmt in ((axes[0], "vol", "Volatilité réalisée annualisée", lambda v: f"{v * 100:.1f}".replace(".", ",")),
                              (axes[1], "turnover", "Turnover moyen par rebalancement", lambda v: _fr(v, 2))):
        for j, name in enumerate(cv.ESTIMATORS):
            vals = [walk[k][name][key] for k in wins]
            bars = ax.bar(x + (j - 1) * (width + 0.02), vals, width, color=EST_COLORS[name], label=name)
            for rect, v in zip(bars, vals):
                ax.annotate(fmt(v), (rect.get_x() + width / 2, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x, [f"{k} jours\nq = {_fr(walk[k]['q'], 2)}" for k in wins])
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    _pct(axes[0])
    axes[0].legend(loc="upper right", fontsize=9)
    _save(fig, world, "02_walk_forward")


# --------------------------------------------------------------------------- #
# 03 — Backtest                                                               #
# --------------------------------------------------------------------------- #
def fig_equity(o: dict, world: str) -> None:
    p = o["p3"]
    r, b = p["strat"][p["scalars"]["primary"]], p["bench"]
    t = p["scalars"]["performance"]
    fig, (a1, a2) = _new(f"Quality : {_fr(t[p['scalars']['primary']]['ann_return'] * 100, 1, True)} %/an — univers : "
                         f"{_fr(t['Univers équipondéré']['ann_return'] * 100, 1, True)} %/an, nets de coûts",
                         "En haut : valeur d'un portefeuille de 1 (échelle log), nette de coûts. En bas : drawdown depuis le plus haut.",
                         nrows=2, size=(9.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for s, c, lab in ((b, NEUTRAL, "Univers équipondéré"), (r, BLUE, "Quality")):
        v = (1 + s).cumprod()
        a1.plot(v.index, v.values, color=c, label=lab)
        a2.plot(s.index, bt.drawdown(s).values, color=c)
    vq, vb = (1 + r).cumprod().iloc[-1], (1 + b).cumprod().iloc[-1]
    _end_label(a1, r.index[-1], vq, "Quality", dy=7 if vq >= vb else -7)
    _end_label(a1, b.index[-1], vb, "Univers", dy=-7 if vq >= vb else 7)
    a1.set_yscale("log")
    top = max(vq, vb, (1 + r).cumprod().max(), (1 + b).cumprod().max())
    a1.set_yticks([v for v in (0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24) if v <= top * 1.1], minor=False)
    a1.yaxis.set_major_formatter(lambda v, _: _fr(v, 2).rstrip("0").rstrip(","))
    a1.minorticks_off()
    a1.set_ylabel("Valeur (départ = 1)")
    a1.legend(loc="upper left", fontsize=9)
    a1.set_xlim(right=r.index[-1] + pd.Timedelta(days=420))
    a2.set_ylabel("Drawdown")
    _pct(a2)
    _save(fig, world, "03_equity")


def _p_cone(ax, o: dict) -> None:
    p = o["p3"]
    r = p["strat"][p["scalars"]["primary"]]
    a = (r - p["bench"]).cumsum()
    ax.fill_between(a.index, p["cone"][0], p["cone"][1], color=NEUTRAL, alpha=0.25, lw=0, label="95 % des trajectoires sans alpha")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(a.index, a.values, color=BLUE, label="Quality moins univers (net de coûts)")
    ax.set_ylabel("Rendement actif cumulé")
    _pct(ax)
    ax.legend(loc="upper left", fontsize=9)


def fig_chance_cone(o: dict, world: str) -> None:
    s = o["p3"]["scalars"]
    inside = s["active_return"]["ci95"][0] <= 0 <= s["active_return"]["ci95"][1]
    fig, ax = _new("L'écart de performance de Quality " + ("reste dans le cône du hasard" if inside else "sort du cône du hasard"),
                   f"Alpha estimé : {_fr(s['active_return']['estimate'] * 100)} %/an, intervalle à 95 % [{_fr(s['active_return']['ci95'][0] * 100)} ; "
                   f"{_fr(s['active_return']['ci95'][1] * 100)}] (bootstrap stationnaire, blocs de 21 jours).")
    _p_cone(ax, o)
    _save(fig, world, "03_chance_cone")


def _p_random(ax, o: dict) -> None:
    p, s = o["p3"], o["p3"]["scalars"]["random_strategies"]
    ax.hist(p["rand_ir"], bins=40, color=NEUTRAL, edgecolor=SURFACE, lw=1)
    ax.axvline(p["quality_ir"], color=BLUE, lw=2.5)
    ax.annotate(f"Quality\n{s['quality_percentile'] * 100:.0f}ᵉ centile", (p["quality_ir"], 0.97), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", va="top", fontsize=9)
    ax.annotate(f"meilleur tirage : {_fr(s['best_ir'], 2)}\nPSR {_fr(s['best_psr'] * 100, 2)} %  →  Sharpe déflaté {_fr(s['best_deflated_sharpe'] * 100, 0)} %",
                (p["rand_ir"].max(), 0), xytext=(0, 38), textcoords="offset points", ha="right", fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("Ratio d'information annualisé (rendement actif / tracking error)")
    ax.set_ylabel("Nombre de stratégies")


def fig_random_lab(o: dict, world: str) -> None:
    s = o["p3"]["scalars"]["random_strategies"]
    fig, ax = _new(f"Sur {s['n']} stratégies tirées au sort, {s['n_with_psr_above_95']} passeraient pour « significatives »",
                   "Portefeuilles aléatoires de même taille que Quality, sans aucune information. « Significative » : PSR > 95 % "
                   "sans corriger du nombre d'essais.")
    _p_random(ax, o)
    _save(fig, world, "03_random_lab")


def fig_factors(o: dict, world: str) -> None:
    p, s = o["p3"], o["p3"]["scalars"]["factor_regression"]
    names, coef, se = p["factor_names"], p["ols"]["coef"][1:], p["ols"]["se"][1:]
    fig, ax = _new("Expositions factorielles du rendement actif de Quality",
                   f"Bêtas (MCO, intervalles à 95 % Newey-West, {s['lags']} retards). Alpha résiduel : {_fr(s['alpha_per_year'] * 100)} %/an "
                   f"(t = {_fr(s['alpha_t'], 2)}), R² = {_fr(s['r2'] * 100)} %.", size=(9.5, 2.4 + 0.5 * len(names)))
    y = np.arange(len(names))[::-1]
    ax.barh(y, coef, height=0.45, color=BLUE)
    ax.errorbar(coef, y, xerr=1.96 * se, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel("Bêta du rendement actif")
    for yi, c, e in zip(y, coef, se):
        ax.annotate(_fr(c, 3), (c + np.sign(c) * 1.96 * e, yi), xytext=(6 * (1 if c >= 0 else -1), 0), textcoords="offset points",
                    ha="left" if c >= 0 else "right", va="center", fontsize=9)
    ax.margins(x=0.25)
    _save(fig, world, "03_factors")


def fig_costs(o: dict, world: str) -> None:
    costs = o["p3"]["costs"]
    x, y = list(costs), list(costs.values())
    fig, ax = _new("Ce que les coûts de transaction retirent au rendement actif",
                   "Rendement actif annuel net de Quality selon le coût par unité de turnover (appliqué aussi à l'univers).")
    ax.plot(x, y, color=BLUE, marker="o", ms=7, mec=SURFACE, mew=1.5)
    ax.axhline(0, color=INK, lw=1)
    for xi, yi in zip(x, y):
        ax.annotate(_fr(yi * 100, 2) + " %", (xi, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel("Coût de transaction (points de base)")
    ax.set_ylabel("Rendement actif net par an")
    ax.margins(y=0.35)
    _pct(ax, digits=1)
    _save(fig, world, "03_costs")


# --------------------------------------------------------------------------- #
# 04 — Risque                                                                 #
# --------------------------------------------------------------------------- #
def fig_garch_vol(o: dict, world: str) -> None:
    p = o["p4"]
    g = p["scalars"]["garch_strategy"]
    vol = p["forecast"]["garch_vol"] * np.sqrt(bt.ANN)
    realised = p["returns"].rolling(21).std().reindex(vol.index) * np.sqrt(bt.ANN)
    fig, ax = _new(f"La volatilité se regroupe en grappes : persistance GARCH de {_fr(g['persistence'], 3)}",
                   f"Volatilité annualisée du portefeuille Quality. GARCH(1,1) : α = {_fr(g['alpha'], 3)}, β = {_fr(g['beta'], 3)}, "
                   f"volatilité de long terme {_fr(g['long_run_vol'] * 100)} %. Une persistance proche de 1 = un choc de risque met des mois à s'effacer.")
    ax.plot(realised.index, realised.values, color=NEUTRAL, lw=1.2, label="réalisée sur 21 jours")
    ax.plot(vol.index, vol.values, color=BLUE, label="prévue la veille par GARCH(1,1)")
    ax.set_ylabel("Volatilité annualisée")
    _pct(ax)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, world, "04_garch_vol")


def fig_var_exceptions(o: dict, world: str, alpha: float = 0.01) -> None:
    p = o["p4"]
    loss, t = p["loss"], p["scalars"]["var_backtests"]
    fig, axes = _new("La VaR filtrée suit le risque ; la VaR historique le découvre après coup",
                     f"Pertes quotidiennes du portefeuille Quality et VaR à {100 * (1 - alpha):.0f} % prévue la veille. Points : dépassements.",
                     nrows=2, size=(10, 6.6), sharex=True, sharey=True)
    for ax, key, lab, c in ((axes[0], "hist", "Simulation historique (500 jours)", ORANGE), (axes[1], "fhs", "Simulation historique filtrée (GARCH)", BLUE)):
        v = p["forecast"][key][f"var_{alpha}"]
        hits = loss > v
        k = t[f"{key}_{alpha}"]["kupiec"]
        ax.vlines(loss.index, 0, loss.clip(lower=0), color=NEUTRAL, lw=0.5)
        ax.plot(v.index, v.values, color=c, lw=1.6)
        ax.scatter(loss.index[hits], loss[hits], s=22, color=INK, zorder=3, edgecolor=SURFACE, lw=0.8)
        ax.set_title(f"{lab} — {k['exceptions']} dépassements pour {_fr(k['expected'])} attendus (Kupiec p = {_fr(k['p_value'], 2)}, "
                     f"indépendance p = {_fr(t[f'{key}_{alpha}']['christoffersen']['p_value'], 2)})", loc="left", fontsize=9.5, color=INK)
        ax.set_ylabel("Perte quotidienne")
        _pct(ax)
    axes[0].set_ylim(0, 1.08 * max(float(loss.max()), float(p["forecast"]["fhs"][f"var_{alpha}"].max())))
    _save(fig, world, "04_var_exceptions")


def fig_qq(o: dict, world: str) -> None:
    p = o["p4"]
    r = p["returns"].values
    series = (((r - r.mean()) / r.std(), ORANGE, "rendements bruts"), (p["fit"]["z"] / p["fit"]["z"].std(), BLUE, "résidus GARCH"))
    kurt = [float(np.mean(x ** 4)) for x, _, _ in series]
    fig, ax = _new(f"Kurtosis : {_fr(kurt[0])} pour les rendements bruts, {_fr(kurt[1])} pour les résidus GARCH (loi normale : 3)",
                   "Quantiles observés contre quantiles d'une loi normale ; sur la diagonale = gaussien. Ce qui reste hors de la "
                   "diagonale après filtrage justifie un quantile empirique des résidus plutôt qu'un quantile gaussien.", size=(8.6, 6.6))
    q = norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
    lim = 1.05 * max(abs(np.sort(s[0])[[0, -1]]).max() for s in series)
    ax.plot([-lim, lim], [-lim, lim], color=INK, lw=1)
    for x, c, lab in series:
        ax.scatter(q, np.sort(x), s=9, color=c, label=lab, alpha=0.8, lw=0)
    ax.set_xlim(q[0] * 1.08, q[-1] * 1.08)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Quantile théorique (loi normale)")
    ax.set_ylabel("Quantile observé (écarts-types)")
    ax.legend(loc="upper left", fontsize=9, markerscale=2.5)
    _save(fig, world, "04_qq")


def _p_backtests(ax, o: dict, cfg: dict) -> None:
    t = o["p4"]["scalars"]["var_backtests"]
    alphas = cfg["risk"]["alphas"]
    labels = {"normal": "Normale", "hist": "Historique", "fhs": "Filtrée (GARCH)"}
    colors = {"normal": NEUTRAL, "hist": ORANGE, "fhs": BLUE}
    x, width = np.arange(len(alphas)), 0.24
    for j, m in enumerate(labels):
        ratio = [t[f"{m}_{a}"]["kupiec"]["exceptions"] / t[f"{m}_{a}"]["kupiec"]["expected"] for a in alphas]
        bars = ax.bar(x + (j - 1) * (width + 0.02), ratio, width, color=colors[m], label=labels[m])
        for rect, a, v in zip(bars, alphas, ratio):
            ax.annotate(f"{_fr(v, 2)}\np = {_fr(t[f'{m}_{a}']['kupiec']['p_value'], 2)}", (rect.get_x() + width / 2, max(v, 1.0) if v > 0.8 else v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(1, color=INK, lw=1)
    ax.set_xticks(x, [f"VaR {100 * (1 - a):.0f} %" for a in alphas])
    ax.set_ylabel("Dépassements observés / attendus")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.22)
    ax.legend(loc="upper left", fontsize=8.5, ncols=3)


def fig_var_backtests(o: dict, world: str, cfg: dict) -> None:
    t, labels = o["p4"]["scalars"]["var_backtests"], {"normal": "normale", "hist": "historique", "fhs": "filtrée (GARCH)"}
    ok = [labels[m] for m in labels if all(t[f"{m}_{a}"][test]["p_value"] > 0.05
                                           for a in cfg["risk"]["alphas"] for test in ("kupiec", "christoffersen"))]
    fig, ax = _new("VaR qui passe les deux tests aux deux seuils : " + (", ".join(ok) if ok else "aucune"),
                   "Rapport dépassements observés / attendus (1 = calibré) et p-valeur du test de Kupiec (nombre de dépassements). "
                   "Le second test, l'indépendance de Christoffersen, est détaillé dans la figure des dépassements et le JSON.")
    _p_backtests(ax, o, cfg)
    _save(fig, world, "04_var_backtests")


# --------------------------------------------------------------------------- #
# Fiche de synthèse                                                           #
# --------------------------------------------------------------------------- #
def fig_tearsheet(o: dict, world: str, cfg: dict) -> None:
    s3, s4 = o["p3"]["scalars"], o["p4"]["scalars"]
    k = s4["var_backtests"]["fhs_0.01"]["kupiec"]
    tiles = [
        ("Alpha de Quality", f"{_fr(s3['active_return']['estimate'] * 100)} %/an",
         f"IC 95 % [{_fr(s3['active_return']['ci95'][0] * 100)} ; {_fr(s3['active_return']['ci95'][1] * 100)}]"),
        ("Plus petit alpha détectable", f"{_fr(s3['min_detectable_alpha'] * 100)} %/an", f"sur {_fr(s3['years'], 0)} ans, puissance 80 %"),
        ("Sharpe déflaté", f"{_fr(s3['deflated_sharpe'] * 100, 0)} %", f"probabilité d'un vrai alpha, {s3['n_trials']} essais"),
        ("VaR 99 % filtrée", f"{k['exceptions']} / {_fr(k['expected'], 0)}", f"dépassements observés / attendus (p = {_fr(k['p_value'], 2)})"),
    ]
    fig = plt.figure(figsize=(13, 10.6))
    name = "monde simulé (vérité connue)" if world == "simulated" else "données réelles"
    fig.suptitle("Portefeuille & risque — le profil Quality rapporte-t-il hors échantillon ?", x=0.03, y=0.985, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.03, 0.943, f"Fiche de synthèse, {name}. {o['world'].returns.shape[1]} titres, "
             f"{o['p3']['bench'].index[0].year}–{o['p3']['bench'].index[-1].year}, rebalancement mensuel, {cfg['backtest']['cost_bps']} pb de coût.", fontsize=10.5, color=MUTED)
    for i, (lab, val, note) in enumerate(tiles):
        x = 0.03 + i * 0.2425
        fig.patches.append(plt.Rectangle((x, 0.825), 0.228, 0.098, transform=fig.transFigure, fc="#f3f2ee", ec="none"))
        fig.text(x + 0.012, 0.912, lab, fontsize=9.5, color=MUTED, va="top")
        fig.text(x + 0.012, 0.892, val, fontsize=19, fontweight="bold", va="top")
        fig.text(x + 0.012, 0.834, note, fontsize=8.5, color=MUTED, va="bottom")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.77, bottom=0.06, hspace=0.36, wspace=0.24)
    panels = [
        ("01 — Dater le signal : " + ("le signal naïf invente un alpha" if "worlds" in o else "réaction du cours au dépôt"),
         (lambda ax: _p_worlds(ax, o)) if "worlds" in o else (lambda ax: _p_events(ax, o))),
        ("02 — Covariance : le nettoyage rapproche de l'oracle", lambda ax: _p_known(ax, o, cfg)),
        ("03 — Backtest : Quality face au cône du hasard", lambda ax: _p_cone(ax, o)),
        ("04 — Risque : dépassements de VaR observés / attendus", lambda ax: _p_backtests(ax, o, cfg)),
    ]
    for spec, (title, draw) in zip(gs, panels):
        ax = fig.add_subplot(spec)
        draw(ax)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    _save(fig, world, "00_tearsheet", layout=False)


def draw_all(o: dict, cfg: dict) -> None:
    world = o["world"].name
    for f in (fig_filing_gap, fig_quality_heatmap, fig_event_study, fig_naive_vs_pit, fig_spectrum, fig_correlations,
              fig_walk_forward, fig_equity, fig_chance_cone, fig_random_lab, fig_factors, fig_costs,
              fig_garch_vol, fig_var_exceptions, fig_qq):
        f(o, world)
    for f in (fig_known_truth, fig_var_backtests, fig_tearsheet):
        f(o, world, cfg)
    print(f"{len(list((FIGURES_DIR / world).glob('*.png')))} figures dans {FIGURES_DIR / world}")
