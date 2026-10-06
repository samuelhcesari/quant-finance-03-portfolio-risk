"""Partie 03 — moteur de backtest et inférence statistique.

Le moteur est volontairement minimal : des poids décidés à la clôture d'une date
de rebalancement, appliqués aux rendements des jours SUIVANTS, qui dérivent
jusqu'au rebalancement d'après. Tout le reste du module répond à une seule
question : la performance mesurée se distingue-t-elle du hasard ?

    stationary_bootstrap   intervalles de confiance sans hypothèse i.i.d.
    psr / deflated_sharpe  Sharpe corrigé de l'asymétrie, des queues et du
                           nombre d'essais (Bailey & López de Prado, 2014)
    newey_west_ols         expositions factorielles, erreurs robustes HAC
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

ANN = 252
EULER = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Moteur                                                                      #
# --------------------------------------------------------------------------- #
def month_ends(index: pd.DatetimeIndex, start: str) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)[start:]
    return pd.DatetimeIndex(s.groupby([s.index.year, s.index.month]).last().values[:-1])


def equal_weight(mask: pd.DataFrame) -> pd.DataFrame:
    m = mask.fillna(False).astype(float)
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def run(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Rendements quotidiens nets et turnover par rebalancement.
    La part non investie (somme des poids < 1) reste en liquidités à rendement nul."""
    W = weights.reindex(columns=returns.columns).fillna(0.0)
    out, turnover, held = [], {}, np.zeros(W.shape[1])
    for k, d in enumerate(W.index):
        end = W.index[k + 1] if k + 1 < len(W) else returns.index[-1]
        seg = returns.loc[(returns.index > d) & (returns.index <= end)].fillna(0.0)
        if seg.empty:
            continue
        w = W.values[k]
        value = np.cumprod(1 + seg.values, axis=0) * w                 # dérive sans rebalancement
        total = value.sum(axis=1) + (1 - w.sum())
        r = total / np.concatenate([[1.0], total[:-1]]) - 1
        turnover[d] = np.abs(w - held).sum()
        r[0] -= turnover[d] * cost_bps / 1e4
        held = value[-1] / total[-1]
        out.append(pd.Series(r, index=seg.index))
    return pd.concat(out), pd.Series(turnover)


# --------------------------------------------------------------------------- #
# Mesures                                                                     #
# --------------------------------------------------------------------------- #
def sharpe(x: np.ndarray, periods: int = ANN) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(periods))


def drawdown(r: pd.Series) -> pd.Series:
    v = (1 + r).cumprod()
    return v / v.cummax() - 1


def summary(r: pd.Series, rf: pd.Series, bench: pd.Series | None = None) -> dict:
    ex = r - rf.reindex(r.index).fillna(0)
    out = dict(
        ann_return=float((1 + r).prod() ** (ANN / len(r)) - 1),
        ann_vol=float(r.std() * np.sqrt(ANN)),
        sharpe=sharpe(ex.values),
        max_drawdown=float(drawdown(r).min()),
    )
    if bench is not None:
        a = (r - bench).dropna()
        out |= dict(active_return=float(a.mean() * ANN), tracking_error=float(a.std() * np.sqrt(ANN)),
                    information_ratio=sharpe(a.values))
    return out


# --------------------------------------------------------------------------- #
# Inférence                                                                   #
# --------------------------------------------------------------------------- #
def stationary_bootstrap(n: int, block: float, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Indices du bootstrap stationnaire (Politis & Romano, 1994) : blocs de
    longueur géométrique de moyenne `block`, ce qui préserve l'autocorrélation
    et les grappes de volatilité que détruirait un tirage i.i.d."""
    idx = np.empty((n_boot, n), dtype=np.int32)
    idx[:, 0] = rng.integers(0, n, n_boot)
    restart = rng.random((n_boot, n)) < 1.0 / block
    jump = rng.integers(0, n, (n_boot, n), dtype=np.int32)
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], jump[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def psr(x: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio : P(vrai Sharpe > sr_benchmark), Sharpe par période.
    Tient compte de la longueur de l'échantillon, de l'asymétrie et du kurtosis."""
    x = np.asarray(x)
    n, sr = len(x), np.mean(x) / np.std(x, ddof=1)
    z = (x - x.mean()) / x.std()
    skew, kurt = np.mean(z ** 3), np.mean(z ** 4)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)))


def expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """Sharpe (par période) que le MEILLEUR de N essais sans aucun talent atteint
    en moyenne — le seuil que le Sharpe retenu doit dépasser."""
    n = len(trial_sharpes)
    if n < 2:
        return 0.0
    return float(np.std(trial_sharpes, ddof=1) * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(x: np.ndarray, trial_sharpes: np.ndarray) -> float:
    return psr(x, expected_max_sharpe(trial_sharpes))


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """MCO avec constante ; écarts-types HAC de Newey-West (noyau de Bartlett)."""
    y, X = np.asarray(y, float), np.column_stack([np.ones(len(y)), X])
    n, k = X.shape
    lags = int(4 * (n / 100) ** (2 / 9)) if lags is None else lags
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    u = (y - X @ coef)[:, None] * X
    meat = u.T @ u
    for l in range(1, lags + 1):
        g = u[l:].T @ u[:-l]
        meat += (1 - l / (lags + 1)) * (g + g.T)
    se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv) * n / (n - k))
    return dict(coef=coef, se=se, t=coef / se, lags=lags, r2=float(1 - np.var(y - X @ coef) / np.var(y)))


def min_detectable_alpha(tracking_error: float, years: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Plus petit alpha annuel qu'un test unilatéral à 5 % détecte 4 fois sur 5."""
    return float((norm.ppf(1 - alpha) + norm.ppf(power)) * tracking_error / np.sqrt(years))
