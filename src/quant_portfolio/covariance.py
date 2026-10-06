"""Partie 02 — estimer une matrice de covariance quand N n'est pas petit devant T.

Avec N actifs et T observations, la covariance empirique a N(N+1)/2 paramètres :
dès que q = N/T n'est plus négligeable, ses petites valeurs propres sont du bruit,
et un optimiseur qui l'inverse parie précisément sur ce bruit.

    sample        covariance empirique (référence naïve)
    ledoit_wolf   rétrécissement linéaire vers un multiple de l'identité
    mp_clip       valeurs propres sous le bord de Marchenko-Pastur remplacées
                  par leur moyenne (Laloux et al., Bouchaud & Potters)

Le juge est le portefeuille à variance minimale : sa variance RÉALISÉE hors
échantillon dépend directement de la qualité de l'estimateur.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import ledoit_wolf as _lw


def sample(X: np.ndarray) -> np.ndarray:
    return np.cov(X, rowvar=False)


def ledoit_wolf(X: np.ndarray) -> np.ndarray:
    return _lw(X)[0]


def mp_edge(q: float, sigma2: float = 1.0) -> float:
    """Plus grande valeur propre d'une matrice de corrélation de pur bruit."""
    return sigma2 * (1 + np.sqrt(q)) ** 2


def mp_density(lam: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    lo, hi = sigma2 * (1 - np.sqrt(q)) ** 2, mp_edge(q, sigma2)
    inside = np.clip((hi - lam) * (lam - lo), 0, None)
    return np.sqrt(inside) / (2 * np.pi * q * sigma2 * lam)


def noise_mask(lam: np.ndarray, q: float) -> tuple[np.ndarray, float]:
    """Quelles valeurs propres sont du bruit, et quelle est la variance de ce bruit ?
    Si la plus grande reste sous le bord (marge N^(-2/3) : à N fini elle le dépasse un
    peu), tout est bruit. Sinon c'est un facteur de marché, que l'on retire avant de
    mesurer la variance du bruit : sigma² = 1 - lambda_max / N (Laloux et al., 1999)."""
    n = len(lam)
    margin = 1 + n ** (-2 / 3)
    if lam[-1] <= mp_edge(q) * margin:
        return np.ones(n, bool), 1.0
    sigma2 = 1 - lam[-1] / n
    return lam <= mp_edge(q, sigma2) * margin, float(sigma2)


def mp_clip(X: np.ndarray) -> np.ndarray:
    T, N = X.shape
    std = X.std(axis=0, ddof=1)
    lam, vec = np.linalg.eigh(np.corrcoef(X, rowvar=False))
    noise, _ = noise_mask(lam, N / T)
    lam = lam.copy()
    lam[noise] = lam[noise].mean()              # conserve la trace
    corr = (vec * lam) @ vec.T
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d) * np.outer(std, std)


ESTIMATORS = {"Empirique": sample, "Ledoit-Wolf": ledoit_wolf, "Marchenko-Pastur": mp_clip}


def gmv(cov: np.ndarray) -> np.ndarray:
    """Poids du portefeuille à variance minimale, somme = 1, ventes à découvert permises."""
    w = np.linalg.solve(cov, np.ones(len(cov)))
    return w / w.sum()


def known_truth_experiment(sigma: np.ndarray, q_grid: list[float], n_rep: int, rng: np.random.Generator) -> dict:
    """Tire T = N/q observations gaussiennes d'une covariance CONNUE et mesure, pour
    chaque estimateur, la volatilité vraie du portefeuille à variance minimale qu'il
    produit, rapportée à celle de l'oracle (1.0 = parfait)."""
    N = len(sigma)
    chol = np.linalg.cholesky(sigma)
    oracle = np.sqrt(gmv(sigma) @ sigma @ gmv(sigma))
    out = {name: [] for name in ESTIMATORS}
    for q in q_grid:
        T = int(round(N / q))
        ratios = {name: [] for name in ESTIMATORS}
        for _ in range(n_rep):
            X = rng.standard_normal((T, N)) @ chol.T
            for name, est in ESTIMATORS.items():
                w = gmv(est(X))
                ratios[name].append(np.sqrt(w @ sigma @ w) / oracle)
        for name in ESTIMATORS:
            out[name].append(float(np.mean(ratios[name])))
    return out
