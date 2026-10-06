"""Partie 04 — mesurer le risque, puis vérifier que la mesure tient.

Trois VaR à un jour, de la plus naïve à la plus soignée :

    normal    quantile gaussien, volatilité constante de la fenêtre
    hist      simulation historique sur une fenêtre glissante
    fhs       simulation historique FILTRÉE : GARCH(1,1) pour la volatilité du
              lendemain, quantile empirique des résidus standardisés
              (Barone-Adesi, Giannopoulos & Vosper, 1999)

Chaque prévision n'utilise que le passé. Les tests de Kupiec (bon NOMBRE de
dépassements) et de Christoffersen (dépassements non groupés) jugent le résultat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm


def garch_filter(x: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Variances conditionnelles h_0..h_n (h_n = prévision du lendemain)."""
    h = np.empty(len(x) + 1)
    h[0] = x.var()
    for t in range(len(x)):
        h[t + 1] = omega + alpha * x[t] ** 2 + beta * h[t]
    return h


def garch_fit(r: np.ndarray) -> dict:
    """GARCH(1,1) par quasi-maximum de vraisemblance gaussien.
    Paramétrisation (alpha, beta, variance de long terme) : contraintes simples."""
    x = np.asarray(r, float) - np.mean(r)
    v = x.var()

    def nll(p: np.ndarray) -> float:
        a, b, scale = p
        if a + b >= 0.9999:
            return 1e12
        h = garch_filter(x, scale * v * (1 - a - b), a, b)[:-1]
        return 0.5 * np.sum(np.log(h) + x ** 2 / h)

    res = minimize(nll, x0=[0.08, 0.88, 1.0], method="L-BFGS-B",
                   bounds=[(1e-4, 0.5), (1e-4, 0.9998), (0.2, 5.0)])
    a, b, scale = res.x
    omega = scale * v * (1 - a - b)
    h = garch_filter(x, omega, a, b)
    return dict(mu=float(np.mean(r)), omega=float(omega), alpha=float(a), beta=float(b),
                h=h[:-1], h_next=float(h[-1]), z=x / np.sqrt(h[:-1]))


def rolling_var(r: pd.Series, alphas: list[float], min_window: int, refit_every: int, hs_window: int) -> dict:
    """VaR et ES à 1 jour, hors échantillon. Renvoie, par méthode, un DataFrame
    de colonnes `var_<alpha>` / `es_<alpha>` (pertes, donc positives), plus la
    volatilité conditionnelle GARCH prévue."""
    x = r.values
    idx = r.index[min_window:]
    cols = [f"{k}_{a}" for a in alphas for k in ("var", "es")]
    rows: dict[str, list] = {"normal": [], "hist": [], "fhs": []}
    vol = pd.Series(index=idx, dtype=float)
    for t in range(min_window, len(x)):
        past = x[:t]
        if (t - min_window) % refit_every == 0:          # ré-estimation périodique des paramètres
            fit = garch_fit(past)
            mu, h_next, z = fit["mu"], fit["h_next"], list(fit["z"])
        else:                                            # entre deux : le filtre avance d'un jour
            e = x[t - 1] - mu
            z.append(e / np.sqrt(h_next))
            h_next = fit["omega"] + fit["alpha"] * e ** 2 + fit["beta"] * h_next
        sig, z_now = np.sqrt(h_next), np.asarray(z)
        win = past[-hs_window:]
        m_, s_ = win.mean(), win.std(ddof=1)
        vol.iloc[t - min_window] = sig
        for a in alphas:
            q, qh = np.quantile(z_now, a), np.quantile(win, a)
            rows["fhs"].append([-(mu + sig * q), -(mu + sig * z_now[z_now <= q].mean())])
            rows["hist"].append([-qh, -win[win <= qh].mean()])
            rows["normal"].append([-(m_ + s_ * norm.ppf(a)), -(m_ - s_ * norm.pdf(norm.ppf(a)) / a)])
    out = {m: pd.DataFrame(np.reshape(v, (len(idx), -1)), index=idx, columns=cols) for m, v in rows.items()}
    out["garch_vol"] = vol
    out["z"] = z_now
    return out


def _xlogy(k: float, p: float) -> float:
    return 0.0 if k == 0 else k * np.log(p)


def kupiec(hits: np.ndarray, alpha: float) -> dict:
    """Test de couverture inconditionnelle : le taux de dépassement vaut-il alpha ?"""
    n, k = len(hits), int(hits.sum())
    pi = k / n
    lr = -2 * (_xlogy(k, alpha) + _xlogy(n - k, 1 - alpha) - _xlogy(k, pi) - _xlogy(n - k, 1 - pi))
    return dict(n=n, exceptions=k, expected=alpha * n, rate=pi, lr=float(lr), p_value=float(chi2.sf(lr, 1)))


def christoffersen(hits: np.ndarray) -> dict:
    """Test d'indépendance : un dépassement rend-il le suivant plus probable ?"""
    h = hits.astype(int)
    n00, n01 = np.sum((h[:-1] == 0) & (h[1:] == 0)), np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10, n11 = np.sum((h[:-1] == 1) & (h[1:] == 0)), np.sum((h[:-1] == 1) & (h[1:] == 1))
    p01, p11 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    l0 = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    l1 = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    lr = -2 * (l0 - l1)
    return dict(p01=float(p01), p11=float(p11), lr=float(lr), p_value=float(chi2.sf(lr, 1)))
