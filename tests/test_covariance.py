"""Partie 02 — Marchenko-Pastur sur du bruit pur, et les estimateurs face à une vérité connue."""

import numpy as np

from quant_portfolio import covariance as cv

RNG = np.random.default_rng(1)


def test_mp_density_integrates_to_one():
    for q in (0.1, 0.5, 0.9):
        x = np.linspace(1e-6, cv.mp_edge(q), 200_001)
        y = cv.mp_density(x, q)
        assert abs(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) - 1) < 2e-3


def test_pure_noise_stays_below_the_edge_and_is_cleaned_to_identity():
    X = RNG.standard_normal((2000, 100))
    lam = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
    assert lam.max() < cv.mp_edge(100 / 2000) * 1.05
    assert cv.noise_mask(lam, 100 / 2000)[0].all()                                # rien n'est pris pour du signal
    corr = cv.mp_clip(X) / np.outer(X.std(0, ddof=1), X.std(0, ddof=1))
    assert np.abs(corr - np.eye(100)).max() < 1e-9


def test_mp_clip_keeps_variances_and_a_real_factor():
    market = RNG.standard_normal((500, 1))
    X = market * RNG.uniform(0.5, 1.5, 40) + RNG.standard_normal((500, 40))
    clean = cv.mp_clip(X)
    assert np.allclose(np.diag(clean), X.var(0, ddof=1))
    assert np.allclose(clean, clean.T) and np.linalg.eigvalsh(clean).min() > 0
    off = ~np.eye(40, dtype=bool)
    assert np.corrcoef(clean[off], np.cov(X, rowvar=False)[off])[0, 1] > 0.9      # le marché survit au nettoyage
    Y = market + RNG.standard_normal((500, 40))                                   # bêtas égaux : bruit homogène
    lam = np.linalg.eigvalsh(np.corrcoef(Y, rowvar=False))
    assert (~cv.noise_mask(lam, 40 / 500)[0]).sum() == 1                          # exactement un facteur détecté


def test_gmv_closed_form_two_assets():
    cov = np.array([[0.04, 0.006], [0.006, 0.09]])
    w = cv.gmv(cov)
    assert np.isclose(w.sum(), 1) and np.isclose(w[0], (0.09 - 0.006) / (0.04 + 0.09 - 2 * 0.006))


def test_cleaned_estimators_beat_the_sample_covariance_when_data_is_scarce():
    beta = RNG.uniform(0.7, 1.3, 40)
    sigma = np.outer(beta, beta) * 1e-4 + np.diag(RNG.uniform(2e-4, 5e-4, 40))
    res = cv.known_truth_experiment(sigma, [0.1, 0.8], 30, RNG)
    assert all(v[0] >= 1 and v[1] >= 1 for v in res.values())                     # personne ne bat l'oracle
    assert res["Ledoit-Wolf"][1] < res["Empirique"][1] and res["Marchenko-Pastur"][1] < res["Empirique"][1]
    assert res["Empirique"][1] > 1.5 > res["Empirique"][0]
