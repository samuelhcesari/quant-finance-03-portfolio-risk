# Portfolio Progress — Projet 3/3 : Portfolio & Risk Analytics (Python)

Suivi d'avancement. Une case n'est cochée que quand le résultat a vraiment tourné, pas juste écrit dans le code.

Ce projet clôt la série : le Projet 1/3 (`quant-finance-01-pipeline`, SQL) produit les signaux, le Projet 2/3 (`quant-finance-02-derivatives`, NumPy) valorise, celui-ci décide, mesure et doute.

## Cadrage

- [x] Une seule question de recherche, un seul signal (Quality), un portefeuille principal déclaré avant le run.
- [x] Paramètres figés dans `configs/research.yaml` ; règles Quality recopiées de `configs/screening/quality.yaml` v1.0 du Projet 1.
- [x] Principe : chaque méthode validée sur un monde simulé à vérité connue avant les données réelles.

## Monde simulé (`simulate.py`)

- [x] 45 titres, 3 secteurs, 4 696 jours, 945 exercices, mêmes colonnes que `financials_annual.csv` du Projet 1.
- [x] Vérités imposées : alpha Quality nul, saut de cours au dépôt, covariance connue, marché GARCH(1,1) Student.
- [x] Reproductible : même graine -> mêmes prix ; deux runs complets -> JSON identique à l'octet.

## 01 — Signal daté (`signal.py`)

- [x] Ratios du Projet 1 reproduits en pandas (croissance sur exercices consécutifs, ROIC, levier, moyennes 3 ans) ; vérifiés à la main dans les tests.
- [x] Jointure as-of sur `filed` + 1 jour de bourse ; péremption du signal après 550 jours.
- [x] Test anti-fuite : signal à `d` inchangé quand on supprime tout dépôt postérieur à `d`. Le signal naïf échoue au même test.
- [x] 200 mondes : alpha daté −0,05 %/an (± 0,13), alpha naïf +0,52 %/an, taux de rejet à 5 % du signal daté = 4,0 %.
- [x] Étude d'événement : +2,9 % / −2,2 % le jour du dépôt pour les entrants / sortants.

## 02 — Covariance (`covariance.py`)

- [x] Empirique, Ledoit-Wolf (scikit-learn), nettoyage Marchenko-Pastur maison (variance du bruit mesurée hors mode de marché, marge de taille finie).
- [x] Bruit pur : toutes les valeurs propres sous le bord, matrice nettoyée = identité.
- [x] Covariance connue : à q = 0,9, volatilité du portefeuille à variance minimale 3,09x l'oracle (empirique), 1,20x (Ledoit-Wolf), 1,17x (Marchenko-Pastur).
- [x] Walk-forward à 63 / 126 / 252 jours : volatilité réalisée et turnover.

## 03 — Backtest (`backtest.py`)

- [x] Moteur : poids à la clôture, rendements à partir du lendemain, dérive entre rebalancements, coûts au turnover. Testé contre un calcul exact.
- [x] Bootstrap stationnaire, PSR, Sharpe déflaté, MCO Newey-West, plus petit alpha détectable.
- [x] Monde illustré : alpha −1,4 %/an, IC 95 % [−5,9 ; +3,0], Sharpe déflaté 26 %, alpha détectable 5,6 %/an.
- [x] 1 000 stratégies aléatoires : 34 « significatives » au PSR 95 % ; meilleure : PSR 99,8 % -> Sharpe déflaté 38 %.

## 04 — Risque (`risk.py`)

- [x] GARCH(1,1) par quasi-maximum de vraisemblance (SciPy) : α 0,079 ± 0,010 et β 0,895 ± 0,015 pour 0,08 / 0,90 vrais.
- [x] VaR et ES à 1 jour, hors échantillon : normale, historique, historique filtrée. Test : réécrire le futur ne change aucune prévision passée.
- [x] Kupiec et Christoffersen sur 3 403 jours : seule la VaR filtrée passe les deux tests aux deux seuils.

## Figures et tests

- [x] 18 figures par monde (`results/figures/simulated/` et `results/figures/real/`), dont une fiche de synthèse. Titres calculés à partir des résultats, pas écrits à la main.
- [x] 28 tests, tous passent (`pytest`).
- [x] Bug trouvé par les tests : la prime Quality optionnelle de la simulation était versée aux mauvais titres (colonnes dans un ordre différent). Corrigé ; le test de récupération de la prime passe.

## Données réelles

- [x] `make fetch` : 43 tickers de prix ajustés Yahoo (2005-01-03 -> 2026-10-06), facteurs Fama-French 5 + momentum (-> 2026-08-31). Un ticker (BLDR) échouait en téléchargement groupé : nouvel essai titre par titre ajouté.
- [x] Cohérence avec le Projet 1 : 107 exercices Quality sur 714, ROIC d'Apple 42,1 % (2020) et 87,4 % (2025) — identiques à ses chiffres publiés.
- [x] Cas réel absent de la simulation : 13 % des exercices (surtout 2007-2009) ne sont connus en XBRL que par un 10-K ultérieur. Règle ajoutée dans `asof_panel` (un exercice ancien ne remplace jamais un plus récent déjà connu) + test dédié ; ces lignes sont exclues de l'étude d'événement.
- [x] 01 : écart naïf − daté +0,38 %/an (t = 0,43) ; jour du dépôt +0,3 % / −0,5 %.
- [x] 02 : 32 valeurs propres sur 36 dans le bruit ; à 63 jours, volatilité réalisée 23,6 % (empirique) contre 15,3 % (Ledoit-Wolf et Marchenko-Pastur).
- [x] 03 : alpha −1,6 %/an, IC 95 % [−7,8 ; +4,1], Sharpe déflaté 12 %, alpha détectable 7,8 %/an ; exposition RMW +0,16 (t = 4,9).
- [x] 04 : persistance GARCH 0,960 ; seule la VaR filtrée passe Kupiec et Christoffersen à 95 % et 99 %.

## Choix de dépendances

- Pas de `statsmodels` ni de `arch` : la régression Newey-West (une quinzaine de lignes) et le GARCH(1,1) (une quarantaine) sont écrits avec NumPy / SciPy et vérifiés contre des paramètres connus, dans la continuité du Projet 2.
