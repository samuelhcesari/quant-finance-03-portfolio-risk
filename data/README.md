# Données

| Fichier | Source | Dans le dépôt |
|---|---|---|
| `financials_annual.csv` | Sortie du [Projet 1](https://github.com/samuelhcesari/quant-finance-01-pipeline) (`data/processed/`), construite depuis SEC EDGAR / XBRL : 714 exercices, 43 entreprises, avec la date de dépôt de chaque 10-K | oui |
| `ff_factors_daily.csv` | [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) : 5 facteurs Fama-French + momentum, quotidiens, en rendements décimaux, depuis 2005 | oui |
| `prices.csv` | Yahoo Finance, clôtures ajustées quotidiennes depuis 2005 | non — `make fetch` |

Les prix ne sont pas committés : les conditions d'utilisation de Yahoo Finance n'en autorisent pas la redistribution. `make fetch` les télécharge en moins d'une minute et rafraîchit au passage les facteurs.

Pour mettre à jour les états financiers, relancer le pipeline du Projet 1 et recopier son `data/processed/financials_annual.csv` ici.
