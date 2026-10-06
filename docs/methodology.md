# Méthodologie

Ce document fixe les conventions, les formules et les choix. Les chiffres sont dans `results/benchmark_<monde>.json`.

## 1. Ce qui est décidé avant le run

Tout ce qui pourrait être ajusté après coup pour embellir un résultat est dans `configs/research.yaml` :

| Choix | Valeur |
|---|---|
| Signal | profil Quality du Projet 1, règles v1.0, inchangées |
| Portefeuille principal | titres Quality équipondérés |
| Variantes déclarées | équipondéré, inverse volatilité, variance minimale (3 essais) |
| Référence | univers équipondéré |
| Rebalancement | dernier jour de bourse du mois |
| Coût | 10 pb par unité de turnover |
| Début | 2010-01-01 |

Le Sharpe déflaté est calculé avec ces 3 essais. Toute variante ajoutée plus tard doit l'être à la liste `VARIANTS` de `run.py`, donc au compte.

## 2. Partie 01 — dater le signal

**Ratios** (identiques aux vues SQL du Projet 1) :

- croissance du CA : `CA_t / CA_{t-1} - 1`, seulement si les exercices sont consécutifs et `CA_{t-1} > 0` ;
- ROIC : `EBIT x (1 - impôt / résultat avant impôt) / (dette + capitaux propres - trésorerie)`, seulement si le résultat avant impôt est positif ;
- dette nette / EBITDA ; marge EBITDA ; moyennes glissantes sur 3 exercices.

Une métrique manquante fait échouer la règle (`allow_null: false`).

**Disponibilité.** `avail_pit = filed + 1 jour de bourse`. Le signal à la date `d` est la dernière ligne dont `avail_pit <= d`, à condition qu'elle ait moins de 550 jours. Le signal naïf remplace `avail_pit` par `period_end_date`.

**Chaîne temporelle complète** : dépôt le jour J -> signal utilisable à J+1 -> pris en compte au rebalancement de fin de mois -> premier rendement encaissé le lendemain du rebalancement.

**Comparatifs tardifs.** Sur EDGAR, certains exercices anciens n'existent en XBRL que comme colonnes comparatives d'un 10-K ultérieur ; leur date de dépôt est alors celle de ce 10-K. Ils restent utilisables (la date est bien celle où la donnée structurée devient disponible), mais ne peuvent jamais remplacer un exercice plus récent déjà connu, et sont exclus de l'étude d'événement (`max_direct_filing_lag_days`).

**Limite connue.** Les résultats annuels sont annoncés (communiqué, 8-K) avant le dépôt du 10-K. La date de dépôt est donc prudente et non exacte : elle garantit l'absence de fuite, au prix d'un signal un peu tardif.

## 3. Partie 02 — covariance

- **Ledoit-Wolf** : `S* = (1 - d) S + d m I`, intensité `d` estimée analytiquement.
- **Marchenko-Pastur** : pour une matrice de corrélation de pur bruit, les valeurs propres sont dans `[s²(1 - √q)², s²(1 + √q)²]` avec `q = N/T`. Si la plus grande valeur propre dépasse le bord, c'est un mode de marché : la variance du bruit est alors `s² = 1 - λ_max / N`. Les valeurs propres sous le bord sont remplacées par leur moyenne (la trace est conservée), puis la matrice est remise à l'échelle des variances empiriques. Une marge `N^(-2/3)` tient compte du dépassement du bord à taille finie.
- **Juge** : le portefeuille à variance minimale `w = S⁻¹1 / (1'S⁻¹1)`, sans contrainte de signe. Sa volatilité vraie `√(w'Σw)` est comparée à celle de l'oracle quand Σ est connue ; sa volatilité réalisée hors échantillon sinon.

## 4. Partie 03 — inférence

- **Rendement actif** `a_t = r_t(Quality) - r_t(univers)` ; alpha = moyenne annualisée.
- **Bootstrap stationnaire** (Politis-Romano) : blocs de longueur géométrique de moyenne 21 jours, 2 000 tirages. Intervalle à 95 % sur l'alpha ; cône du hasard = trajectoires cumulées des rendements actifs recentrés.
- **PSR** : `Φ((SR - SR*) √(n - 1) / √(1 - γ₃ SR + (γ₄ - 1)/4 SR²))`, Sharpe par période, `γ₃` asymétrie, `γ₄` kurtosis.
- **Sharpe déflaté** : PSR avec `SR* = √V [(1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e))]`, `V` variance des Sharpe des `N` essais, `γ` constante d'Euler.
- **Plus petit alpha détectable** : `(z₀.₉₅ + z₀.₈₀) x tracking error / √années`.
- **Facteurs** : MCO du rendement actif quotidien, écarts-types Newey-West, `⌊4 (n/100)^(2/9)⌋` retards.
- **Stratégies aléatoires** : chaque mois, autant de titres que Quality en détient, tirés uniformément ; rendements mensuels bruts.

## 5. Partie 04 — risque

- **GARCH(1,1)** : `h_{t+1} = ω + α ε_t² + β h_t`, quasi-maximum de vraisemblance gaussien, paramétré en (α, β, variance de long terme).
- **VaR filtrée** : `VaR = -(μ + √h_{t+1} x quantile_α(z))`, où `z` sont les résidus standardisés passés. Paramètres ré-estimés tous les 63 jours, filtre mis à jour chaque jour.
- **Kupiec** : rapport de vraisemblance entre le taux de dépassement observé et `α`, loi χ²(1).
- **Christoffersen** : rapport de vraisemblance entre une chaîne de Markov à deux états et des dépassements indépendants, loi χ²(1).
- **Expected Shortfall** : contrôle simple, perte moyenne réalisée les jours de dépassement rapportée à l'ES prévu.

## 6. Monde simulé

Rendement du titre i : `r = rf + β_i x marché + facteur sectoriel + bruit propre + saut de dépôt`.

- Marché : GARCH(1,1), α = 0,06, β = 0,92, innovations Student à 8 degrés de liberté, volatilité 16 %.
- Saut de dépôt : 4 % par écart-type de surprise de marge, le premier jour de bourse à partir du dépôt.
- Fondamentaux : marge en AR(1) autour d'un niveau propre à l'entreprise, croissance, rotation du capital et levier tirés par entreprise.
- `quality_premium` (nul par défaut) ajoute un alpha vrai aux titres Quality à partir du moment où le signal est public ; un test vérifie qu'une prime de 6 %/an est retrouvée.
