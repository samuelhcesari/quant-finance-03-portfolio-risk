.PHONY: install simulated fetch real test

install:
	pip install -r requirements.txt
	pip install -e .

# --- Monde simulé : aucune donnée externe, aucune clef, ~2 minutes ------------
simulated:
	python -m quant_portfolio.run --world simulated

# --- Données réelles : nécessite le Projet 1 cloné à côté et déjà exécuté -----
fetch:
	python -m quant_portfolio.world

real:
	python -m quant_portfolio.run --world real

test:
	pytest

