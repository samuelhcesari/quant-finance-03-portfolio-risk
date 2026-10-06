"""Chemins et chargement de la configuration. Pas de logique métier ici."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load(path: Path | str = ROOT / "configs" / "research.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
