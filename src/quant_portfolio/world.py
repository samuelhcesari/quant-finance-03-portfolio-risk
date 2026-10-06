"""Un « monde » = tout ce dont les quatre parties ont besoin, simulé ou réel.

Le reste du code ne sait pas lequel des deux il manipule : c'est ce qui permet
de valider chaque méthode sur un monde où la vérité est connue (`simulate.py`)
avant de l'appliquer aux données du Projet 1.

Monde réel :
    make fetch                      # prix ajustés (Yahoo) ; rafraîchit aussi les facteurs
    python -m quant_portfolio.run --world real
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from quant_portfolio.config import ROOT


@dataclass
class World:
    name: str
    fund: pd.DataFrame       # 1 ligne = 1 entreprise x 1 exercice (colonnes du Projet 1)
    prices: pd.DataFrame     # clôtures ajustées quotidiennes, colonnes = tickers
    sectors: pd.Series       # ticker -> secteur
    factors: pd.DataFrame    # rendements quotidiens : MKT (excédentaire), RF, + autres
    truth: dict = field(default_factory=dict)   # vide pour le monde réel

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change(fill_method=None).iloc[1:]


def load_real(cfg: dict) -> World:
    d = cfg["data"]
    fund = pd.read_csv(ROOT / d["fundamentals"], parse_dates=["period_end_date", "filed"])
    prices = pd.read_csv(ROOT / d["prices"], index_col=0, parse_dates=True).sort_index()
    factors = pd.read_csv(ROOT / d["factors"], index_col=0, parse_dates=True)
    with open(ROOT / d["universe"], encoding="utf-8") as f:
        universe = yaml.safe_load(f)["sectors"]
    sectors = pd.Series({t: s for s, v in universe.items() for t in v["tickers"]})
    tickers = [t for t in sectors.index if t in prices.columns]
    return World("real", fund[fund.ticker.isin(tickers)], prices[tickers], sectors[tickers], factors)


def fetch(cfg: dict) -> None:  # pragma: no cover — accès réseau
    """Télécharge ce que le Projet 1 ne fournit pas : un historique de prix long et
    AJUSTÉ des dividendes (le Projet 1 garde 5 ans de clôtures brutes), non
    redistribuable donc jamais committé, et rafraîchit les facteurs Fama-French."""
    import io
    import urllib.request
    import zipfile

    import yfinance as yf

    d = cfg["data"]
    with open(ROOT / d["universe"], encoding="utf-8") as f:
        tickers = sorted(t for v in yaml.safe_load(f)["sectors"].values() for t in v["tickers"])
    px = yf.download(tickers, start="2005-01-01", auto_adjust=True, progress=False, threads=False)["Close"]
    for t in [t for t in tickers if t not in px or px[t].notna().sum() == 0]:      # nouvel essai, un par un
        px[t] = yf.Ticker(t).history(start="2005-01-01", auto_adjust=True)["Close"].tz_localize(None)
    px.index = px.index.tz_localize(None)
    missing = [t for t in tickers if px[t].notna().sum() == 0]
    if missing:
        raise SystemExit(f"prix introuvables pour {missing}")
    Path(ROOT / d["prices"]).parent.mkdir(exist_ok=True)
    px.to_csv(ROOT / d["prices"])
    print(f"prix : {px.shape[1]} tickers, {px.index[0].date()} -> {px.index[-1].date()}")

    base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    frames = []
    for name in ("F-F_Research_Data_5_Factors_2x3_daily", "F-F_Momentum_Factor_daily"):
        raw = urllib.request.urlopen(f"{base}{name}_CSV.zip").read()
        text = zipfile.ZipFile(io.BytesIO(raw)).read(zipfile.ZipFile(io.BytesIO(raw)).namelist()[0])
        lines = [l for l in text.decode("latin-1").splitlines() if l[:8].strip().isdigit() or "Mkt" in l or "Mom" in l]
        df = pd.read_csv(io.StringIO("\n".join(lines)), index_col=0)
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
        frames.append(df.rename(columns=lambda c: c.strip()) / 100.0)
    ff = pd.concat(frames, axis=1, sort=True).dropna().rename(columns={"Mkt-RF": "MKT", "Mom": "MOM"}).loc["2005":]
    ff.to_csv(ROOT / d["factors"])
    print(f"facteurs : {list(ff.columns)}, {ff.index[0].date()} -> {ff.index[-1].date()}")


if __name__ == "__main__":  # pragma: no cover
    from quant_portfolio import config

    fetch(config.load())
