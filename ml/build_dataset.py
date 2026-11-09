"""Construit le jeu de données d'entraînement : une ligne = (zone, jour).

Pour chaque (zone, jour D) :
  - FEATURES : météo du jour D et des jours précédents (jamais du futur !)
  - ÉTIQUETTES : label_d1 = 1 si un feu de végétation est détecté dans la zone le jour D+1
                 (label_d2 / label_d3 : jours D+2 et D+3, pour l'extension 48 h / 72 h)

Étapes :
  1. Lire les feux FIRMS, garder type 0 (végétation) et confiance n/h, les placer dans une zone.
  2. Exclure les zones à "source de chaleur fixe" (feu presque toute l'année, sans saisonnalité).
  3. Calculer les features météo (jours secs consécutifs, cumuls de pluie, moyennes, retards).
  4. Ajouter les étiquettes, ne garder que les mois de saison (juin-octobre par défaut).
  5. Découper CHRONOLOGIQUEMENT en train / validation / test.

Prérequis : data/grid.csv, data/raw/firms/VIIRS_SNPP_SP/*.csv, data/weather_daily.csv
  (python ingestion/fetch_weather.py --flatten)

Usage :  python ml/build_dataset.py
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DRY_MM = 1.0   # un jour est "sec" si la pluie est < 1 mm
# Variables météo brutes (colonnes de weather_daily.csv) -> noms courts
RAW = {
    "temperature_2m_max": "tmax",
    "relative_humidity_2m_min": "rh_min",
    "wind_speed_10m_max": "wind_max",
    "precipitation_sum": "precip",
    "soil_moisture_0_to_7cm_mean": "soil",
    "et0_fao_evapotranspiration": "et0",
}
FEATURES = ["tmax", "rh_min", "wind_max", "precip", "soil", "et0",
            "dry_days", "rain_7d", "rain_30d", "tmax_mean3", "rh_min_mean3",
            "tmax_lag1", "rh_min_lag1", "month", "doy", "lat", "lon",
            "tmax_anom", "rh_min_anom", "soil_anom", "zone_rate"]
# Historique de feu (optionnel dans train.py avec --fire-history) : fait partie du jeu de données
# mais n'est PAS dans FEATURES, pour pouvoir comparer "météo seule" et "météo + feux récents".
FIRE_FEATURES = ["fire_1d", "fire_3d", "fire_7d", "nbr_fire_1d", "nbr_fire_7d"]


# ---------------------------------------------------------------- 1. Feux
def load_fires(firms_dir, west, south, step, n_lat, n_lon):
    """Renvoie un tableau (zone_id, date) : jours où la zone a au moins un feu retenu."""
    frames = []
    for f in sorted(Path(firms_dir).glob("*.csv")):
        frames.append(pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "confidence", "type"],
                                  dtype={"confidence": str}, keep_default_na=False))
    if not frames:
        raise SystemExit(f"Aucun fichier FIRMS dans {firms_dir}")
    df = pd.concat(frames, ignore_index=True)
    n_all = len(df)
    df = df[(df["type"] == 0) & (df["confidence"] != "l")].copy()   # filtres décidés après exploration
    df["i"] = np.floor((df["latitude"] - south) / step).astype(int)
    df["j"] = np.floor((df["longitude"] - west) / step).astype(int)
    df = df[(df["i"] >= 0) & (df["i"] < n_lat) & (df["j"] >= 0) & (df["j"] < n_lon)]
    df["zone_id"] = "Z" + df["i"].astype(str).str.zfill(3) + "_" + df["j"].astype(str).str.zfill(3)
    df["date"] = pd.to_datetime(df["acq_date"])
    pos = df[["zone_id", "date"]].drop_duplicates().reset_index(drop=True)
    print(f"Feux : {n_all} détections lues -> {len(df)} retenues -> {len(pos)} zone-jours positifs")
    return pos


# ------------------------------------------------- 2. Zones à source fixe
def find_static_zones(pos, min_days=100, ratio=0.3):
    """Zone suspecte = au moins `min_days` jours de feu ET taux de feu en hiver > ratio x taux en été.
    Un vrai feu de forêt est saisonnier ; une source de chaleur fixe brûle toute l'année."""
    span = pd.date_range(pos["date"].min(), pos["date"].max())
    n_win = max(int(span.month.isin([12, 1, 2]).sum()), 1)
    n_sum = max(int(span.month.isin([7, 8]).sum()), 1)
    month = pos["date"].dt.month
    total = pos.groupby("zone_id").size()
    win = pos[month.isin([12, 1, 2])].groupby("zone_id").size().reindex(total.index, fill_value=0) / n_win
    summ = pos[month.isin([7, 8])].groupby("zone_id").size().reindex(total.index, fill_value=0) / n_sum
    flagged = sorted(total[(total >= min_days) & (win > ratio * summ)].index)
    # Tableau de preuves : pour juger à l'œil si l'exclusion est justifiée
    detail = pd.DataFrame({"jours_feu": total, "pct_jours_hiver": 100 * win,
                           "pct_jours_juil_aout": 100 * summ}).loc[flagged].round(1)
    return flagged, detail


# ------------------------------------------------------ 3. Features météo
def add_weather_features(w):
    """Features par zone ET par année (la météo n'est téléchargée que d'avril à octobre :
    on ne calcule donc jamais une fenêtre à cheval sur deux années)."""
    w = w.rename(columns=RAW).sort_values(["zone_id", "date"]).reset_index(drop=True)
    w["year"] = w["date"].dt.year
    keys = [w["zone_id"], w["year"]]
    g = w.groupby(["zone_id", "year"], sort=False)

    # Jours secs consécutifs : compteur qui repart à 0 chaque fois qu'il pleut
    dry = (w["precip"] < DRY_MM).astype(int)
    run_id = (1 - dry).groupby(keys).cumsum()
    w["dry_days"] = dry.groupby(keys + [run_id]).cumsum()

    # Cumuls de pluie et moyennes glissantes (fenêtre qui INCLUT le jour D, jamais après)
    w["rain_7d"] = g["precip"].transform(lambda s: s.rolling(7, min_periods=7).sum())
    w["rain_30d"] = g["precip"].transform(lambda s: s.rolling(30, min_periods=30).sum())
    w["tmax_mean3"] = g["tmax"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    w["rh_min_mean3"] = g["rh_min"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    # Retards : valeur de la veille
    w["tmax_lag1"] = g["tmax"].shift(1)
    w["rh_min_lag1"] = g["rh_min"].shift(1)

    w["month"] = w["date"].dt.month
    w["doy"] = w["date"].dt.day_of_year
    return w


def add_anomalies(w, train_end):
    """Écart à la "normale" de la zone = moyenne du MÊME MOIS sur les années d'ENTRAÎNEMENT.
    Pourquoi : ce qui déclenche les feux, c'est une chaleur/sécheresse INHABITUELLE pour la zone,
    pas 35 °C en valeur absolue (normal au sud, canicule au nord). Les normales n'utilisent que
    le train : aucune information des années de validation/test n'y entre."""
    cols = ["tmax", "rh_min", "soil"]
    normal = (w[w["year"] <= train_end].groupby(["zone_id", "month"])[cols].mean()
              .add_suffix("_norm").reset_index())
    w = w.merge(normal, on=["zone_id", "month"], how="left")
    for c in cols:
        w[f"{c}_anom"] = w[c] - w[f"{c}_norm"]
    return w.drop(columns=[f"{c}_norm" for c in cols])


def add_zone_prior(df, train_end, shrink=30):
    """zone_rate = fréquence historique de feu de la zone (sur les années d'entraînement),
    "rétrécie" vers la moyenne globale quand la zone a peu de données.
    Pour une ligne d'ENTRAÎNEMENT, on retire l'année de la ligne elle-même du calcul
    (sinon le modèle verrait ses propres étiquettes = fuite). Pour validation/test,
    on utilise toutes les années d'entraînement."""
    tr = df[df["year"] <= train_end]
    per_year = (tr.groupby(["zone_id", "year"])["label_d1"].agg(s_y="sum", n_y="count").reset_index())
    total = (per_year.groupby("zone_id")[["s_y", "n_y"]].sum()
             .rename(columns={"s_y": "S", "n_y": "N"}).reset_index())
    g = tr["label_d1"].mean()
    df = df.merge(total, on="zone_id", how="left").merge(per_year, on=["zone_id", "year"], how="left")
    df[["S", "N", "s_y", "n_y"]] = df[["S", "N", "s_y", "n_y"]].fillna(0)
    df["zone_rate"] = (df["S"] - df["s_y"] + shrink * g) / (df["N"] - df["n_y"] + shrink)
    return df.drop(columns=["S", "N", "s_y", "n_y"])


def add_fire_history(df, pos, static, n_lat, n_lon):
    """Feux RÉCENTS, disponibles le jour D (l'application les aura via FIRMS temps réel) :
      fire_1d     : feu dans la zone le jour D            fire_3d : au moins un feu sur D-2..D
      fire_7d     : nombre de jours avec feu sur D-6..D
      nbr_fire_1d : feu dans une des 8 zones voisines le jour D
      nbr_fire_7d : nombre de jours avec feu chez les voisines sur D-6..D
    Jamais de jour après D (les fenêtres se terminent à D). Les zones à source fixe sont retirées
    de l'historique, sinon elles apparaîtraient comme des "feux voisins" permanents."""
    days = pd.date_range(min(pos["date"].min(), df["date"].min()), max(pos["date"].max(), df["date"].max()))
    p = pos[~pos["zone_id"].isin(static)]
    A = np.zeros((n_lat, n_lon, len(days)), dtype=np.int8)          # grille x jours : 1 = feu
    A[p["zone_id"].str[1:4].astype(int).to_numpy(), p["zone_id"].str[5:8].astype(int).to_numpy(),
      days.get_indexer(p["date"])] = 1

    def window_sum(X, k):                                            # somme sur les k jours D-k+1..D
        c = np.cumsum(X, axis=2, dtype=np.int32)
        lag = np.zeros_like(c)
        lag[:, :, k:] = c[:, :, :-k]
        return c - lag

    P = np.pad(A, ((1, 1), (1, 1), (0, 0)))
    nb = sum(P[1 + di:1 + di + n_lat, 1 + dj:1 + dj + n_lon]
             for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0))
    nb = (nb > 0).astype(np.int8)                                    # au moins un voisin en feu
    feats = {"fire_1d": A, "fire_3d": (window_sum(A, 3) > 0).astype(np.int8), "fire_7d": window_sum(A, 7),
             "nbr_fire_1d": nb, "nbr_fire_7d": window_sum(nb, 7)}
    zi = df["zone_id"].str[1:4].astype(int).to_numpy()
    zj = df["zone_id"].str[5:8].astype(int).to_numpy()
    ti = days.get_indexer(df["date"])
    for name, arr in feats.items():
        df[name] = arr[zi, zj, ti]
    return df


# ------------------------------------------------------------ 4. Étiquettes
def add_labels(df, pos, horizons=(1, 2, 3)):
    for h in horizons:
        # un feu le jour D+h dans la zone  <=>  la ligne (zone, D) est positive pour l'horizon h
        key = pos.assign(date=pos["date"] - pd.Timedelta(days=h))[["zone_id", "date"]]
        key[f"label_d{h}"] = 1
        df = df.merge(key, how="left", on=["zone_id", "date"])
        df[f"label_d{h}"] = df[f"label_d{h}"].fillna(0).astype(int)
    return df


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="data/grid.csv")
    ap.add_argument("--firms-dir", default="data/raw/firms/VIIRS_SNPP_SP")
    ap.add_argument("--weather", default="data/weather_daily.csv")
    ap.add_argument("--out", default="data/dataset.csv")
    ap.add_argument("--months", nargs="+", type=int, default=[6, 7, 8, 9, 10],
                    help="mois gardés pour l'entraînement (défaut : juin à octobre)")
    ap.add_argument("--train-end", type=int, default=2022, help="dernière année d'entraînement")
    ap.add_argument("--valid-year", type=int, default=2023, help="année de validation (test = années suivantes)")
    args = ap.parse_args()

    bbox = os.getenv("STUDY_BBOX", "6.0,34.5,11.6,37.5")
    step = float(os.getenv("GRID_STEP_DEG", "0.25"))
    west, south, east, north = (float(x) for x in bbox.split(","))
    n_lat, n_lon = int(round((north - south) / step)), int(round((east - west) / step))

    grid = pd.read_csv(args.grid)
    pos = load_fires(args.firms_dir, west, south, step, n_lat, n_lon)

    static, detail = find_static_zones(pos)
    detail = detail.join(grid.set_index("zone_id")[["lat", "lon"]])   # centre de la zone
    print(f"Zones exclues (source de chaleur fixe probable) : {len(static)}")
    print(detail.to_string())
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).with_name("excluded_zones.txt").write_text("\n".join(static), encoding="utf-8")

    w = pd.read_csv(args.weather, parse_dates=["date"])
    # Quelques fichiers téléchargés au tout début contiennent l'année entière ; on ne garde que
    # avril-octobre pour que TOUTES les zones aient exactement la même fenêtre de calcul.
    w = w[w["date"].dt.month.between(4, 10)]
    w = add_weather_features(w)
    w = add_anomalies(w, args.train_end)
    w = w.merge(grid[["zone_id", "lat", "lon"]], on="zone_id", how="left")
    df = add_labels(w, pos)
    df = add_fire_history(df, pos, static, n_lat, n_lon)

    # Filtres : zones valides, mois de saison, features complètes
    df = df[~df["zone_id"].isin(static)]
    df = df[df["month"].isin(args.months)]
    before = len(df)
    df = df.dropna(subset=[f for f in FEATURES if f != "zone_rate"])
    print(f"Lignes : {before} -> {len(df)} après suppression des features incomplètes")
    df = add_zone_prior(df, args.train_end)   # après les filtres : taux calculé sur la saison retenue

    # Découpe chronologique (jamais aléatoire : sinon on "triche" avec le futur)
    df["split"] = np.select([df["year"] <= args.train_end, df["year"] == args.valid_year],
                            ["train", "valid"], default="test")
    cols = ["zone_id", "date", "split"] + FEATURES + FIRE_FEATURES + ["label_d1", "label_d2", "label_d3"]
    df[cols].to_csv(args.out, index=False)

    print(f"\nJeu de données écrit : {args.out}  ({len(df)} lignes, {df['zone_id'].nunique()} zones)")
    print(f"{'split':6s} {'années':>11s} {'lignes':>9s} {'positifs(d1)':>13s} {'taux':>7s}")
    for s in ("train", "valid", "test"):
        d = df[df["split"] == s]
        if len(d):
            print(f"{s:6s} {d['year'].min()}-{d['year'].max():<6d} {len(d):9d} {int(d['label_d1'].sum()):13d} "
                  f"{100 * d['label_d1'].mean():6.2f}%")