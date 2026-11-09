"""Exploration des feux FIRMS téléchargés : quel est le vrai taux de positifs ?

Ce script répond à la question : « sur toutes les lignes (zone x jour), quelle
proportion a au moins un feu de végétation détecté ? »  C'est le taux de la classe
positive, qui détermine la difficulté du problème (déséquilibre de classes).

Filtres appliqués (justifiés par nos observations) :
  - type == 0        : feu de végétation présumé (on écarte sources fixes, volcans, mer)
  - confiance l écartée : on garde n (nominal) et h (haute)

Usage (depuis la racine du projet) :
  python ingestion/explore_firms.py
  python ingestion/explore_firms.py --start 2015-01-01 --end 2025-12-31
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grid import zone_of  # noqa: E402  (notre fonction : point -> zone)


def read_rows(folder):
    for f in sorted(Path(folder).glob("*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh)


def month_lengths(start, end):
    """Nombre de jours observés pour chaque numéro de mois (1..12) sur [start, end]."""
    days = Counter()
    d = start
    while d <= end:
        days[d.month] += 1
        d += timedelta(days=1)
    return days


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/firms/VIIRS_SNPP_SP")
    ap.add_argument("--start", help="début de la période (défaut : 1re détection)")
    ap.add_argument("--end", help="fin de la période (défaut : dernière détection)")
    args = ap.parse_args()

    bbox = os.getenv("STUDY_BBOX", "6.0,34.5,11.6,37.5")
    step = float(os.getenv("GRID_STEP_DEG", "0.25"))
    west, south, east, north = (float(x) for x in bbox.split(","))
    n_lat = int(round((north - south) / step))
    n_lon = int(round((east - west) / step))
    n_zones = n_lat * n_lon

    total = kept = 0
    positives = set()                 # ensemble de (zone, date) avec au moins un feu
    dates = []
    for r in read_rows(args.dir):
        total += 1
        dates.append(r["acq_date"])
        if r["type"] != "0" or r["confidence"] == "l":
            continue
        lat, lon = float(r["latitude"]), float(r["longitude"])
        i = int((lat - south) // step)
        j = int((lon - west) // step)
        if not (0 <= i < n_lat and 0 <= j < n_lon):
            continue                  # hors grille (bord de la boîte)
        kept += 1
        positives.add((zone_of(lat, lon, west, south, step), r["acq_date"]))

    if not dates:
        raise SystemExit(f"Aucun fichier trouvé dans {args.dir}")

    start = date.fromisoformat(args.start or min(dates))
    end = date.fromisoformat(args.end or max(dates))
    n_days = (end - start).days + 1
    total_zone_days = n_zones * n_days

    print(f"Période : {start} -> {end}  ({n_days} jours, {n_zones} zones)")
    print(f"Détections lues : {total} | gardées après filtres : {kept}")
    print(f"Zone-jours positifs : {len(positives)} sur {total_zone_days} "
          f"= {100 * len(positives) / total_zone_days:.2f} %")

    # Taux par mois de l'année (agrégé sur toutes les années)
    by_month = Counter(int(d[5:7]) for _, d in positives)
    days_per_month = month_lengths(start, end)
    print("\nTaux de zone-jours positifs par mois :")
    for m in range(1, 13):
        if days_per_month[m]:
            rate = 100 * by_month[m] / (n_zones * days_per_month[m])
            print(f"  mois {m:02d} : {by_month[m]:6d} positifs  ({rate:.2f} %)")

    # Taux par année
    by_year = Counter(d[:4] for _, d in positives)
    print("\nZone-jours positifs par année :", dict(sorted(by_year.items())))

    # Zones les plus actives
    by_zone = Counter(z for z, _ in positives)
    print(f"\nZones avec au moins 1 feu : {len(by_zone)} sur {n_zones}")
    # Un vrai feu de forêt est SAISONNIER (été). Une zone qui "brûle" aussi en hiver
    # ressemble plutôt à une source de chaleur fixe (industrie, torchère) mal classée.
    winter = Counter(z for z, d in positives if d[5:7] in ("12", "01", "02"))
    summer = Counter(z for z, d in positives if d[5:7] in ("07", "08"))
    # jours d'hiver / d'été observés (max(...,1) évite une division par zéro sur un petit test)
    n_win = max(days_per_month[12] + days_per_month[1] + days_per_month[2], 1)
    n_sum = max(days_per_month[7] + days_per_month[8], 1)
    lat0 = lambda zid: south + int(zid[1:4]) * step      # latitude sud de la zone
    lon0 = lambda zid: west + int(zid[5:8]) * step       # longitude ouest de la zone
    print("\n15 zones les plus actives :")
    print("  zone      lat_sud  lon_ouest  jours_feu  % jours hiver  % jours juil-août")
    for z, c in by_zone.most_common(15):
        print(f"  {z}   {lat0(z):6.2f}   {lon0(z):6.2f}     {c:6d}     "
              f"{100 * winter[z] / n_win:6.1f}        {100 * summer[z] / n_sum:6.1f}")

    # Zones suspectes : feu détecté plus de 10 % des jours d'hiver
    suspects = sorted(z for z in by_zone if winter[z] / n_win > 0.10)
    print(f"\nZones suspectes (feu > 10 % des jours d'hiver) : {len(suspects)}")
    print("  ", suspects)