"""Télécharge la météo quotidienne (archive ERA5 via Open-Meteo) pour chaque zone.

Pour chaque zone on interroge le CENTRE de la cellule (colonnes lat/lon de data/grid.csv).
Pour limiter le nombre de requêtes, on envoie plusieurs zones par requête (BATCH).

Comme fetch_firms.py, le script est REPRENABLE : chaque (année, lot de zones) est
sauvé dans un fichier JSON ; relancer le script saute ce qui existe déjà.

SAISON : on ne télécharge que du 1er avril au 31 octobre (le modèle s'entraîne sur
juin-octobre ; avril-mai servent aux variables "cumul de pluie sur 30 jours", etc.).
Cela réduit de 40 % le volume demandé. --full-year pour télécharger toute l'année.

LIMITES D'APPELS : Open-Meteo compte une requête "lourde" (plusieurs zones x longue
période) comme PLUSIEURS appels. Le script lit le message d'erreur 429 et réagit :
  - limite par minute : il attend 65 s puis réessaie tout seul ;
  - limite par heure  : il attend 10 min puis réessaie (jusqu'à 7 fois) ;
  - limite par jour   : il s'arrête ; relancez-le demain, il reprendra.

Prérequis : python ingestion/grid.py  (crée data/grid.csv)

Exemples :
  python ingestion/fetch_weather.py --years 2025 2025 --max-zones 10   # petit test
  python ingestion/fetch_weather.py --years 2015 2025                  # tout
  python ingestion/fetch_weather.py --flatten                          # JSON -> data/weather_daily.csv
"""
import argparse
import csv
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://archive-api.open-meteo.com/v1/archive"
# Variables quotidiennes (noms vérifiés par un vrai appel sur l'API)
DAILY_VARS = [
    "temperature_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "precipitation_sum",
    "soil_moisture_0_to_7cm_mean",
    "et0_fao_evapotranspiration",
]
BATCH = 10  # zones par requête
RAW_DIR = Path("data/raw/weather")
SEASON = ("04-01", "10-31")  # (début, fin) de la période téléchargée chaque année
# Pause entre deux requêtes. Observé : ~7 requêtes passent par minute avant la limite,
# donc 9 s évite de la toucher (et d'obtenir des réponses coupées).
PAUSE = 9


class StopDownload(Exception):
    """Limite quotidienne (ou erreur inconnue) : on arrête proprement."""


def load_zones(path="data/grid.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch_batch(zones, start_date, end_date):
    """Une requête : météo de la période pour un lot de zones."""
    params = {
        "latitude": ",".join(z["lat"] for z in zones),
        "longitude": ",".join(z["lon"] for z in zones),
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC",
    }
    url = URL + "?" + urllib.parse.urlencode(params, safe=",")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    # Avec plusieurs coordonnées l'API renvoie une liste ; avec une seule, un objet.
    return data if isinstance(data, list) else [data]


def fetch_with_limits(zones, start_date, end_date):
    """Comme fetch_batch, mais gère les erreurs 429 selon le type de limite."""
    hourly_waits = 0
    broken = 0
    while True:
        try:
            return fetch_batch(zones, start_date, end_date)
        # ATTENTION à l'ordre : HTTPError doit être traité AVANT URLError (c'est une sous-classe).
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            reason = e.read().decode("utf-8", "replace")[:300]
            low = reason.lower()
            if "minutely" in low:
                print("  limite par minute atteinte -> pause de 65 s")
                time.sleep(65)
            elif "hourly" in low and hourly_waits < 7:
                hourly_waits += 1
                print(f"  limite par heure atteinte -> pause de 10 min ({hourly_waits}/7)")
                time.sleep(600)
            else:
                raise StopDownload(f"Limite atteinte (probablement quotidienne). "
                                   f"Message du serveur : {reason!r}\n"
                                   "Relancez la même commande plus tard (demain si besoin) : "
                                   "le script reprendra où il s'est arrêté.")
        except (json.JSONDecodeError, http.client.IncompleteRead, TimeoutError,
                ConnectionError, urllib.error.URLError) as e:
            # Réponse coupée en plein milieu / connexion perdue : temporaire, on réessaie.
            broken += 1
            if broken > 5:
                raise StopDownload(f"Réponses coupées 5 fois de suite ({e}). "
                                   "Relancez plus tard.")
            print(f"  réponse incomplète ({type(e).__name__}) -> pause de 65 s ({broken}/5)")
            time.sleep(65)


def download(zones, y0, y1, full_year=False):
    batches = [zones[i:i + BATCH] for i in range(0, len(zones), BATCH)]
    total = (y1 - y0 + 1) * len(batches)
    done = 0
    start_md, end_md = ("01-01", "12-31") if full_year else SEASON
    for year in range(y0, y1 + 1):
        out = RAW_DIR / str(year)
        out.mkdir(parents=True, exist_ok=True)
        for k, batch in enumerate(batches):
            done += 1
            f = out / f"batch_{k:03d}.json"
            if f.exists():
                continue
            try:
                results = fetch_with_limits(batch, f"{year}-{start_md}", f"{year}-{end_md}")
            except StopDownload as e:
                print(f"\n{e}\n(requête {done}/{total})")
                return
            # On garde les zone_id dans l'ordre de la requête (l'API "arrondit" les
            # coordonnées, donc on ne se fie pas à lat/lon de la réponse).
            payload = {"zone_ids": [z["zone_id"] for z in batch],
                       "results": [r["daily"] for r in results]}
            f.write_text(json.dumps(payload), encoding="utf-8")
            print(f"  {done}/{total} requêtes (année {year}, lot {k})")
            time.sleep(PAUSE)
    print("Terminé.")


def flatten(out_path="data/weather_daily.csv"):
    """Assemble tous les JSON en un CSV : une ligne = (zone_id, date, variables...)."""
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["zone_id", "date"] + DAILY_VARS)
        for f in sorted(RAW_DIR.glob("*/batch_*.json")):
            payload = json.loads(f.read_text(encoding="utf-8"))
            for zone_id, daily in zip(payload["zone_ids"], payload["results"]):
                for i, day in enumerate(daily["time"]):
                    w.writerow([zone_id, day] + [daily[v][i] for v in DAILY_VARS])
                    n += 1
    print(f"{n} lignes écrites dans {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs=2, type=int, default=[2015, 2025], metavar=("DEBUT", "FIN"))
    ap.add_argument("--max-zones", type=int, help="limiter le nombre de zones (pour tester)")
    ap.add_argument("--full-year", action="store_true", help="toute l'année au lieu d'avril-octobre")
    ap.add_argument("--flatten", action="store_true", help="assembler les JSON en CSV")
    args = ap.parse_args()

    if args.flatten:
        flatten()
    else:
        zones = load_zones()
        if args.max_zones:
            zones = zones[:args.max_zones]
        print(f"{len(zones)} zones, années {args.years[0]}-{args.years[1]}")
        download(zones, *args.years, full_year=args.full_year)