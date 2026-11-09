"""Télécharge l'historique des feux NASA FIRMS par tranches de 5 jours.

Pourquoi 5 jours ? Nous avons testé : l'API refuse 10 jours (réponse vide/erreur),
elle accepte 5 jours maximum par requête.

Le script est REPRENABLE : chaque tranche est sauvée dans son propre fichier
(data/raw/firms/<source>/<date_debut>.csv). Si on le relance, les tranches déjà
téléchargées sont sautées. Utile si la connexion coupe.

N'utilise que la bibliothèque standard Python (rien à installer).

Exemples (depuis la racine du projet) :
  python ingestion/fetch_firms.py --start 2025-06-01 --end 2025-09-30
  python ingestion/fetch_firms.py --start 2015-01-01 --end 2025-12-31
  python ingestion/fetch_firms.py --stats --start 2025-06-01 --end 2025-09-30
"""
import argparse
import csv
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_DAYS = 5  # limite constatée de l'API
HEADER_START = "latitude,longitude"  # une réponse valide commence par cet en-tête


def load_env(path=".env"):
    """Lit le fichier .env (lignes CLE=valeur) sans bibliothèque externe."""
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    # Les variables d'environnement du système ont priorité sur le .env
    env.update({k: v for k, v in os.environ.items() if k in ("FIRMS_MAP_KEY", "STUDY_BBOX")})
    return env


def date_chunks(start, end, step=MAX_DAYS):
    """Découpe [start, end] en tranches de `step` jours : (date_debut, nb_jours)."""
    d = start
    while d <= end:
        n = min(step, (end - d).days + 1)
        yield d, n
        d += timedelta(days=n)


WAITS = [5, 30, 90]  # secondes d'attente entre les essais (on laisse l'API "respirer")


def download_chunk(key, source, bbox, day, n_days):
    """Télécharge une tranche ; renvoie le texte CSV ou lève l'erreur après 4 essais."""
    url = f"{BASE_URL}/{key}/{source}/{bbox}/{n_days}/{day.isoformat()}"
    for attempt in range(len(WAITS) + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                text = resp.read().decode("utf-8")
            # Une réponse valide commence par l'en-tête CSV. Sinon c'est un message d'erreur.
            if not text.startswith(HEADER_START):
                raise RuntimeError(f"Réponse inattendue : {text[:150]!r}")
            return text
        except Exception as e:
            detail = str(e)
            if isinstance(e, urllib.error.HTTPError):
                # Le corps de la réponse d'erreur contient souvent la vraie raison
                body = e.read().decode("utf-8", "replace")[:200]
                detail = f"{e} | message du serveur : {body!r}"
            if attempt == len(WAITS):
                raise RuntimeError(f"tranche {day} abandonnée : {detail}")
            print(f"  tranche {day}, essai {attempt + 1} échoué ({detail}) "
                  f"-> pause de {WAITS[attempt]} s")
            time.sleep(WAITS[attempt])


def fetch_range(key, source, bbox, start, end, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(date_chunks(start, end))
    failed = []
    consecutive = 0
    for i, (day, n) in enumerate(chunks, 1):
        f = out_dir / f"{day.isoformat()}.csv"
        if f.exists():
            continue  # déjà téléchargé
        try:
            text = download_chunk(key, source, bbox, day, n)
        except RuntimeError as e:
            # On n'arrête pas tout pour une tranche : on la note et on continue.
            print(f"  ECHEC {e}")
            failed.append(day)
            consecutive += 1
            if consecutive >= 5:
                print("5 échecs d'affilée : probablement une limite de l'API. "
                      "Attendez 15 à 30 minutes puis relancez la même commande.")
                break
            continue
        consecutive = 0
        f.write_text(text, encoding="utf-8")
        if i % 20 == 0 or i == len(chunks):
            print(f"  {i}/{len(chunks)} tranches (dernière : {day})")
        time.sleep(0.5)  # on reste très en dessous du quota (5000 req / 10 min)
    if failed:
        print(f"\n{len(failed)} tranche(s) non téléchargée(s) : relancez la même "
              "commande pour réessayer uniquement celles-là.")


def read_rows(out_dir):
    """Relit toutes les tranches téléchargées."""
    rows = []
    for f in sorted(out_dir.glob("*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def print_stats(rows):
    """Aperçu rapide : combien de détections par mois, par type, par confiance."""
    print(f"\n{len(rows)} détections au total")
    print("Par mois :", dict(sorted(Counter(r["acq_date"][:7] for r in rows).items())))
    # type : 0 = feu de végétation présumé, 1 = volcan, 2 = source fixe terrestre, 3 = en mer
    print("Par type (0 = feu végétation, 2 = source fixe) :", dict(Counter(r["type"] for r in rows)))
    print("Par confiance (l/n/h pour VIIRS) :", dict(Counter(r["confidence"] for r in rows)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="AAAA-MM-JJ")
    ap.add_argument("--end", required=True, help="AAAA-MM-JJ")
    ap.add_argument("--source", default="VIIRS_SNPP_SP", help="VIIRS_SNPP_SP (archive) ou VIIRS_SNPP_NRT (récent)")
    ap.add_argument("--stats", action="store_true", help="afficher des statistiques à la fin")
    args = ap.parse_args()

    env = load_env()
    key = env.get("FIRMS_MAP_KEY")
    if not key or key == "votre_cle_ici":
        raise SystemExit("FIRMS_MAP_KEY manquante : remplissez le fichier .env")
    bbox = env.get("STUDY_BBOX", "6.0,34.5,11.6,37.5")

    out_dir = Path("data/raw/firms") / args.source
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    print(f"Téléchargement {args.source} {start} -> {end}, zone {bbox}")
    fetch_range(key, args.source, bbox, start, end, out_dir)
    print("Terminé.")
    if args.stats:
        print_stats(read_rows(out_dir))
