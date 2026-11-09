"""Génère la grille de zones (cellules de ~0,25°) sur la zone d'étude.

Une zone = une cellule de la grille. C'est l'unité de prédiction :
une ligne du jeu d'entraînement = (zone, jour).

Usage :  python ingestion/grid.py      -> écrit data/grid.csv
"""
import csv
import os
from pathlib import Path


def build_grid(west, south, east, north, step):
    """Renvoie la liste des cellules : id + bornes + centre (lat/lon)."""
    cells = []
    # On parcourt la zone de l'ouest vers l'est, du sud vers le nord.
    # round(..., 4) évite les erreurs d'arrondi des flottants (0.1+0.2 != 0.3).
    n_lon = int(round((east - west) / step))
    n_lat = int(round((north - south) / step))
    for i in range(n_lat):
        for j in range(n_lon):
            c_south = round(south + i * step, 4)
            c_west = round(west + j * step, 4)
            cells.append({
                "zone_id": f"Z{i:03d}_{j:03d}",   # ex. Z012_034 (ligne, colonne)
                "west": c_west,
                "south": c_south,
                "east": round(c_west + step, 4),
                "north": round(c_south + step, 4),
                "lat": round(c_south + step / 2, 4),  # centre de la cellule
                "lon": round(c_west + step / 2, 4),
            })
    return cells


def zone_of(lat, lon, west, south, step):
    """Trouve la zone qui contient un point (ex. une détection de feu)."""
    i = int((lat - south) // step)
    j = int((lon - west) // step)
    return f"Z{i:03d}_{j:03d}"


if __name__ == "__main__":
    bbox = os.getenv("STUDY_BBOX", "6.0,34.5,11.6,37.5")
    step = float(os.getenv("GRID_STEP_DEG", "0.25"))
    west, south, east, north = (float(x) for x in bbox.split(","))

    cells = build_grid(west, south, east, north, step)

    out = Path("data/grid.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        writer.writeheader()
        writer.writerows(cells)
    print(f"{len(cells)} zones écrites dans {out}")
