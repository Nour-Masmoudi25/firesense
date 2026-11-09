"""Entraîne et évalue les modèles de risque d'incendie (horizon J+1 par défaut).

Trois modèles, du plus simple au plus puissant :
  climatology : AUCUNE météo. Probabilité = fréquence historique de feu pour (zone, mois).
                C'est la barre à battre : si la météo n'apporte rien, XGBoost ne fera pas mieux.
  logistic    : régression logistique sur les features météo (baseline "ML simple").
  xgboost     : modèle principal (arbres de décision boostés).

Évaluation (le déséquilibre de classes rend la précision inutilisable) :
  - AUC-PR  (average precision) : métrique principale. Un modèle au hasard obtient le TAUX DE POSITIFS.
  - ROC-AUC : secondaire.
  - rappel / précision dans le top 5 % : "si on alerte sur les 5 % de zone-jours les plus risqués,
    quelle part des feux attrape-t-on ?"  C'est la lecture opérationnelle.
  - Brier : qualité de la probabilité (plus bas = mieux). Courbe de calibration affichée.

Prérequis : data/dataset.csv (python ml/build_dataset.py)

Usage :
  python ml/train.py                       # climatology + logistic + xgboost (si installé)
  python ml/train.py --models climatology logistic
  python ml/train.py --horizon 2           # prédire J+2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import FEATURES, FIRE_FEATURES  # noqa: E402  (mêmes listes que le jeu de données)


# ------------------------------------------------------------- baseline 1
class Climatology:
    """P(feu | zone, mois) estimée sur le train, "rétrécie" vers la fréquence du mois
    quand on a peu d'observations pour une zone (évite 0 % ou 100 % sur 3 jours de données)."""

    def __init__(self, shrink=20):
        self.shrink = shrink

    def fit(self, df, label):
        self.month_rate = df.groupby("month")[label].mean()
        self.global_rate = df[label].mean()
        g = df.groupby(["zone_id", "month"])[label].agg(["sum", "count"])
        self.table = g
        return self

    def predict_proba(self, df):
        m_rate = df["month"].map(self.month_rate).fillna(self.global_rate)
        idx = pd.MultiIndex.from_frame(df[["zone_id", "month"]])
        s = self.table["sum"].reindex(idx).fillna(0).to_numpy()
        n = self.table["count"].reindex(idx).fillna(0).to_numpy()
        return (s + self.shrink * m_rate.to_numpy()) / (n + self.shrink)


# --------------------------------------------------------------- métriques
def evaluate(y, p):
    y = np.asarray(y)
    k = max(int(0.05 * len(y)), 1)
    top = np.argsort(-p, kind="stable")[:k]                    # les 5 % les plus risqués
    return {
        "base_rate": y.mean(),
        "AUC-PR": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan"),
        "rappel_top5%": y[top].sum() / max(y.sum(), 1),
        "precision_top5%": y[top].mean(),
        "Brier": brier_score_loss(y, np.clip(p, 0, 1)),
    }


def calibration_table(y, p, bins=8):
    """Par tranche de probabilité prédite : taux de feu réellement observé."""
    q = pd.qcut(pd.Series(p), q=bins, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": np.asarray(y), "tranche": q.values}).groupby("tranche", observed=True)
    return t.agg(proba_moyenne=("p", "mean"), taux_observe=("y", "mean"), n=("y", "size")).round(4)


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--models", nargs="+", default=["climatology", "logistic", "xgboost"])
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--calibration", action="store_true", help="afficher la courbe de calibration")
    ap.add_argument("--fire-history", action="store_true",
                    help="ajouter les feux récents (zone + voisines) aux features : météo + feux récents")
    args = ap.parse_args()

    label = f"label_d{args.horizon}"
    FEATS = FEATURES + (FIRE_FEATURES if args.fire_history else [])
    print("Features :", "météo + feux récents" if args.fire_history else "météo seule", f"({len(FEATS)})")
    df = pd.read_csv(args.data, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    train = df[df["split"] == "train"]
    evals = {s: df[df["split"] == s] for s in ("valid", "test") if (df["split"] == s).any()}
    # Garde-fou : une "année de test" minuscule (ex. un test de téléchargement) n'a pas de sens
    evals = {s: d for s, d in evals.items() if d[label].sum() >= 20}
    if not evals:
        raise SystemExit("Aucun jeu de validation/test avec assez de feux (>= 20). "
                         "Attendez le reste de la météo ou ajustez --train-end/--valid-year.")
    print(f"Horizon J+{args.horizon} | train : {len(train)} lignes, {int(train[label].sum())} feux "
          f"({100 * train[label].mean():.2f} %)")
    for s, d in evals.items():
        print(f"{s:6s} : {len(d)} lignes, {int(d[label].sum())} feux ({100 * d[label].mean():.2f} %)")

    models = {}
    spw = 1.0   # poids de la classe positive (XGBoost le recalcule plus bas)
    X_tr, y_tr = train[FEATS], train[label]

    if "climatology" in args.models:
        models["climatology"] = Climatology().fit(train, label)

    if "logistic" in args.models:
        # Sans class_weight : on veut des probabilités CALIBRÉES (le classement ne change pas).
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        models["logistic"] = lr.fit(X_tr, y_tr)

    if "xgboost" in args.models:
        try:
            import xgboost as xgb
        except ImportError:
            print("\n[xgboost non installé : `pip install xgboost` puis relancez]")
        else:
            # scale_pos_weight = négatifs / positifs : compense le déséquilibre de classes
            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            valid_name = "valid" if "valid" in evals else next(iter(evals))
            v = evals[valid_name]
            xg = xgb.XGBClassifier(
                n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=spw,
                eval_metric="aucpr", early_stopping_rounds=40, n_jobs=-1, random_state=42)
            # L'arrêt précoce regarde la validation : le modèle s'arrête quand elle ne s'améliore plus.
            xg.fit(X_tr, y_tr, eval_set=[(v[FEATS], v[label])], verbose=False)
            models["xgboost"] = xg
            print(f"\nXGBoost : {xg.best_iteration + 1} arbres retenus (scale_pos_weight = {spw:.1f})")

    # ---- Évaluation
    def proba(name, m, d):
        if name == "climatology":
            return m.predict_proba(d)
        p = m.predict_proba(d[FEATS])[:, 1]
        if name == "xgboost":
            # scale_pos_weight = w multiplie les COTES (p/(1-p)) prédites par w. On le défait :
            # p_vrai = p / (p + w * (1 - p)). Le classement ne change pas, mais la probabilité
            # redevient réaliste (indispensable pour les niveaux de risque et le score de Brier).
            p = p / (p + spw * (1 - p))
        return p

    for s, d in evals.items():
        rows = {}
        for name, m in models.items():
            rows[name] = evaluate(d[label], proba(name, m, d))
        print(f"\n=== Résultats sur {s} ({int(d[label].sum())} feux / {len(d)} lignes) ===")
        print(pd.DataFrame(rows).T.round(4).to_string())
        print(f"(un modèle au hasard aurait une AUC-PR de {d[label].mean():.4f})")
        if args.calibration:
            for name, m in models.items():
                print(f"\nCalibration {name} ({s}) :")
                print(calibration_table(d[label], proba(name, m, d)).to_string())

    # ---- Explicabilité simple
    if "logistic" in models:
        coefs = models["logistic"][-1].coef_[0]
        order = np.argsort(-np.abs(coefs))[:8]
        print("\nRégression logistique : 8 variables les plus influentes (coef. sur variables standardisées)")
        for i in order:
            print(f"  {FEATS[i]:14s} {coefs[i]:+.3f}")
    if "xgboost" in models:
        imp = models["xgboost"].feature_importances_
        order = np.argsort(-imp)[:8]
        print("\nXGBoost : 8 variables les plus importantes")
        for i in order:
            print(f"  {FEATS[i]:14s} {imp[i]:.3f}")