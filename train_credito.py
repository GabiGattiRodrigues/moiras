"""Case 1 do Moiras — risco de crédito (Give Me Some Credit, Kaggle).

Roda o pipeline completo e grava tudo em artifacts/credito/ para o app ler:
  baseline (regressão logística) x XGBoost, KS/AUC/lift, calibração, PSI, SHAP.

Uso:  python train_credito.py --data data/
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from moiras.credito import TARGET, fit_caps, log_features, prepare, padronizar
from moiras.metrics import calibration_table, decile_table, psi, summary

SEED = 42


def load(path: Path) -> pd.DataFrame:
    return padronizar(pd.read_csv(path, index_col=0))


def logistic_baseline() -> object:
    # Transformações logarítmicas para caudas pesadas; imputação pela mediana.
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5),
    )


logit_features = log_features


def tune_xgb(Xtr, ytr, Xva, yva) -> tuple[xgb.XGBClassifier, dict]:
    """Grade pequena, escolhida pela AUC na validação (com early stopping)."""
    grid = itertools.product([3, 4, 5], [20, 50], [0.7, 0.9])
    best, best_auc, best_params = None, -1, None
    for depth, mcw, sub in grid:
        m = xgb.XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=depth, min_child_weight=mcw,
            subsample=sub, colsample_bytree=0.8, reg_lambda=5, eval_metric="auc",
            early_stopping_rounds=100, tree_method="hist", random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        auc = m.best_score
        if auc > best_auc:
            best, best_auc = m, auc
            best_params = {"max_depth": depth, "min_child_weight": mcw, "subsample": sub,
                           "n_arvores": int(m.best_iteration + 1)}
    return best, best_params


def main(data_dir: str, out_dir: str) -> None:
    out = Path(out_dir) / "credito"
    out.mkdir(parents=True, exist_ok=True)

    raw = load(Path(data_dir) / "cs-training.csv")
    # Base não tem data -> split estratificado (limitação declarada no app).
    # O split é feito ANTES da limpeza: os limites (caps) vêm só do treino.
    rtv, rte = train_test_split(raw, test_size=0.20, stratify=raw[TARGET], random_state=SEED)
    rtr, rva = train_test_split(rtv, test_size=0.25, stratify=rtv[TARGET], random_state=SEED)
    caps = fit_caps(rtr)
    (out / "caps.json").write_text(json.dumps(caps, indent=2), encoding="utf-8")
    prep = lambda r: prepare(r, caps)
    Xtr, Xva, Xte = (prep(r).drop(columns=TARGET) for r in (rtr, rva, rte))
    ytr, yva, yte = rtr[TARGET], rva[TARGET], rte[TARGET]
    X, y = Xtr, raw[TARGET]

    # --- baseline
    lr = logistic_baseline().fit(logit_features(Xtr), ytr)
    p_lr = lr.predict_proba(logit_features(Xte))[:, 1]

    # --- XGBoost
    model, params = tune_xgb(Xtr, ytr, Xva, yva)
    p_xgb = model.predict_proba(Xte)[:, 1]
    model.save_model(out / "xgb.json")

    metrics = {
        "n_treino": len(Xtr), "n_validacao": len(Xva), "n_teste": len(Xte),
        "taxa_evento": float(raw[TARGET].mean()), "params_xgb": params,
        "logistica": summary(yte, p_lr), "xgboost": summary(yte, p_xgb),
    }

    # --- tabelas de decis / calibração
    decile_table(yte, p_xgb).to_csv(out / "decis_xgb.csv", index=False)
    decile_table(yte, p_lr).to_csv(out / "decis_lr.csv", index=False)
    calibration_table(yte, p_xgb).to_csv(out / "calibracao_xgb.csv", index=False)
    calibration_table(yte, p_lr).to_csv(out / "calibracao_lr.csv", index=False)
    pd.DataFrame({"y": yte.to_numpy(), "p_xgb": p_xgb, "p_lr": p_lr}).to_csv(out / "scores_teste.csv", index=False)

    # --- PSI: base de treino (referência) x base de teste do Kaggle (features) e x score
    test_raw = load(Path(data_dir) / "cs-test.csv").drop(columns=[TARGET], errors="ignore")
    Xkt = prep(test_raw)[X.columns]
    Xkt.to_csv(out / "base_nova.csv.gz")  # fila de propostas novas (sem rótulo) que o app vai pontuar
    rows = [{"variavel": c, "psi": psi(Xtr[c], Xkt[c])} for c in X.columns]
    rows.append({"variavel": "SCORE (prob. XGBoost)", "psi": psi(model.predict_proba(Xtr)[:, 1],
                                                                model.predict_proba(Xkt)[:, 1])})
    pd.DataFrame(rows).sort_values("psi", ascending=False).to_csv(out / "psi.csv", index=False)

    # --- SHAP (amostra de 3000 do teste)
    samp = Xte.sample(3000, random_state=SEED)
    expl = shap.TreeExplainer(model)
    sv = expl(samp)
    pd.DataFrame(sv.values, columns=X.columns, index=samp.index).to_csv(out / "shap_valores.csv")
    samp.to_csv(out / "shap_features.csv")
    metrics["shap_base"] = float(np.ravel(sv.base_values)[0])
    pd.DataFrame({"variavel": X.columns,
                  "shap_medio_abs": np.abs(sv.values).mean(axis=0)}
                 ).sort_values("shap_medio_abs", ascending=False).to_csv(out / "shap_importancia.csv", index=False)

    (out / "metricas.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    main(a.data, a.out)
