"""Recalcula só as divergências (PSI, KL nas duas direções, Jensen-Shannon) e o IV, sem retreinar a disputa.

Usa exatamente os mesmos splits dos scripts de treino e o modelo que já venceu:
  credito            : carrega o xgb.json salvo pelo train_credito.py
  turnover/aquisicao : reajusta o modelo escolhido com os hiperparâmetros gravados em comparacao.csv
e confere se as probabilidades do teste batem com scores_teste.csv antes de gravar qualquer coisa.

Uso:  python train_divergencias.py --data data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from moiras.compare import SEED, modelos
from moiras.credito import TARGET, fit_caps, padronizar, prepare
from moiras.metrics import exportar_divergencias


def conferir(p_novo, p_salvo, nome: str) -> None:
    dif = float(np.max(np.abs(np.asarray(p_novo) - np.asarray(p_salvo))))
    print(f"  [{nome}] maior diferença vs scores_teste.csv: {dif:.2e}")
    if dif > 1e-4:
        raise SystemExit(f"{nome}: o modelo reconstruído não bate com o salvo — rode o treino completo.")


def credito(data: Path, art: Path) -> None:
    out = art / "credito"
    raw = padronizar(pd.read_csv(data / "cs-training.csv", index_col=0))
    rtv, rte = train_test_split(raw, test_size=0.20, stratify=raw[TARGET], random_state=SEED)
    rtr, _ = train_test_split(rtv, test_size=0.25, stratify=rtv[TARGET], random_state=SEED)
    caps = json.loads((out / "caps.json").read_text(encoding="utf-8"))
    prep = lambda r: prepare(r, caps)
    Xtr, Xte = (prep(r).drop(columns=TARGET) for r in (rtr, rte))
    model = xgb.XGBClassifier()
    model.load_model(out / "xgb.json")
    p_te = model.predict_proba(Xte)[:, 1]
    conferir(p_te, pd.read_csv(out / "scores_teste.csv").p_xgb, "credito")
    Xkt = prep(padronizar(pd.read_csv(data / "cs-test.csv", index_col=0)).drop(columns=[TARGET], errors="ignore"))
    Xkt = Xkt[Xtr.columns]
    exportar_divergencias(Xtr, rtr[TARGET], Xkt, model.predict_proba(Xtr)[:, 1], model.predict_proba(Xkt)[:, 1],
                          out, "SCORE (prob. XGBoost)", p_teste=p_te, y_teste=rte[TARGET])


def caso(nome: str, data: Path, art: Path) -> None:
    from train_cases import CASOS
    out = art / nome
    Xtr, ytr, Xte, yte, _, grade, _ = CASOS[nome](data)
    cmp_ = pd.read_csv(out / "comparacao.csv")
    esc = cmp_.loc[cmp_.escolhido, "modelo"].iloc[0]
    params = json.loads(cmp_.loc[cmp_.escolhido, "params"].iloc[0])
    est, grid = modelos(len(Xtr), grade)[esc]
    chaves = {k.split("__")[-1]: k for k in grid}  # "C" -> "logisticregression__C" quando é pipeline
    est.set_params(**{chaves.get(k, k): v for k, v in params.items()})
    est.fit(Xtr, ytr)
    p_te = est.predict_proba(Xte)[:, 1]
    conferir(p_te, pd.read_csv(out / "scores_teste.csv")[esc], f"{nome} ({esc})")
    exportar_divergencias(Xtr, ytr, Xte, est.predict_proba(Xtr)[:, 1], p_te, out, "SCORE do modelo escolhido",
                          p_teste=p_te, y_teste=yte, so_continuas=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    data, art = Path(a.data), Path(a.out)
    credito(data, art)
    for n in ("turnover", "aquisicao"):
        caso(n, data, art)
    print("ok: psi.csv, iv.csv e iv_faixas.csv atualizados nos três cases")
