"""Métricas e utilitários comuns a todos os cases do Moiras.

Tudo aqui é agnóstico ao case: recebe y (0/1) e p (probabilidade) e devolve
as métricas que se discutem em modelagem de propensão/risco.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score, roc_curve)


def ks_stat(y, p) -> float:
    """KS = maior distância entre a distribuição acumulada de bons e maus."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summary(y, p) -> dict:
    return {
        "auc": float(roc_auc_score(y, p)),
        "gini": float(2 * roc_auc_score(y, p) - 1),
        "ks": ks_stat(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
    }


def decile_table(y, p, n: int = 10) -> pd.DataFrame:
    """Tabela de decis (1 = maior risco): taxa de evento, lift e ganho acumulado."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decil"] = pd.qcut(df["p"].rank(method="first", ascending=False), n,
                          labels=range(1, n + 1)).astype(int)
    t = df.groupby("decil").agg(clientes=("y", "size"), eventos=("y", "sum"),
                                taxa_evento=("y", "mean"), p_medio=("p", "mean"))
    base = df["y"].mean()
    t["lift"] = t["taxa_evento"] / base
    t["ganho_acum"] = t["eventos"].cumsum() / t["eventos"].sum()
    return t.reset_index()


def calibration_table(y, p, n: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["faixa"] = pd.qcut(df["p"], n, duplicates="drop")
    t = df.groupby("faixa", observed=True).agg(p_medio=("p", "mean"), taxa_real=("y", "mean"),
                                               n=("y", "size")).reset_index(drop=True)
    return t


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index. Faixas definidas pelos quantis da base de referência.

    Regra de bolso: <0,10 estável | 0,10-0,25 atenção | >0,25 mudou de verdade.
    """
    e = pd.Series(expected).dropna().to_numpy()
    a = pd.Series(actual).dropna().to_numpy()
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # variável quase constante
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def profit_curve(y, p, gain_bom: float, perda_mau: float, n: int = 200) -> pd.DataFrame:
    """Lucro esperado ao aprovar quem tem p < corte.

    gain_bom  : margem ganha ao aprovar um bom pagador
    perda_mau : prejuízo ao aprovar um mau pagador
    Negar não custa nada (simplificação declarada: ignora custo de oportunidade).
    """
    y, p = np.asarray(y), np.asarray(p)
    cuts = np.linspace(0.005, 0.5, n)
    rows = []
    for c in cuts:
        aprov = p < c
        bons = int(((y == 0) & aprov).sum())
        maus = int(((y == 1) & aprov).sum())
        rows.append({"corte": c, "aprovacao": aprov.mean(),
                     "inadimplencia_aprovados": maus / max(aprov.sum(), 1),
                     "lucro": bons * gain_bom - maus * perda_mau})
    return pd.DataFrame(rows)


def profit_by_decile(y, p, gain_bom: float, perda_mau: float) -> pd.DataFrame:
    """Lucro esperado cortando por decil de risco, em vez de por probabilidade contínua.

    Decil 1 = 10% mais arriscado, decil 10 = 10% mais seguro (mesma convenção de decile_table).
    Cada linha ``n`` responde: "e se eu aprovar os n decis mais seguros (10, 9, ..., 10-n+1)?" —
    é assim que uma política de crédito real costuma cortar, por faixa, não por casa decimal de probabilidade.
    """
    y, p = np.asarray(y), np.asarray(p)
    ordem = np.argsort(p)  # do mais seguro (menor p) para o mais arriscado
    yo = y[ordem]
    n_total = len(p)
    rows = []
    for n in range(1, 11):
        k = int(round(n_total * n / 10))
        k = max(k, 1)
        aprov_y = yo[:k]
        # corte = maior probabilidade ainda aprovada (fronteira do decil aprovado)
        corte = float(p[ordem][:k].max()) if k > 0 else 0.0
        bons = int((aprov_y == 0).sum())
        maus = int((aprov_y == 1).sum())
        rows.append({"decis_aprovados": n, "corte": corte, "aprovacao": k / n_total,
                     "inadimplencia_aprovados": maus / max(k, 1),
                     "lucro": bons * gain_bom - maus * perda_mau})
    return pd.DataFrame(rows)
