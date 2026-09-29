"""Pré-processamento do case de crédito, reutilizado no treino e na aplicação em base nova.

Regra de ouro de produção: qualquer limite/corte usado na limpeza (caps) é
calculado UMA vez no treino e guardado; a base nova é transformada com os mesmos valores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"

RENAME = {
    "RevolvingUtilizationOfUnsecuredLines": "uso_limite",
    "age": "idade",
    "NumberOfTime30-59DaysPastDueNotWorse": "atraso_30_59",
    "DebtRatio": "razao_divida",
    "MonthlyIncome": "renda_mensal",
    "NumberOfOpenCreditLinesAndLoans": "linhas_credito",
    "NumberOfTimes90DaysLate": "atraso_90",
    "NumberRealEstateLoansOrLines": "financ_imoveis",
    "NumberOfTime60-89DaysPastDueNotWorse": "atraso_60_89",
    "NumberOfDependents": "dependentes",
}
ENTRADAS = list(RENAME.values())  # colunas que o cliente precisa trazer


def padronizar(df: pd.DataFrame) -> pd.DataFrame:
    """Aceita nomes originais do Kaggle ou já em português."""
    return df.rename(columns=RENAME)


def fit_caps(raw: pd.DataFrame) -> dict:
    """Limites calculados só na base de treino."""
    return {"uso_limite": float(raw["uso_limite"].quantile(0.999)),
            "renda_mensal": float(raw["renda_mensal"].quantile(0.999)),
            "razao_divida": 5.0}


def prepare(df: pd.DataFrame, caps: dict) -> pd.DataFrame:
    d = df.copy()
    atrasos = ["atraso_30_59", "atraso_60_89", "atraso_90"]
    # 96/98 nas colunas de atraso são códigos de "sem informação", não 98 atrasos.
    d["atraso_codigo"] = (d[atrasos] >= 96).any(axis=1).astype(int)
    d[atrasos] = d[atrasos].where(d[atrasos] < 96, np.nan)
    d.loc[d["idade"] < 18, "idade"] = np.nan  # idade 0 é erro de cadastro
    d["uso_limite"] = d["uso_limite"].clip(upper=caps["uso_limite"])
    d["razao_divida"] = d["razao_divida"].clip(upper=caps["razao_divida"])
    d["renda_faltante"] = d["renda_mensal"].isna().astype(int)
    d["renda_mensal"] = d["renda_mensal"].clip(upper=caps["renda_mensal"])
    d["atrasos_total"] = d[atrasos].sum(axis=1, min_count=1)
    d["renda_per_capita"] = d["renda_mensal"] / (1 + d["dependentes"].fillna(0))
    d["tem_atraso_grave"] = (d["atraso_90"].fillna(0) > 0).astype(int)
    return d


def decil_edges(p_ref, n: int = 10) -> np.ndarray:
    """Fronteiras dos decis de risco a partir de uma distribuição de referência (o teste rotulado): calculadas
    UMA vez e reaplicadas em qualquer contexto (fila nova, um cliente, cenário simulado) — mesma régua sempre,
    é o que torna "decil 3" comparável entre uma tabela e outra."""
    qs = np.quantile(np.asarray(p_ref), np.linspace(0, 1, n + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    return qs


def decil(p, edges) -> pd.Series:
    """Decil de risco (1 = maior probabilidade prevista, 10 = menor), pelas fronteiras fixas de `decil_edges`."""
    labels = list(range(len(edges) - 1, 0, -1))
    return pd.cut(pd.Series(p), edges, labels=labels, include_lowest=True).astype(int)


def log_features(X: pd.DataFrame) -> pd.DataFrame:
    """log1p nas variáveis de cauda pesada. Não muda árvores (transformação monotônica), mas é o que
    deixa a regressão logística justa na disputa."""
    Z = X.copy()
    for col in ["uso_limite", "razao_divida", "renda_mensal", "renda_per_capita", "atraso_30_59", "atraso_60_89",
                "atraso_90", "atrasos_total", "linhas_credito", "financ_imoveis"]:
        Z[col] = np.log1p(Z[col])
    return Z
