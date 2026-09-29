"""Disputa entre famílias de modelo com a mesma régua, decidida por uma regra declarada ANTES de rodar.

Regra (1 erro-padrão): olha-se a AUC de validação cruzada (só no treino). Entre os modelos cuja AUC está a
menos de 1 erro-padrão do melhor, vence o MAIS SIMPLES (logística < árvore < random forest < XGBoost/LightGBM/CatBoost).
O teste final (outra base / outro período) só confirma; não participa da escolha.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from xgboost import XGBClassifier

from moiras.metrics import summary

SEED = 42
# Complexidade usada na regra de desempate (1 EP): entre os empatados, vence o de MENOR número aqui.
# XGBoost/LightGBM/CatBoost ficam no mesmo patamar (4) por serem essencialmente tão complexos quanto o outro —
# nenhum "vence por simplicidade" sobre os outros dois; entre eles, o desempate vai para a melhor AUC de validação.
COMPLEXIDADE = {"Regressão logística": 1, "Árvore de decisão": 2, "Random forest": 3,
                "XGBoost": 4, "LightGBM": 4, "CatBoost": 4}
SLUG = {"Regressão logística": "logistica", "Árvore de decisão": "arvore", "Random forest": "rf",
        "XGBoost": "xgb", "LightGBM": "lgbm", "CatBoost": "catboost"}


def modelos(n_treino: int, grade: str = "normal") -> dict:
    """Devolve {nome: (estimador, grade de hiperparâmetros)}. `grade='leve'` para bases grandes."""
    folha = max(5, int(n_treino * 0.005))  # tamanho mínimo de folha proporcional à base
    leve = grade == "leve"
    return {
        "Regressão logística": (
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000)),
            {"logisticregression__C": [0.01, 0.1, 1.0]}),
        "Árvore de decisão": (
            make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(random_state=SEED)),
            {"decisiontreeclassifier__max_depth": [3, 4, 6] if leve else [2, 3, 4, 5, 6],
             "decisiontreeclassifier__min_samples_leaf": [folha, folha * 3]}),
        "Random forest": (
            make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=100 if leve else 300, n_jobs=-1, random_state=SEED)),
            {"randomforestclassifier__max_depth": [6, 10] if leve else [4, 8, 12],
             "randomforestclassifier__min_samples_leaf": [folha]}),
        "XGBoost": (
            XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="auc",
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=5),
            {"max_depth": [2, 3, 4], "learning_rate": [0.05], "n_estimators": [150, 400],
             "min_child_weight": [5, 20]} if not leve else
            {"max_depth": [3, 4], "learning_rate": [0.05], "n_estimators": [300], "min_child_weight": [20]}),
        "LightGBM": (
            LGBMClassifier(random_state=SEED, n_jobs=-1, subsample=0.8, colsample_bytree=0.8, verbosity=-1),
            {"max_depth": [3, 4, 5], "learning_rate": [0.05], "n_estimators": [150, 400],
             "min_child_samples": [10, 30]} if not leve else
            {"max_depth": [3, 4], "learning_rate": [0.05], "n_estimators": [300], "min_child_samples": [30]}),
        "CatBoost": (
            CatBoostClassifier(random_state=SEED, verbose=False, subsample=0.8, rsm=0.8),
            {"depth": [3, 4, 5], "learning_rate": [0.05], "iterations": [150, 400],
             "min_data_in_leaf": [10, 30]} if not leve else
            {"depth": [3, 4], "learning_rate": [0.05], "iterations": [300], "min_data_in_leaf": [30]}),
    }


@dataclass
class Resultado:
    tabela: pd.DataFrame
    ajustados: dict          # nome -> estimador já treinado no treino todo
    escolhido: str
    melhor_cv: str
    scores: pd.DataFrame     # y + probabilidade de cada modelo no teste


def escolher(tab: pd.DataFrame) -> tuple[str, str]:
    """Regra de 1 erro-padrão. Devolve (escolhido, melhor por AUC).

    Entre os empatados (dentro de 1 EP do melhor), vence o de menor complexidade; entre modelos de MESMA
    complexidade (ex.: XGBoost x LightGBM x CatBoost), o desempate vai para a maior AUC de validação.
    """
    top = tab.loc[tab.cv_auc.idxmax()]
    limite = top.cv_auc - top.cv_ep
    ok = tab[tab.cv_auc >= limite].sort_values(["complexidade", "cv_auc"], ascending=[True, False])
    return ok.iloc[0].modelo, top.modelo


def disputar(Xtr, ytr, Xte, yte, cv, grade: str = "normal", log=print) -> Resultado:
    """cv: qualquer splitter do sklearn (estratificado repetido, série temporal...).

    Erro-padrão = desvio das AUCs por fold / sqrt(nº de folds distintos). Com repetições, o divisor usa
    só as divisões independentes (conservador).
    """
    linhas, ajustados, probs = [], {}, {"y": np.asarray(yte)}
    n_splits = cv.get_n_splits(Xtr, ytr)
    n_indep = getattr(cv, "n_splits", n_splits)
    for nome, (est, g) in modelos(len(Xtr), grade).items():
        t0 = time.time()
        gs = GridSearchCV(est, g, scoring="roc_auc", cv=cv,
                          n_jobs=1 if nome in ("Random forest", "XGBoost", "LightGBM", "CatBoost") else -1,
                          refit=True)
        gs.fit(Xtr, ytr)
        i = gs.best_index_
        folds = np.array([gs.cv_results_[f"split{k}_test_score"][i] for k in range(n_splits)])
        p = gs.best_estimator_.predict_proba(Xte)[:, 1]
        probs[nome] = p
        ajustados[nome] = gs.best_estimator_
        m = summary(yte, p)
        linhas.append({
            "modelo": nome, "complexidade": COMPLEXIDADE[nome],
            "cv_auc": float(folds.mean()), "cv_ep": float(folds.std(ddof=1) / np.sqrt(n_indep)),
            "teste_auc": m["auc"], "teste_ks": m["ks"], "teste_pr_auc": m["pr_auc"], "teste_brier": m["brier"],
            "p_medio_teste": float(p.mean()),
            "params": json.dumps({k.split("__")[-1]: (v if not isinstance(v, np.generic) else v.item())
                                  for k, v in gs.best_params_.items()}),
            "segundos": round(time.time() - t0, 1),
        })
        log(f"  {nome}: cv={folds.mean():.4f}±{linhas[-1]['cv_ep']:.4f} teste={m['auc']:.4f} ({linhas[-1]['segundos']}s)")
    tab = pd.DataFrame(linhas)
    esc, melhor = escolher(tab)
    tab["escolhido"] = tab.modelo == esc
    tab["melhor_cv"] = tab.modelo == melhor
    scores = pd.DataFrame(probs).rename(columns={"y": "y"})
    return Resultado(tab, ajustados, esc, melhor, scores)


def exportar_dt_real(modelo_ajustado, cols, out: Path, prefixo: str = "") -> dict:
    """Exporta a árvore de decisão REAL que competiu (não uma versão simplificada à parte).

    `modelo_ajustado` é o pipeline (imputer, DecisionTreeClassifier) devolvido em `Resultado.ajustados`.
    Devolve a profundidade e o número de folhas, para a página explicar o que está mostrando.
    """
    dt = modelo_ajustado[-1]
    dot = export_graphviz(dt, feature_names=list(cols), class_names=["fica/não", "evento"], filled=True,
                          rounded=True, impurity=False, proportion=True, precision=2)
    (out / f"{prefixo}arvore_dt.dot").write_text(dot, encoding="utf-8")
    return {"profundidade": int(dt.get_depth()), "folhas": int(dt.get_n_leaves())}


def exportar_rf_arvores(modelo_ajustado, Xte, yte, cols, out: Path, prefixo: str = "", top_k: int = 6) -> pd.DataFrame:
    """Rankeia as árvores de um random forest já ajustado pela AUC individual de cada uma no teste,
    e exporta as `top_k` melhores para dar para "passear" por elas como se faz com o XGBoost.

    A árvore nº 0 no ranking é a melhor sozinha; o random forest usa a média de TODAS elas, não só dessa.
    """
    imputer = modelo_ajustado[0]
    rf = modelo_ajustado[-1]
    Xi = imputer.transform(Xte)
    aucs = []
    for arv in rf.estimators_:
        p = arv.predict_proba(Xi)[:, 1]
        try:
            aucs.append(roc_auc_score(yte, p))
        except ValueError:
            aucs.append(float("nan"))
    ranking = (pd.DataFrame({"arvore": range(len(aucs)), "auc_individual": aucs})
              .sort_values("auc_individual", ascending=False).reset_index(drop=True))
    ranking.to_csv(out / f"{prefixo}rf_ranking.csv", index=False)
    for pos in range(min(top_k, len(ranking))):
        idx = int(ranking.loc[pos, "arvore"])
        dot = export_graphviz(rf.estimators_[idx], feature_names=list(cols), class_names=["fica/não", "evento"],
                              filled=True, rounded=True, impurity=False, proportion=True, precision=2, max_depth=3)
        (out / f"{prefixo}rf_arvore_{pos}.dot").write_text(dot, encoding="utf-8")
    return ranking
