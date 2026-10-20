"""Cases 2 e 3 do Moiras — disputa entre logística, árvore, random forest e XGBoost.

  turnover   : IBM HR Attrition (Kaggle; base FICTÍCIA criada pela IBM) — quem sai da empresa?
  aquisicao  : Bank Marketing (UCI) — quem aceita um depósito a prazo?   credito    : Give Me Some Credit — reaproveita o pré-processamento do case 1 (só a disputa de modelos).

Uso:  python train_cases.py turnover --data data/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (RepeatedStratifiedKFold, TimeSeriesSplit,
                                     train_test_split)

from moiras.compare import SEED, SLUG, disputar, exportar_dt_real, exportar_rf_arvores
from moiras.metrics import calibration_table, decile_table, exportar_divergencias, summary

BOOSTING = ("XGBoost", "LightGBM", "CatBoost")


def exportar_shap(modelo_ajustado, Xte: pd.DataFrame, out: Path) -> float:
    """SHAP para as famílias de boosting (sem pipeline de imputação por trás) — genérico, mesmo código para
    XGBoost, LightGBM e CatBoost. Contribuição de cada variável, por caso do teste, em log-odds.

    Devolve o valor-base (log-odds médio da base) para o simulador "um caso por vez" reconstituir a probabilidade.
    """
    import shap
    explainer = shap.TreeExplainer(modelo_ajustado)
    sv = explainer(Xte)
    base = np.ravel(sv.base_values)
    base = float(base[0]) if base.size else float(sv.base_values)
    contrib = pd.DataFrame(sv.values, columns=Xte.columns, index=Xte.index)
    contrib.to_csv(out / "shap_valores.csv")
    Xte.to_csv(out / "shap_features.csv")
    (pd.DataFrame({"variavel": Xte.columns, "shap_medio_abs": np.abs(contrib).mean(axis=0).values})
     .sort_values("shap_medio_abs", ascending=False).to_csv(out / "shap_importancia.csv", index=False))
    return base

# ----------------------------------------------------------------------------- turnover
HR_PT = {
    "Age": "Idade", "BusinessTravel": "Viagens a trabalho", "DailyRate": "Diária", "Department": "Departamento",
    "DistanceFromHome": "Distância de casa", "Education": "Escolaridade (1-5)", "EducationField": "Área de formação",
    "EnvironmentSatisfaction": "Satisfação com ambiente", "Gender": "Gênero", "HourlyRate": "Valor hora",
    "JobInvolvement": "Envolvimento no trabalho", "JobLevel": "Nível do cargo", "JobRole": "Função",
    "JobSatisfaction": "Satisfação com o trabalho", "MaritalStatus": "Estado civil", "MonthlyIncome": "Salário mensal",
    "MonthlyRate": "Taxa mensal", "NumCompaniesWorked": "Empresas anteriores", "OverTime": "Hora extra",
    "PercentSalaryHike": "% último aumento", "PerformanceRating": "Avaliação de desempenho",
    "RelationshipSatisfaction": "Satisfação com relacionamentos", "StockOptionLevel": "Nível de stock options",
    "TotalWorkingYears": "Anos de carreira", "TrainingTimesLastYear": "Treinamentos no último ano",
    "WorkLifeBalance": "Equilíbrio vida-trabalho", "YearsAtCompany": "Anos na empresa",
    "YearsInCurrentRole": "Anos no cargo atual", "YearsSinceLastPromotion": "Anos desde a última promoção",
    "YearsWithCurrManager": "Anos com o gestor atual",
}


def carregar_turnover(data: Path):
    df = pd.read_csv(data / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    y = (df.pop("Attrition") == "Yes").astype(int)
    df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"])  # constantes / ID
    df = df.rename(columns=HR_PT)
    X = pd.get_dummies(df, prefix_sep=" = ", drop_first=True, dtype=int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    meta = {
        "titulo": "Turnover (Tomoyo): quem tende a sair da empresa?",
        "base": "IBM HR Analytics Employee Attrition (Kaggle) — base FICTÍCIA criada por cientistas de dados da IBM",
        "evento": "saída voluntária", "unidade": "funcionário",
        "validacao": "Split estratificado 75/25. Validação cruzada 5x3 repetida só no treino escolhe o modelo.",
        "periodo_treino": "—", "periodo_teste": "25% dos funcionários, sorteados",
        "acao": "programa de retenção", "valor_evento": 60000, "sucesso_acao": 0.30, "custo_acao": 4000,
        "texto_valor": "custo de substituir quem sai", "texto_acao": "conversa/ação de retenção",
        "notas": ["Base fictícia: serve para comparar modelos, não para tirar conclusões sobre pessoas reais.",
                  "Só 237 saídas em 1.470 funcionários: o teste tem ~59 eventos, então as métricas de teste são ruidosas.",
                  "Em People Analytics o uso correto é risco agregado por área/perfil, não decidir sobre um indivíduo."],
    }
    return Xtr, ytr, Xte, yte, cv, "normal", meta


# ----------------------------------------------------------------------------- aquisição
BANK_PT = {"age": "idade", "job": "profissão", "marital": "estado civil", "education": "escolaridade",
           "default": "inadimplente", "housing": "financ. imobiliário", "loan": "empréstimo pessoal",
           "contact": "canal", "month": "mês", "day_of_week": "dia da semana",
           "campaign": "contatos nesta campanha", "pdays": "dias desde último contato",
           "previous": "contatos anteriores", "poutcome": "resultado campanha anterior",
           "emp.var.rate": "variação do emprego (macro)", "cons.price.idx": "índice de preços (macro)",
           "cons.conf.idx": "confiança do consumidor (macro)", "euribor3m": "euribor 3 meses (macro)",
           "nr.employed": "nº de empregados (macro)"}


def carregar_aquisicao(data: Path):
    df = pd.read_csv(data / "bank-additional-full.csv", sep=";")
    y = (df.pop("y") == "yes").astype(int)
    df = df.drop(columns=["duration"])  # só se sabe depois da ligação: vazamento
    # Variáveis macro (juros, emprego...) só identificam a época da campanha, não o cliente.
    df = df.drop(columns=["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"])
    df["nunca_contatado_antes"] = (df["pdays"] == 999).astype(int)
    df["pdays"] = df["pdays"].where(df["pdays"] != 999, np.nan)
    df = df.rename(columns=BANK_PT)
    X = pd.get_dummies(df, prefix_sep=" = ", drop_first=True, dtype=int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)
    meta = {
        "titulo": "Aquisição de produto: quem aceita um depósito a prazo?",
        "base": "Bank Marketing (UCI, banco português, 2008–2010, campanhas por telefone)",
        "evento": "aceitou o depósito", "unidade": "cliente",
        "validacao": "Split estratificado 80/20. Validação cruzada 5 dobras só no treino escolhe o modelo.",
        "periodo_treino": "80% dos contatos, sorteados", "periodo_teste": "20% dos contatos, sorteados",
        "acao": "ligação de venda", "valor_evento": 100, "sucesso_acao": 1.0, "custo_acao": 20,
        "texto_valor": "margem de uma conversão", "texto_acao": "custo de uma ligação",
        "notas": ["A coluna `duration` (duração da ligação) foi descartada: só existe depois da ligação e inflaria o modelo.",
                  "As variáveis macro (euribor, emprego, confiança) também foram descartadas: identificam a época da campanha, não o cliente.",
                  "Testei validação fora do tempo (treino nos 80% mais antigos, teste nos 20% mais recentes): o desempenho despenca e varia de 0,45 a 0,75 de AUC entre os modelos, porque a taxa de conversão vai de 6% para 31% e os juros saem da faixa do treino. O split aleatório superestima o desempenho que se teria numa campanha futura.",
                  "Base de 2008–2010 em Portugal: não representa a realidade atual nem brasileira."],
    }
    return Xtr, ytr, Xte, yte, cv, "normal", meta


# ----------------------------------------------------------------------------- comum
def salvar(res, Xtr, ytr, Xte, yte, meta, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tab = res.tabela
    tab.to_csv(out / "comparacao.csv", index=False)
    res.scores.to_csv(out / "scores_teste.csv", index=False)
    esc = res.escolhido
    p = res.scores[esc]

    for nome, p_ in [(esc, p)] + [(m, res.scores[m]) for m in tab.modelo if m != esc]:
        slug = SLUG[nome]
        decile_table(yte, p_).to_csv(out / f"decis_{slug}.csv", index=False)
        calibration_table(yte, p_).to_csv(out / f"calibracao_{slug}.csv", index=False)

    # importância por permutação (vale para qualquer família de modelo) no modelo escolhido
    pi = permutation_importance(res.ajustados[esc], Xte, yte, scoring="roc_auc", n_repeats=5,
                                random_state=SEED, n_jobs=1)
    (pd.DataFrame({"variavel": Xte.columns, "queda_auc": pi.importances_mean, "desvio": pi.importances_std})
     .sort_values("queda_auc", ascending=False).to_csv(out / "importancia.csv", index=False))

    # coeficientes da logística (por desvio-padrão), sempre úteis para comparar
    lr = res.ajustados["Regressão logística"]
    coef = lr[-1].coef_[0]
    intercepto = float(lr[-1].intercept_[0])
    (pd.DataFrame({"variavel": Xte.columns, "coef_por_dp": coef, "odds_ratio": np.exp(coef)})
     .sort_values("coef_por_dp", key=np.abs, ascending=False).to_csv(out / "logistica_coef.csv", index=False))

    # contribuição de cada variável (em log-odds) para cada cliente do teste, para o simulador "um por vez"
    imputer, scaler = lr[0], lr[1]
    Xte_pad = pd.DataFrame(scaler.transform(imputer.transform(Xte)), columns=Xte.columns, index=Xte.index)
    contrib = Xte_pad * coef
    contrib.to_csv(out / "logistica_contrib.csv")
    Xte.to_csv(out / "logistica_valores.csv")
    meta["logistica_intercepto"] = intercepto

    # a árvore de decisão REAL que competiu (não uma versão simplificada à parte)
    dt_info = exportar_dt_real(res.ajustados["Árvore de decisão"], Xtr.columns, out)
    p_dt = res.ajustados["Árvore de decisão"].predict_proba(Xte)[:, 1]
    meta["arvore_dt"] = {**dt_info, **summary(yte, p_dt)}

    # random forest: rankeia as árvores internas por AUC individual e exporta as melhores para "passear"
    rf_ranking = exportar_rf_arvores(res.ajustados["Random forest"], Xte, yte, Xte.columns, out)
    meta["rf_n_arvores"] = int(len(rf_ranking))
    meta["rf_top_auc"] = float(rf_ranking.auc_individual.max())

    # SHAP do vencedor, só para as famílias de boosting (mesma lógica do case 1, genérica via TreeExplainer)
    if esc in BOOSTING:
        meta["shap_base"] = exportar_shap(res.ajustados[esc], Xte, out)

    # PSI, KL nas duas direções e Jensen-Shannon (treino x teste) + IV de cada variável e do score (no treino)
    ptr = res.ajustados[esc].predict_proba(Xtr)[:, 1]
    exportar_divergencias(Xtr, ytr, Xte, ptr, p.to_numpy(), out, "SCORE do modelo escolhido",
                          p_teste=p.to_numpy(), y_teste=yte, so_continuas=True)

    meta.update({"n_treino": len(Xtr), "n_teste": len(Xte), "taxa_treino": float(ytr.mean()),
                 "taxa_teste": float(yte.mean()), "escolhido": esc, "melhor_cv": res.melhor_cv})
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def carregar_credito(data: Path):
    """Mesma divisão e mesma limpeza do case 1; aqui só se disputa a família de modelo."""
    from moiras.credito import TARGET, fit_caps, log_features, padronizar, prepare
    raw = padronizar(pd.read_csv(data / "cs-training.csv", index_col=0))
    rtv, rte = train_test_split(raw, test_size=0.20, stratify=raw[TARGET], random_state=SEED)
    rtr, _ = train_test_split(rtv, test_size=0.25, stratify=rtv[TARGET], random_state=SEED)
    caps = fit_caps(rtr)
    Xtr, Xte = (log_features(prepare(r, caps).drop(columns=TARGET)) for r in (rtr, rte))
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=SEED)
    return Xtr, rtr[TARGET], Xte, rte[TARGET], cv, "leve", {}


CASOS = {"turnover": carregar_turnover, "aquisicao": carregar_aquisicao, "credito": carregar_credito}


def main(caso: str, data: str, out: str) -> None:
    Xtr, ytr, Xte, yte, cv, grade, meta = CASOS[caso](Path(data))
    print(f"[{caso}] treino {Xtr.shape} teste {Xte.shape} | taxa treino {ytr.mean():.3f} teste {yte.mean():.3f}")
    res = disputar(Xtr, ytr, Xte, yte, cv, grade)
    if caso == "credito":  # não sobrescreve os artefatos do case 1: usa o prefixo "cmp_" para tudo daqui
        outp = Path(out) / "credito"
        res.tabela.to_csv(outp / "comparacao.csv", index=False)
        res.scores.to_csv(outp / "cmp_scores.csv", index=False)
        for nome in SLUG:
            slug = SLUG[nome]
            decile_table(yte, res.scores[nome]).to_csv(outp / f"cmp_decis_{slug}.csv", index=False)
            calibration_table(yte, res.scores[nome]).to_csv(outp / f"cmp_calibracao_{slug}.csv", index=False)
        dt_info = exportar_dt_real(res.ajustados["Árvore de decisão"], Xtr.columns, outp, prefixo="cmp_")
        p_dt = res.ajustados["Árvore de decisão"].predict_proba(Xte)[:, 1]
        rf_ranking = exportar_rf_arvores(res.ajustados["Random forest"], Xte, yte, Xte.columns, outp, prefixo="cmp_")
        cmp_meta = {"arvore_dt": {**dt_info, **summary(yte, p_dt)},
                    "rf_n_arvores": int(len(rf_ranking)), "rf_top_auc": float(rf_ranking.auc_individual.max())}
        (outp / "cmp_meta.json").write_text(json.dumps(cmp_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        salvar(res, Xtr, ytr, Xte, yte, meta, Path(out) / caso)
    print(res.tabela[["modelo", "cv_auc", "cv_ep", "teste_auc", "teste_ks", "escolhido", "melhor_cv"]].to_string(index=False))
    print("Escolhido pela regra de 1 EP:", res.escolhido, "| melhor por AUC de CV:", res.melhor_cv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("caso", choices=list(CASOS))
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    main(a.caso, a.data, a.out)
