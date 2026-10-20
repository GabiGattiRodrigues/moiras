"""Capa do Moiras: a ideia, o placar dos cases e a porta de entrada para cada um."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from moiras.compare import escolher
from moiras.lang import L

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

h1, h2 = st.columns([2, 3], vertical_alignment="center")
with h1:
    st.image(str(ROOT / "assets" / "moiras.png"), width="stretch")
    st.caption(L(
        "As Moiras fiam, medem e cortam o fio do destino. Aqui, o destino de cada cliente ou funcionário vira uma probabilidade.",
        "The Fates spin, measure and cut the thread of destiny. Here, each customer's or employee's fate becomes a probability.",
    ))
with h2:
    st.title("Moiras")
    st.markdown(L(
        "#### Seis famílias de modelo disputam cada case com a mesma régua. **Quem ganha é o resultado, não a premissa.**",
        "#### Six model families compete on each case with the same ruler. **The result wins, not the assumption.**",
    ))

st.markdown(L(
    "Modelo de propensão costuma morrer num notebook com um AUC bonito e uma escolha feita por hábito. Aqui, para cada case, "
    "**regressão logística, árvore de decisão, random forest, XGBoost, LightGBM e CatBoost** competem na mesma validação, e a vencedora sai de uma "
    "regra definida antes de rodar: **entre os modelos a menos de 1 erro-padrão do melhor, vence o mais simples**. "
    "Depois vêm o ponto de corte por custo, a explicação e as limitações de cada base.",
    "Propensity models usually die in a notebook with a nice-looking AUC and a choice made out of habit. Here, for each "
    "case, **logistic regression, decision tree, random forest, XGBoost, LightGBM and CatBoost** compete on the same validation, and the "
    "winner comes from a rule set before running anything: **among models within 1 standard error of the best, the "
    "simplest one wins**. Then come the cost-based cutoff, the explanation, and each dataset's limitations.",
))


@st.cache_data
def placar() -> pd.DataFrame:
    linhas = []
    cred = pd.read_csv(ART / "credito" / "comparacao.csv")
    mc = json.loads((ART / "credito" / "metricas.json").read_text(encoding="utf-8"))
    esc, _ = escolher(cred)
    linhas.append(("💳 Crédito", "Give Me Some Credit (real)", "inadimplência em 2 anos", mc["taxa_evento"], cred, esc))
    for nome, pasta, base in [("👥 Turnover", "turnover", "IBM HR (fictícia)"), ("📞 Aquisição", "aquisicao", "Bank Marketing (real)")]:
        m = json.loads((ART / pasta / "meta.json").read_text(encoding="utf-8"))
        linhas.append((nome, base, m["evento"], m["taxa_treino"], pd.read_csv(ART / pasta / "comparacao.csv"), m["escolhido"]))
    rows = []
    for nome, base, ev, taxa, c, esc in linhas:
        e = c.set_index("modelo")
        rows.append({"Case": nome, "Base": base, "Vencedor": esc,
                     "Logística": e.loc["Regressão logística", "teste_auc"], "Árvore": e.loc["Árvore de decisão", "teste_auc"],
                     "Random forest": e.loc["Random forest", "teste_auc"], "XGBoost": e.loc["XGBoost", "teste_auc"]})
    return pd.DataFrame(rows)


st.subheader(L("O placar", "The scoreboard"))
pl = placar()
if st.session_state.get("lang") == "en":
    pl = pl.rename(columns={"Base": "Data", "Vencedor": "Winner", "Logística": "Logistic"})
fmt = {c: "{:.3f}" for c in (["Logistic", "Árvore", "Random forest", "XGBoost"] if st.session_state.get("lang") == "en"
                              else ["Logística", "Árvore", "Random forest", "XGBoost"])}
st.dataframe(pl.style.format(fmt), width="stretch", hide_index=True)
st.caption(L(
    "Colunas de modelo = AUC no teste (dados que nenhum modelo viu). A escolha do vencedor usa só a validação cruzada do treino; o teste confirma.",
    "Model columns = AUC on the test set (data no model saw). The winner is picked using only cross-validation on the "
    "training data; the test set only confirms it.",
))

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.page_link("credito.py", label=L("Case 1 · Crédito", "Case 1 · Credit"), icon="💳")
    st.caption(L(
        "Quem vai atrasar? Boosting ganha com folga; inclui aplicação numa fila de 101 mil propostas novas.",
        "Who's going to default? Boosting wins by a wide margin; includes applying the model to a queue of 101k new applications.",
    ))
with c2:
    st.page_link("turnover.py", label=L("Case 2 · Turnover", "Case 2 · Turnover"), icon="👥")
    st.caption(L(
        "Quem tende a sair? Base pequena: aqui a simplicidade tem a chance de vencer.",
        "Who's likely to leave? Small dataset: here simplicity gets a real chance to win.",
    ))
with c3:
    st.page_link("aquisicao.py", label=L("Case 3 · Aquisição", "Case 3 · Acquisition"), icon="📞")
    st.caption(L(
        "Quem aceita o produto? Muitas categorias e interações; inclui o teste que descartei (fora do tempo).",
        "Who takes the product? Lots of categories and interactions; includes the test I discarded (out-of-time).",
    ))
with c4:
    st.page_link("conceitos.py", label="Conceitos / Concepts", icon="📚")
    st.caption(L(
        "Não conhece AUC, KS, PSI, KL, IV, SHAP...? Glossário completo, em português e inglês, sem jargão.",
        "Don't know AUC, KS, PSI, KL, IV, SHAP...? Full glossary, in Portuguese and English, jargon-free.",
    ))

st.divider()
st.caption("Moiras · Gabriela Gatti Rodrigues · [LinkedIn](https://linkedin.com/in/gabriela-gatti-rodrigues)")
