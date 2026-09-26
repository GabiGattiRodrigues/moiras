"""Moiras — app de classificação com XGBoost (Streamlit)."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from moiras.compare import SLUG
from moiras.credito import ENTRADAS, FAIXA_DESC, RENAME, faixa, padronizar, prepare
from moiras.lang import L, lang
from moiras.metrics import profit_curve, psi

COR4 = {"Regressão logística": "#1f5fbf", "Árvore de decisão": "#e0a030", "Random forest": "#2a9d6f", "XGBoost": "#d1495b"}

ART = Path(__file__).parent / "artifacts" / "credito"
DASH_URL = "https://dashboardinteligente.streamlit.app"
AZUL, AZUL_ESC, CINZA, VERM = "#1f5fbf", "#0b2a5b", "#9aa5b1", "#d1495b"


NOMES = {
    "uso_limite": "Uso do limite rotativo", "atrasos_total": "Total de atrasos (30+)",
    "idade": "Idade", "razao_divida": "Razão dívida/renda", "linhas_credito": "Linhas de crédito abertas",
    "financ_imoveis": "Financiamentos imobiliários", "renda_mensal": "Renda mensal",
    "renda_per_capita": "Renda per capita", "atraso_30_59": "Atrasos 30–59 dias",
    "atraso_60_89": "Atrasos 60–89 dias", "atraso_90": "Atrasos 90+ dias", "dependentes": "Dependentes",
    "renda_faltante": "Renda não informada", "atraso_codigo": "Código 96/98 nos atrasos",
    "tem_atraso_grave": "Teve atraso 90+",
}
NOMES_EN = {
    "uso_limite": "Revolving credit utilization", "atrasos_total": "Total late payments (30+)",
    "idade": "Age", "razao_divida": "Debt/income ratio", "linhas_credito": "Open credit lines",
    "financ_imoveis": "Real estate loans", "renda_mensal": "Monthly income",
    "renda_per_capita": "Income per household member", "atraso_30_59": "Late payments 30–59 days",
    "atraso_60_89": "Late payments 60–89 days", "atraso_90": "Late payments 90+ days", "dependentes": "Dependents",
    "renda_faltante": "Income not reported", "atraso_codigo": "Code 96/98 in late payments",
    "tem_atraso_grave": "Had a 90+ day late payment",
}


def nome(v: str) -> str:
    return (NOMES_EN if lang() == "en" else NOMES).get(v, v)


@st.cache_data
def carregar():
    r = lambda n: pd.read_csv(ART / n)
    import json
    d = {
        "m": json.loads((ART / "metricas.json").read_text(encoding="utf-8")),
        "scores": r("scores_teste.csv"), "dec": r("decis_xgb.csv"), "dec_lr": r("decis_lr.csv"),
        "cal": r("calibracao_xgb.csv"), "cal_lr": r("calibracao_lr.csv"),
        "psi": r("psi.csv"), "imp": r("shap_importancia.csv"),
        "shap": pd.read_csv(ART / "shap_valores.csv", index_col=0),
        "feat": pd.read_csv(ART / "shap_features.csv", index_col=0),
    }
    if (ART / "comparacao.csv").exists():
        d["cmp"] = r("comparacao.csv")
        d["cmp_scores"] = r("cmp_scores.csv")
        d["cmp_dec"] = {m: r(f"cmp_decis_{s}.csv") for m, s in SLUG.items()}
        d["cmp_cal"] = {m: r(f"cmp_calibracao_{s}.csv") for m, s in SLUG.items()}
        d["cmp_meta"] = json.loads((ART / "cmp_meta.json").read_text(encoding="utf-8"))
        d["cmp_dt_dot"] = (ART / "cmp_arvore_dt.dot").read_text(encoding="utf-8")
        d["cmp_rf_ranking"] = r("cmp_rf_ranking.csv")
        d["cmp_rf_dots"] = [(ART / f"cmp_rf_arvore_{i}.dot").read_text(encoding="utf-8")
                            for i in range(6) if (ART / f"cmp_rf_arvore_{i}.dot").exists()]
    return d


@st.cache_resource
def carregar_modelo():
    m = xgb.XGBClassifier()
    m.load_model(str(ART / "xgb.json"))
    return m


@st.cache_data
def tabela_arvores() -> pd.DataFrame:
    return carregar_modelo().get_booster().trees_to_dataframe()


def arvores_ganho(n_usadas: int) -> pd.Series:
    """Ganho total de cada árvore usada na predição (early stopping corta o resto)."""
    t = tabela_arvores()
    t = t[(t.Feature != "Leaf") & (t.Tree < n_usadas)]
    return t.groupby("Tree").Gain.sum()


def dot_arvore(n: int) -> str:
    t = tabela_arvores()
    t = t[t.Tree == n]
    cliente_lbl = "clients" if lang() == "en" else "clientes"
    sim_lbl, nao_lbl = ("yes", "no") if lang() == "en" else ("sim", "não")
    linhas = ['digraph {', 'rankdir=LR; nodesep=0.25; ranksep=0.6; bgcolor="transparent";',
              'node [fontname="Helvetica", fontsize=13, style="filled,rounded", shape=box];',
              'edge [fontname="Helvetica", fontsize=11, color="#6b7785"];']
    for r in t.itertuples():
        if r.Feature == "Leaf":
            cor = "#f6c9cf" if r.Gain > 0 else "#c9dcf6"
            cover = f"{r.Cover:,.0f}".replace(",", ".")
            linhas.append(f'"{r.ID}" [label="{r.Gain:+.2f}\\n~{cover} {cliente_lbl}", fillcolor="{cor}", '
                          'shape=ellipse, color="#9aa5b1"];')
        else:
            nm = nome(r.Feature)
            linhas.append(f'"{r.ID}" [label="{nm}\\n< {r.Split:.4g} ?", fillcolor="#eef3fb", color="#1f5fbf"];')
            linhas.append(f'"{r.ID}" -> "{r.Yes}" [label="{sim_lbl}"];')
            linhas.append(f'"{r.ID}" -> "{r.No}" [label="{nao_lbl}"];')
    linhas.append("}")
    return "\n".join(linhas)


@st.cache_data
def base_nova():
    """Fila de propostas novas (base de teste do Kaggle, sem rótulo), pontuada pelo modelo já treinado."""
    X = pd.read_csv(ART / "base_nova.csv.gz", index_col=0)
    p = carregar_modelo().predict_proba(X)[:, 1]
    return X, pd.Series(p, index=X.index, name="p")


def contribuicoes(row: pd.DataFrame) -> pd.Series:
    """Contribuição de cada variável (log-odds) para um cliente, direto do XGBoost."""
    c = carregar_modelo().get_booster().predict(xgb.DMatrix(row), pred_contribs=True)[0]
    return pd.Series(c[:-1], index=row.columns)


@st.cache_data
def caps():
    import json
    return json.loads((ART / "caps.json").read_text(encoding="utf-8"))


def pontuar(raw: pd.DataFrame) -> pd.DataFrame:
    X = prepare(padronizar(raw), caps())
    cols = carregar_modelo().get_booster().feature_names
    p = carregar_modelo().predict_proba(X[cols])[:, 1]
    return pd.DataFrame({"prob_inadimplencia": p, "faixa": faixa(pd.Series(p, index=X.index)).astype(str)}, index=X.index)


D = carregar()
M = D["m"]

# ------------------------------------------------------------------ capa
st.title(L("Case 1 · Risco de crédito", "Case 1 · Credit risk"))
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(L(
        "Quem vai atrasar? Dados reais do [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) "
        f"(150 mil clientes, {M['taxa_evento']:.1%} de inadimplência em 2 anos). O caminho completo: "
        "**baseline → XGBoost → calibração → ponto de corte por custo → explicação → aplicação em base nova → monitoramento**.",
        "Who's going to default? Real data from [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) "
        f"(150k customers, {M['taxa_evento']:.1%} default rate over 2 years). The full path: "
        "**baseline → XGBoost → calibration → cost-based cutoff → explanation → scoring a new dataset → monitoring**.",
    ))
with c2:
    st.link_button(L("Ver o Dashboard Inteligente ↗", "See the Intelligent Dashboard ↗"), DASH_URL, width="stretch")
st.page_link("conceitos.py", label=L("📚 Não conhece os termos abaixo (KS, AUC, Brier...)? Veja o glossário",
                                      "📚 Don't know the terms below (KS, AUC, Brier...)? See the glossary"), icon="📚")

x, l = M["xgboost"], M["logistica"]
k = st.columns(4)
k[0].metric(L("KS (teste)", "KS (test)"), f"{x['ks']:.3f}", f"{x['ks'] - l['ks']:+.3f} " + L("vs logística", "vs logistic"),
            help=L("Mede o quanto o modelo separa bons de maus pagadores. 0 = não separa nada, 100% = separação "
                   "perfeita. Acima de 30–40% já é considerado forte no mercado de crédito. Veja a aba Conceitos.",
                   "Measures how well the model separates good from bad payers. 0 = separates nothing, 100% = "
                   "perfect separation. Above 30–40% is already considered strong in the credit industry. See the "
                   "Concepts tab."))
k[1].metric(L("AUC (teste)", "AUC (test)"), f"{x['auc']:.3f}", f"{x['auc'] - l['auc']:+.3f} " + L("vs logística", "vs logistic"),
            help=L("Chance de, pegando um cliente que deu calote e um que não deu, o modelo apontar o risco maior "
                   "para quem deu calote. 50% = está chutando; 100% = acerta sempre. Veja a aba Conceitos.",
                   "Chance that, picking one customer who defaulted and one who didn't, the model gives a higher "
                   "risk to the one who defaulted. 50% = guessing; 100% = always right. See the Concepts tab."))
top2 = D["dec"].loc[D["dec"].decil <= 2]
k[2].metric(L("Maus pagadores nos 20% de maior risco", "Bad payers in the top 20% riskiest"),
            f"{top2.eventos.sum() / D['dec'].eventos.sum():.0%}",
            help=L("Dos 20% de clientes que o modelo aponta como mais arriscados (2 primeiros decis), que fração de "
                   "todos os calotes da base está concentrada ali. Quanto mais perto de 100%, melhor o modelo separa.",
                   "Of the 20% of customers the model flags as riskiest (top 2 deciles), what share of all defaults "
                   "in the data is concentrated there. The closer to 100%, the better the model separates."))
k[3].metric(L("Brier (menor é melhor)", "Brier (lower is better)"), f"{x['brier']:.4f}", f"{x['brier'] - l['brier']:+.4f}",
            delta_color="inverse",
            help=L("Mede se a probabilidade em si está certa (não só a ordem): 0 é perfeito, quanto menor melhor. "
                   "Complementa a AUC/KS, que só olham a ordem dos riscos. Veja a aba Conceitos.",
                   "Measures whether the probability itself is right (not just the ranking): 0 is perfect, lower "
                   "is better. Complements AUC/KS, which only look at the risk ranking. See the Concepts tab."))
st.caption(L(
    f"Métricas em {M['n_teste']:,} clientes que o modelo nunca viu. Treino {M['n_treino']:,} · validação "
    f"{M['n_validacao']:,} (early stopping) · teste {M['n_teste']:,}.".replace(",", "."),
    f"Metrics on {M['n_teste']:,} customers the model never saw. Train {M['n_treino']:,} · validation "
    f"{M['n_validacao']:,} (early stopping) · test {M['n_teste']:,}.",
))

abas = (["1 · Model", "2 · Decision", "3 · Explanation", "4 · New data", "5 · Monitoring", "6 · Limitations"]
        if lang() == "en" else
        ["1 · Modelo", "2 · Decisão", "3 · Explicação", "4 · Base nova", "5 · Monitoramento", "6 · Limites"])
t1, t2, t3, t4, t5, t6 = st.tabs(abas)

# ------------------------------------------------------------------ 1 modelo
with t1:
    st.subheader(L("Quatro famílias de modelo, uma disputa só", "Four model families, one single competition"))
    st.markdown(L(
        "**Como ler esta página, em miúdos:** regressão logística, árvore de decisão, random forest e XGBoost tentaram, cada "
        "um do seu jeito, prever a chance de um cliente não pagar. Testamos cada um várias vezes, embaralhando quem entra "
        "no treino, e medimos o quanto cada um acerta com a **AUC** (uma nota de 50%, que é chutar cara ou coroa, a 100%, "
        "que é acertar sempre). A nota vem com uma margem de erro (o **± EP**); duas notas que se sobrepõem dentro dessa "
        "margem são, no fundo, um empate.",
        "**How to read this page, plainly:** logistic regression, decision tree, random forest and XGBoost each tried, "
        "in their own way, to predict the chance a customer won't pay. We tested each one several times, reshuffling "
        "who's in the training fold, and measured how well each one does with **AUC** (a grade from 50%, a coin flip, "
        "to 100%, always right). That grade comes with a margin of error (the **± SE**); two grades that overlap "
        "within that margin are, in practice, a tie.",
    ))
    st.markdown(L(
        "**A regra de escolha, definida antes de rodar:** entre os modelos empatados com o melhor (dentro de 1 erro-padrão), "
        "vence o **mais simples** (logística < árvore < random forest < XGBoost) — um modelo mais simples é mais fácil de "
        "explicar e manter, então só vale trocar por um mais complexo se ele realmente entregar mais.",
        "**The selection rule, set before running anything:** among models tied with the best one (within 1 standard "
        "error), the **simplest** wins (logistic < decision tree < random forest < XGBoost) — a simpler model is "
        "easier to explain and maintain, so it's only worth switching to a more complex one if it truly delivers more.",
    ))
    if "cmp" in D:
        ct = D["cmp"]
        e4 = ct.set_index("modelo")
        esc4 = ct.loc[ct.escolhido, "modelo"].iloc[0]
        melhor4 = ct.loc[ct.melhor_cv, "modelo"].iloc[0]
        esc4_l, melhor4_l = (L(esc4, esc4), L(melhor4, melhor4))  # model names kept as-is (proper nouns)
        if esc4 == melhor4:
            st.success(L(
                f"**Por que {esc4} venceu:** teve a melhor nota de validação ({e4.loc[esc4, 'cv_auc']:.3f} de AUC), "
                "e nenhum modelo mais simples chegou perto o bastante para tomar o lugar pela regra. Venceu limpo.",
                f"**Why {esc4} won:** it had the best validation score ({e4.loc[esc4, 'cv_auc']:.3f} AUC), and no "
                "simpler model came close enough to take its place under the rule. A clean win.",
            ))
        else:
            st.info(L(
                f"**Por que {esc4} venceu, e não {melhor4}:** {melhor4} teve a nota mais alta na validação "
                f"({e4.loc[melhor4, 'cv_auc']:.3f}), mas {esc4} chegou perto o suficiente ({e4.loc[esc4, 'cv_auc']:.3f}, "
                f"dentro de 1 erro-padrão) e é mais simples. Pela regra, o empate técnico vai para o mais simples.",
                f"**Why {esc4} won, and not {melhor4}:** {melhor4} had the highest validation score "
                f"({e4.loc[melhor4, 'cv_auc']:.3f}), but {esc4} came close enough ({e4.loc[esc4, 'cv_auc']:.3f}, within "
                "1 standard error) and is simpler. Under the rule, a technical tie goes to the simpler model.",
            ))
        show = ct[["modelo", "cv_auc", "cv_ep", "teste_auc", "teste_ks", "teste_brier", "escolhido"]].copy()
        show["escolhido"] = np.where(show.escolhido, "✓", "")
        if lang() == "en":
            show["modelo"] = show["modelo"].replace({"Regressão logística": "Logistic regression",
                                                       "Árvore de decisão": "Decision tree"})
            show.columns = ["Model", "Validation AUC", "± SE", "Test AUC", "Test KS", "Test Brier", "Chosen"]
        else:
            show.columns = ["Modelo", "AUC validação", "± EP", "AUC teste", "KS teste", "Brier teste", "Escolhido"]
        st.dataframe(show.style.format({show.columns[1]: "{:.4f}", show.columns[2]: "{:.4f}", show.columns[3]: "{:.4f}",
                                        show.columns[4]: "{:.3f}", show.columns[5]: "{:.4f}"}), width="stretch", hide_index=True)
        st.caption(L(
            "AUC de validação: busca em grade com validação cruzada de 3 dobras, só no treino (o que decide a escolha). "
            "AUC/KS/Brier de teste: nos 30 mil clientes que nenhum modelo usou para se ajustar (só confirma, não decide). "
            "O XGBoost do resto desta página usa early stopping em vez da grade, por isso os números de teste da aba variam um pouco.",
            "Validation AUC: grid search with 3-fold cross-validation, only on the training data (this decides the "
            "choice). Test AUC/KS/Brier: on the 30k customers no model used to fit itself (only confirms, doesn't "
            "decide). The XGBoost used in the rest of this page uses early stopping instead of the grid, so the "
            "test numbers on this tab vary slightly.",
        ))
        a, b = st.columns(2)
        with a:
            st.markdown(L("**Lift por decil de risco** (1 = maior risco previsto)", "**Lift by risk decile** (1 = highest predicted risk)"))
            f = go.Figure()
            for m in ct.modelo:
                f.add_scatter(x=D["cmp_dec"][m].decil, y=D["cmp_dec"][m].lift, mode="lines+markers",
                             name=L(m, {"Regressão logística": "Logistic regression", "Árvore de decisão": "Decision tree"}.get(m, m)),
                             line_color=COR4[m], line_width=4 if m == esc4 else 1.5)
            f.add_hline(y=1, line_dash="dot", line_color="black")
            f.update_layout(height=340, margin=dict(t=10), xaxis_title=L("Decil", "Decile"), yaxis_title="Lift")
            st.plotly_chart(f, width="stretch")
        with b:
            st.markdown(L("**Calibração**: probabilidade prevista × taxa real", "**Calibration**: predicted probability × real rate"))
            f = go.Figure()
            mx = max(max(D["cmp_cal"][m].p_medio.max(), D["cmp_cal"][m].taxa_real.max()) for m in ct.modelo)
            f.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dot", color="black"), name=L("Perfeita", "Perfect"))
            for m in ct.modelo:
                f.add_scatter(x=D["cmp_cal"][m].p_medio, y=D["cmp_cal"][m].taxa_real, mode="lines+markers",
                             name=L(m, {"Regressão logística": "Logistic regression", "Árvore de decisão": "Decision tree"}.get(m, m)),
                             line_color=COR4[m], line_width=4 if m == esc4 else 1.5)
            f.update_layout(height=340, margin=dict(t=10), xaxis_title=L("Prob. média prevista", "Mean predicted prob."),
                            yaxis_title=L("Taxa real", "Real rate"))
            st.plotly_chart(f, width="stretch")
    st.markdown(L(
        "**Sem `scale_pos_weight` de propósito.** Reponderar a classe rara melhora recall, mas distorce a probabilidade; "
        "como aqui a probabilidade vira preço e limite, mantive o modelo calibrado.",
        "**No `scale_pos_weight`, on purpose.** Reweighting the rare class improves recall but distorts the "
        "probability; since the probability here becomes price and limit, I kept the model calibrated.",
    ))
    with st.expander(L("Tabela de decis (XGBoost, versão com early stopping usada no resto da página)",
                        "Decile table (XGBoost, early-stopping version used in the rest of the page)")):
        st.dataframe(D["dec"].style.format({"taxa_evento": "{:.1%}", "p_medio": "{:.1%}", "lift": "{:.2f}", "ganho_acum": "{:.1%}"}),
                     width="stretch", hide_index=True)

# ------------------------------------------------------------------ 2 decisão
with t2:
    st.subheader(L("Onde cortar? Depende de quanto custa errar", "Where to cut? It depends on how much a mistake costs"))
    st.caption(L("⚠️ Valores abaixo são **hipóteses ilustrativas ajustáveis**, não dados reais de nenhuma instituição.",
                 "⚠️ Values below are **illustrative, adjustable assumptions**, not real data from any institution."))
    st.markdown(L(
        "O modelo só entrega uma probabilidade de calote; para virar uma decisão (aprovar ou negar) alguém precisa "
        "dizer **quanto se ganha quando o cliente paga** e **quanto se perde quando ele não paga**. É isso que os "
        "dois controles abaixo representam — mude os valores e veja o ponto de corte (linha pontilhada no gráfico) andar.",
        "The model only delivers a default probability; turning it into a decision (approve or deny) requires "
        "someone to say **how much you gain when the customer pays** and **how much you lose when they don't**. "
        "That's what the two controls below represent — change the values and watch the cutoff (dotted line on the "
        "chart) move.",
    ))
    a, b = st.columns(2)
    ganho = a.slider(L("Margem ao aprovar um bom pagador (R$)", "Margin when approving a good payer ($)"), 100, 3000, 600, 50,
                      help=L("Quanto a empresa ganha, em média, quando aprova um cliente que paga em dia (juros, "
                             "tarifas, relacionamento). Quanto maior esse número, mais vale a pena arriscar aprovar "
                             "gente na dúvida.",
                             "How much the company earns, on average, when it approves a customer who pays on time "
                             "(interest, fees, relationship). The higher this number, the more it pays off to take "
                             "a risk on borderline customers."))
    perda = b.slider(L("Prejuízo ao aprovar um mau pagador (R$)", "Loss when approving a bad payer ($)"), 500, 20000, 5000, 250,
                      help=L("Quanto a empresa perde, em média, quando aprova um cliente que não paga (o valor "
                             "emprestado e não devolvido, cobrança, etc.). Quanto maior esse número, mais rigoroso "
                             "o corte precisa ser para compensar o risco.",
                             "How much the company loses, on average, when it approves a customer who doesn't pay "
                             "(the money lent and not returned, collections, etc.). The higher this number, the "
                             "stricter the cutoff needs to be to offset the risk."))
    pc = profit_curve(D["scores"].y, D["scores"].p_xgb, ganho, perda)
    pc_lr = profit_curve(D["scores"].y, D["scores"].p_lr, ganho, perda)
    best, best_lr = pc.loc[pc.lucro.idxmax()], pc_lr.loc[pc_lr.lucro.idxmax()]
    f = go.Figure()
    f.add_scatter(x=pc.corte, y=pc.lucro, name="XGBoost", line_color=AZUL)
    f.add_scatter(x=pc_lr.corte, y=pc_lr.lucro, name=L("Logística", "Logistic"), line_color=CINZA)
    f.add_vline(x=best.corte, line_dash="dot", line_color=AZUL)
    f.update_layout(height=340, margin=dict(t=10), xaxis_title=L("Aprovar quem tem prob. de inadimplência < corte",
                                                                  "Approve whoever has default prob. < cutoff"),
                    yaxis_title=L("Lucro esperado no teste (R$)", "Expected profit on test set ($)"), xaxis_tickformat=".0%")
    st.plotly_chart(f, width="stretch")
    st.caption(L(
        "**Como ler este gráfico:** o eixo de baixo é o ponto de corte (aprovar todo mundo com probabilidade de "
        "calote abaixo dele); o eixo da esquerda é o lucro esperado somando todos os clientes do teste com aquele "
        "corte. A linha sobe enquanto aprovar mais gente ainda vale a pena, atinge um pico (o corte ótimo, linha "
        "pontilhada) e depois desce, porque aprovar clientes cada vez mais arriscados passa a dar prejuízo.",
        "**How to read this chart:** the bottom axis is the cutoff (approve everyone with default probability below "
        "it); the left axis is the expected profit summed over all test customers at that cutoff. The line rises "
        "while approving more people still pays off, peaks (the optimal cutoff, dotted line), then falls, because "
        "approving increasingly risky customers starts to lose money.",
    ))
    k = st.columns(4)
    k[0].metric(L("Corte ótimo (XGBoost)", "Optimal cutoff (XGBoost)"), f"{best.corte:.1%}",
                help=L("O ponto de corte que gera o maior lucro esperado nos dados de teste, dadas as hipóteses de "
                       "margem e prejuízo ao lado. Veja a explicação completa logo abaixo.",
                       "The cutoff that generates the highest expected profit on the test data, given the margin "
                       "and loss assumptions above. See the full explanation right below."))
    k[1].metric(L("Aprovação", "Approval rate"), f"{best.aprovacao:.0%}",
                help=L("Que fração de todos os clientes do teste seria aprovada usando esse corte ótimo.",
                       "What share of all test customers would be approved using this optimal cutoff."))
    k[2].metric(L("Inadimplência dos aprovados", "Default rate among approved"), f"{best.inadimplencia_aprovados:.1%}",
                help=L("Entre os clientes aprovados por esse corte, que fração realmente daria calote — não é zero "
                       "de propósito: um pouco de risco compensa, porque a margem de quem paga cobre o prejuízo.",
                       "Among customers approved by this cutoff, what share would actually default — it's not zero "
                       "on purpose: a bit of risk pays off, because the margin from those who pay covers the loss."))
    k[3].metric(L("Lucro vs logística no corte ótimo", "Profit vs logistic at the optimal cutoff"),
                f"$ {best.lucro - best_lr.lucro:,.0f}" if lang() == "en" else f"R$ {best.lucro - best_lr.lucro:,.0f}".replace(",", "."),
                help=L("Quanto a mais (ou a menos) de lucro esperado o XGBoost entrega, no melhor corte de cada "
                       "modelo, comparado com a regressão logística.",
                       "How much more (or less) expected profit XGBoost delivers, at each model's own best cutoff, "
                       "compared with logistic regression."))
    st.markdown(L(
        f"**O que esses números querem dizer:** o modelo dá uma probabilidade de inadimplência para cada cliente. "
        f"\"Corte ótimo de {best.corte:.1%}\" significa **aprovar todo mundo cuja probabilidade prevista de não pagar "
        f"seja menor que {best.corte:.1%}, e recusar o resto**. Com as hipóteses acima, isso aprova {best.aprovacao:.0%} "
        f"da fila (os mais seguros) e recusa os outros {1 - best.aprovacao:.0%} (os mais arriscados); entre os aprovados, "
        f"{best.inadimplencia_aprovados:.1%} ainda vão dar calote — um risco que compensa, porque a margem dos que pagam "
        "cobre esse prejuízo. Passar desse ponto e aprovar mais gente já dá prejuízo, porque a probabilidade de dar calote "
        "vezes o valor do prejuízo passa a valer mais do que a margem que se ganha. Com essas hipóteses, isso acontece perto "
        f"de **{ganho / (ganho + perda):.1%}** de probabilidade — mexa nos sliders acima e veja o corte andar.",
        f"**What these numbers mean:** the model gives each customer a default probability. \"Optimal cutoff of "
        f"{best.corte:.1%}\" means **approve everyone whose predicted probability of not paying is below "
        f"{best.corte:.1%}, and reject the rest**. With the assumptions above, this approves {best.aprovacao:.0%} of "
        f"the queue (the safest ones) and rejects the other {1 - best.aprovacao:.0%} (the riskiest); among the "
        f"approved, {best.inadimplencia_aprovados:.1%} will still default — a risk that pays off, because the "
        "margin from those who pay covers that loss. Go past this point and approve more people, and it already "
        "loses money, because the default probability times the loss amount outweighs the margin earned. With "
        f"these assumptions, that happens near **{ganho / (ganho + perda):.1%}** probability — move the sliders "
        "above and watch the cutoff shift.",
    ))

# ------------------------------------------------------------------ 3 explicação
with t3:
    st.subheader(L("Por que o modelo decidiu isso?", "Why did the model decide this?"))
    a, b = st.columns([2, 3])
    with a:
        st.markdown(L("**Importância global** (|SHAP| médio)", "**Global importance** (mean |SHAP|)"))
        imp = D["imp"].head(10).iloc[::-1].copy()
        imp["nome"] = imp.variavel.map(nome)
        f = px.bar(imp, x="shap_medio_abs", y="nome", orientation="h", color_discrete_sequence=[AZUL])
        f.update_layout(height=380, margin=dict(t=10), xaxis_title=L("Impacto médio no log-odds", "Mean impact on log-odds"), yaxis_title="")
        st.plotly_chart(f, width="stretch")
    with b:
        st.markdown(L("**Um cliente por vez**", "**One customer at a time**"))
        probs = 1 / (1 + np.exp(-(D["m"]["shap_base"] + D["shap"].sum(axis=1))))
        opcoes = ["High risk", "Medium risk", "Low risk", "Random"] if lang() == "en" else ["Alto risco", "Risco médio", "Baixo risco", "Aleatório"]
        modo = st.radio(L("Escolha o perfil", "Choose the profile"), opcoes, horizontal=True)
        chave = {"High risk": "Alto risco", "Medium risk": "Risco médio", "Low risk": "Baixo risco", "Random": "Aleatório"}.get(modo, modo)
        pool = {"Alto risco": probs[probs > 0.35], "Risco médio": probs[(probs > 0.05) & (probs < 0.10)],
                "Baixo risco": probs[probs < 0.01], "Aleatório": probs}[chave]
        if "seed" not in st.session_state:
            st.session_state.seed = 0
        if st.button(L("Sortear outro cliente", "Draw another customer")):
            st.session_state.seed += 1
        idx = pool.sample(1, random_state=st.session_state.seed).index[0]
        contrib = D["shap"].loc[idx]
        vals = D["feat"].loc[idx]
        tb = pd.DataFrame({"variavel": contrib.index, "shap": contrib.values, "valor": vals.values})
        tb = tb.reindex(tb.shap.abs().sort_values(ascending=False).index).head(8).iloc[::-1]
        nd = "n/a" if lang() == "en" else "n/d"
        tb["rotulo"] = tb.apply(lambda r: f"{nome(r.variavel)} = {r.valor:g}" if pd.notna(r.valor) else f"{nome(r.variavel)} = {nd}", axis=1)
        f = go.Figure(go.Bar(x=tb.shap, y=tb.rotulo, orientation="h",
                             marker_color=[VERM if v > 0 else AZUL for v in tb.shap]))
        f.update_layout(height=380, margin=dict(t=10),
                        xaxis_title=L("Contribuição no log-odds (vermelho ↑ risco, azul ↓ risco)",
                                      "Contribution to log-odds (red ↑ risk, blue ↓ risk)"), yaxis_title="")
        st.plotly_chart(f, width="stretch")
        st.markdown(L(f"Probabilidade de inadimplência prevista: **{probs[idx]:.1%}** "
                    f"(média da base: {D['m']['taxa_evento']:.1%}).",
                    f"Predicted default probability: **{probs[idx]:.1%}** "
                    f"(base average: {D['m']['taxa_evento']:.1%})."))
    st.caption(L("O SHAP soma as contribuições em log-odds a partir do risco médio da base; a soma vira a probabilidade acima. "
               "Número no gráfico e número do texto são o mesmo. Não conhece o SHAP? Veja a aba **Conceitos** no menu.",
               "SHAP adds up log-odds contributions starting from the base's average risk; that sum becomes the "
               "probability above. The chart number and the text number are the same. Don't know SHAP? See the "
               "**Concepts** tab in the menu."))

    st.divider()
    st.subheader(L("Por dentro do modelo: uma das árvores", "Inside the model: one of the trees"))
    n_usadas = int(M["params_xgb"]["n_arvores"])
    ganho_arv = arvores_ganho(n_usadas)
    st.markdown(L(
        f"O XGBoost é uma **soma de {n_usadas} árvores pequenas**: cada uma corrige o erro das anteriores. "
        "Nenhuma árvore sozinha decide; o risco final é a soma das folhas por onde o cliente cai, passada pela "
        "função logística. Abaixo, a árvore que mais contribui para o modelo (maior ganho total nas divisões), "
        "e você pode passear pelas outras.",
        f"XGBoost is a **sum of {n_usadas} small trees**: each one corrects the previous ones' mistakes. No single "
        "tree decides on its own; the final risk is the sum of the leaves the customer lands on, passed through "
        "the logistic function. Below, the tree that contributes the most to the model (highest total gain across "
        "its splits) — you can browse the others too.",
    ))
    melhor = int(ganho_arv.idxmax())
    ntree = st.slider(L("Árvore nº", "Tree #"), 0, n_usadas - 1, melhor)
    st.graphviz_chart(dot_arvore(ntree), width="stretch")
    st.caption(L(
        f"Árvore {ntree}" + (" (a de maior ganho)" if ntree == melhor else "") +
        ". Cada caixa é uma pergunta sobre o cliente: siga a seta **sim** ou **não** "
        "(dado ausente segue o caminho que o treino achou melhor). As folhas são valores em log-odds: "
        "vermelho empurra o risco para cima, azul para baixo. O valor de uma folha isolada não é uma probabilidade.",
        f"Tree {ntree}" + (" (the highest-gain one)" if ntree == melhor else "") +
        ". Each box is a yes/no question about the customer: follow the **yes** or **no** arrow (a missing value "
        "follows whichever path training found best). Leaves are log-odds values: red pushes risk up, blue pushes "
        "it down. A single leaf's value is not, by itself, a probability.",
    ))

    if "cmp_meta" in D:
        cm = D["cmp_meta"]
        cauc = D["cmp"].set_index("modelo")
        st.divider()
        st.markdown(L(
            f"**Como comparação: a árvore de decisão sozinha** (profundidade {cm['arvore_dt']['profundidade']}, "
            f"{cm['arvore_dt']['folhas']} folhas) — a mesma que competiu na aba Modelo, não uma versão simplificada",
            f"**As a comparison: the decision tree on its own** (depth {cm['arvore_dt']['profundidade']}, "
            f"{cm['arvore_dt']['folhas']} leaves) — the same one that competed in the Model tab, not a simplified version",
        ))
        st.caption(L("Lê-se de cima para baixo: em cada caixa, siga o sim/não até uma folha (oval). Cor mais forte = maior "
                   "concentração de inadimplência.",
                   "Read top to bottom: at each box, follow yes/no until a leaf (oval). Stronger color = higher "
                   "concentration of default."))
        st.graphviz_chart(D["cmp_dt_dot"], width="stretch")
        st.caption(L(f"Sozinha, essa árvore tem AUC {cm['arvore_dt']['auc']:.3f} no teste, contra "
                   f"{cauc.loc['XGBoost', 'teste_auc']:.3f} do XGBoost — dá para ver o padrão principal, mas perde nuance.",
                   f"On its own, this tree has AUC {cm['arvore_dt']['auc']:.3f} on the test set, against "
                   f"{cauc.loc['XGBoost', 'teste_auc']:.3f} for XGBoost — you can see the main pattern, but it "
                   "loses nuance."))

        st.markdown(L(f"**E o random forest: um bando de {cm['rf_n_arvores']} árvores, cada uma vota**",
                    f"**And the random forest: a pack of {cm['rf_n_arvores']} trees, each one votes**"))
        st.caption(L("Nenhuma árvore do random forest decide sozinha: cada uma vê uma parte diferente dos dados e dá seu "
                   "palpite; a resposta final é a média de todas. Abaixo, passeie pelas que sozinhas foram melhores no teste.",
                   "No single random forest tree decides on its own: each one sees a different slice of the data "
                   "and gives its own guess; the final answer is the average of all of them. Below, browse the "
                   "ones that scored best on their own on the test set."))
        posr = st.slider(L("Árvore do random forest (nº 0 = a melhor sozinha)", "Random forest tree (# 0 = best alone)"),
                          0, len(D["cmp_rf_dots"]) - 1, 0, key="rf_credito")
        st.graphviz_chart(D["cmp_rf_dots"][posr], width="stretch")
        auc_r = float(D["cmp_rf_ranking"].loc[posr, "auc_individual"])
        st.caption(L(f"Essa árvore sozinha (só os 3 primeiros níveis, para caber na tela) tem AUC {auc_r:.3f} no teste. "
                   f"O random forest inteiro, com a média de todas as {cm['rf_n_arvores']}, chega a "
                   f"{cauc.loc['Random forest', 'teste_auc']:.3f} — o bando acerta mais porque os erros de cada árvore, "
                   "em geral, não se repetem nas outras.",
                   f"This tree alone (only the first 3 levels, to fit the screen) has AUC {auc_r:.3f} on the test "
                   f"set. The whole random forest, averaging all {cm['rf_n_arvores']}, reaches "
                   f"{cauc.loc['Random forest', 'teste_auc']:.3f} — the pack does better because each tree's "
                   "mistakes generally don't repeat across the others."))

# ------------------------------------------------------------------ 4 base nova
with t4:
    st.subheader(L("Na vida real: uma fila de propostas que o modelo nunca viu", "In real life: a queue of applications the model never saw"))
    Xn, pn = base_nova()
    st.markdown(L(
        f"O modelo treinado foi aplicado, sem retreino, a **{len(Xn):,} clientes novos** (a base de teste do Kaggle), "
        "que **não têm rótulo**: como numa esteira de crédito, ninguém sabe ainda quem vai atrasar. "
        "O resultado só aparece daqui a 2 anos, e é por isso que faixa de risco, corte e monitoramento importam."
        .replace(",", "."),
        f"The trained model was applied, with no retraining, to **{len(Xn):,} new customers** (Kaggle's test set), "
        "which **have no label**: just like a real credit pipeline, nobody knows yet who's going to default. "
        "The outcome only shows up 2 years from now, which is exactly why risk band, cutoff and monitoring matter.",
    ))
    fn = faixa(pn).astype(str)
    sc = D["scores"].copy()
    sc["faixa"] = faixa(sc.p_xgb).astype(str)
    col1n = L("Prob. prevista", "Predicted prob.")
    tb = pd.DataFrame({col1n: FAIXA_DESC})
    col2n, col3n, col4n = (L("% da fila nova", "% of new queue"), L("% da base de teste", "% of test set"),
                            L("Inadimplência real (teste rotulado)", "Real default rate (labeled test set)"))
    tb[col2n] = fn.value_counts(normalize=True)
    tb[col3n] = sc.faixa.value_counts(normalize=True)
    tb[col4n] = sc.groupby("faixa").y.mean()
    tb = tb.loc[list(FAIXA_DESC)]
    tb.index.name = L("Faixa", "Band")
    a, b = st.columns([3, 2])
    with a:
        st.dataframe(tb.style.format({c: "{:.1%}" for c in tb.columns[1:]}), width="stretch")
    with b:
        f = go.Figure()
        f.add_bar(x=tb.index, y=tb[col2n], name=L("Fila nova", "New queue"), marker_color=AZUL)
        f.add_bar(x=tb.index, y=tb[col3n], name=L("Teste rotulado", "Labeled test set"), marker_color=CINZA)
        f.update_layout(barmode="group", height=250, margin=dict(t=10, b=10), yaxis_tickformat=".0%",
                        yaxis_title=L("% dos clientes", "% of customers"))
        st.plotly_chart(f, width="stretch")
    st.caption(L("A distribuição da fila nova por faixa é praticamente a do teste, então o modelo está sendo aplicado a uma população parecida "
               "com a do treino. A coluna de inadimplência real mostra que cada faixa significa o que promete: A perto de 1%, E acima de 40%.",
               "The new queue's distribution by band is practically the same as the test set's, so the model is being applied to a "
               "population similar to the one it trained on. The real-default-rate column shows each band means what it promises: "
               "A near 1%, E above 40%."))

    st.markdown(L("**Política de crédito aplicada à fila** (usa o corte e as hipóteses de margem/prejuízo da aba Decisão)",
                "**Credit policy applied to the queue** (uses the cutoff and margin/loss assumptions from the Decision tab)"))
    aprov = pn < best.corte
    lucro = float(((1 - pn[aprov]) * ganho - pn[aprov] * perda).sum())
    k = st.columns(4)
    k[0].metric(L("Propostas", "Applications"), f"{len(pn):,}".replace(",", "."))
    k[1].metric(L("Aprovadas", "Approved"), f"{aprov.mean():.0%}")
    k[2].metric(L("Inadimplência esperada dos aprovados", "Expected default rate of approved"), f"{pn[aprov].mean():.1%}")
    k[3].metric(L("Lucro esperado", "Expected profit"),
                (f"$ {lucro / 1e6:,.1f} mi" if lang() == "en" else f"R$ {lucro / 1e6:,.1f} mi".replace(",", "X").replace(".", ",").replace("X", ".")))

    st.markdown(L("**Consultar uma proposta da fila**", "**Look up an application in the queue**"))
    if "cli" not in st.session_state:
        st.session_state.cli = int(pn.idxmax())
    c1_, c2_ = st.columns([1, 3])
    with c1_:
        st.session_state.cli = st.number_input(L("ID do cliente", "Customer ID"), int(Xn.index.min()), int(Xn.index.max()), st.session_state.cli)
        if st.button(L("Sortear cliente", "Draw a customer")):
            st.session_state.cli = int(pn.sample(1).index[0]); st.rerun()
    cid = st.session_state.cli
    pc_ = float(pn.loc[cid])
    ct = contribuicoes(Xn.loc[[cid]]).reindex(Xn.columns)
    tbc = pd.DataFrame({"v": ct.index, "shap": ct.values, "valor": Xn.loc[cid].values})
    tbc = tbc.reindex(tbc.shap.abs().sort_values(ascending=False).index).head(6).iloc[::-1]
    nd = "n/a" if lang() == "en" else "n/d"
    tbc["rot"] = tbc.apply(lambda r: f"{nome(r.v)} = {r.valor:g}" if pd.notna(r.valor) else f"{nome(r.v)} = {nd}", axis=1)
    with c2_:
        dec = L("APROVAR", "APPROVE") if pc_ < best.corte else L("NEGAR", "DENY")
        st.markdown(L(
            f"**Cliente {cid}** · prob. de inadimplência **{pc_:.1%}** · faixa **{faixa(pd.Series([pc_])).astype(str)[0]}** "
            f"· decisão pelo corte de {best.corte:.1%}: **{dec}**",
            f"**Customer {cid}** · default prob. **{pc_:.1%}** · band **{faixa(pd.Series([pc_])).astype(str)[0]}** "
            f"· decision at the {best.corte:.1%} cutoff: **{dec}**",
        ))
        f = go.Figure(go.Bar(x=tbc.shap, y=tbc.rot, orientation="h", marker_color=[VERM if v > 0 else AZUL for v in tbc.shap]))
        f.update_layout(height=260, margin=dict(t=10, b=10),
                        xaxis_title=L("Contribuição no log-odds (vermelho ↑ risco)", "Contribution to log-odds (red ↑ risk)"), yaxis_title="")
        st.plotly_chart(f, width="stretch")

    st.markdown(L("**Suba a sua própria base**", "**Upload your own data**"))
    st.caption(L("CSV com as 10 colunas do Give Me Some Credit (nomes originais do Kaggle ou em português). Nada é guardado: a base é pontuada na hora.",
               "CSV with the 10 Give Me Some Credit columns (Kaggle's original names or Portuguese). Nothing is stored: the data is scored on the fly."))
    modelo = pd.DataFrame([[0.15, 34, 0, 0.30, 6500, 7, 0, 1, 0, 1], [0.95, 52, 2, 0.65, 3200, 12, 1, 0, 0, 3],
                           [0.05, 41, 0, 0.10, 12000, 5, 0, 2, 0, 0]], columns=list(RENAME))
    st.download_button(L("Baixar planilha modelo (3 clientes fictícios)", "Download sample sheet (3 fictional customers)"),
                        modelo.to_csv(index=False).encode(), "modelo_base_nova.csv", "text/csv")
    up = st.file_uploader(L("Arquivo CSV", "CSV file"), type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
            raw = raw.drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed") or c == "SeriousDlqin2yrs"])
            raw = padronizar(raw)
            falta = [c for c in ENTRADAS if c not in raw.columns]
            if falta:
                st.error(L("Faltam colunas: ", "Missing columns: ") + ", ".join(falta))
            else:
                res = pontuar(raw)
                out = pd.concat([raw.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
                out["decisao"] = np.where(out.prob_inadimplencia < best.corte, L("APROVAR", "APPROVE"), L("NEGAR", "DENY"))
                st.success(L(
                    f"{len(out):,} clientes pontuados · {(out.decisao == 'APROVAR').mean():.0%} aprovados pelo corte de {best.corte:.1%}".replace(",", "."),
                    f"{len(out):,} customers scored · {(out.decisao == 'APPROVE').mean():.0%} approved at the {best.corte:.1%} cutoff",
                ))
                st.dataframe(out.head(50).style.format({"prob_inadimplencia": "{:.1%}"}), width="stretch")
                st.download_button(L("Baixar base pontuada", "Download scored data"), out.to_csv(index=False).encode(), "base_pontuada.csv", "text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(L(f"Não consegui ler o arquivo: {e}", f"Couldn't read the file: {e}"))

# ------------------------------------------------------------------ 5 monitoramento
with t5:
    st.subheader(L("O modelo continua valendo amanhã?", "Will the model still hold up tomorrow?"))
    st.markdown(L(
        "**PSI** mede quanto a distribuição de uma variável (ou do score) mudou em relação à base de treino — não "
        "mede se o modelo acerta, mede se o mundo que ele está vendo hoje ainda parece com o mundo em que ele "
        "aprendeu. Regra de bolso: `< 0,10` estável · `0,10–0,25` atenção · `> 0,25` população mudou. "
        "Explicação completa na aba **Conceitos**.",
        "**PSI** measures how much a variable's (or the score's) distribution changed relative to the training "
        "data — it doesn't measure whether the model is accurate, it measures whether the world it's seeing today "
        "still looks like the world it learned from. Rule of thumb: `< 0.10` stable · `0.10–0.25` watch closely · "
        "`> 0.25` population changed. Full explanation on the **Concepts** tab.",
    ))
    ps = D["psi"].copy()
    ps["variavel"] = ps.variavel.map(nome)
    if lang() == "en":
        ps = ps.rename(columns={"variavel": "variable", "psi": "psi"})
    st.dataframe(ps.style.format({c: "{:.4f}" for c in ps.columns if c != ("variable" if lang() == "en" else "variavel")}),
                 width="stretch", hide_index=True, height=250)
    st.info(L(
        "O PSI real dá praticamente zero porque a base de teste do Kaggle é um sorteio aleatório da mesma população. "
        "Isso mostra que o cálculo funciona, mas não mostra um alerta. Para ver o alerta disparar, use o teste de estresse abaixo.",
        "The real PSI comes out near zero because Kaggle's test set is a random draw from the same population. "
        "That shows the calculation works, but doesn't trigger an alert. To see the alert fire, use the stress "
        "test below.",
    ))
    st.markdown(L("**Teste de estresse: e se a população mudar?**", "**Stress test: what if the population changes?**"))
    a, b = st.columns(2)
    cenarios_pt = ["Clientes mais endividados (uso do limite ↑)", "Queda de renda", "Mais atrasos recentes"]
    cenarios_en = ["More indebted customers (limit usage ↑)", "Income drop", "More recent late payments"]
    cen_disp = a.selectbox(L("Cenário", "Scenario"), cenarios_en if lang() == "en" else cenarios_pt)
    cen = cenarios_pt[cenarios_en.index(cen_disp)] if lang() == "en" else cen_disp
    inten = b.slider(L("Intensidade", "Intensity"), 0.0, 1.0, 0.4, 0.05)
    base = D["feat"].copy()
    est = base.copy()
    if cen.startswith("Clientes"):
        est["uso_limite"] = est["uso_limite"] * (1 + 1.5 * inten)
        var = "uso_limite"
    elif cen.startswith("Queda"):
        est["renda_mensal"] = est["renda_mensal"] * (1 - 0.5 * inten)
        est["renda_per_capita"] = est["renda_mensal"] / (1 + est["dependentes"].fillna(0))
        var = "renda_mensal"
    else:
        add = np.random.default_rng(1).binomial(1, min(inten, 1) * 0.5, len(est))
        est["atraso_30_59"] = est["atraso_30_59"].fillna(0) + add
        est["atrasos_total"] = est[["atraso_30_59", "atraso_60_89", "atraso_90"]].sum(axis=1, min_count=1)
        var = "atraso_30_59"
    mdl = carregar_modelo()
    p0, p1 = mdl.predict_proba(base)[:, 1], mdl.predict_proba(est)[:, 1]
    psi_v, psi_s = psi(base[var], est[var]), psi(p0, p1)
    k = st.columns(3)
    k[0].metric(f"PSI — {nome(var)}", f"{psi_v:.3f}", help=L("PSI da variável isolada no cenário simulado.", "PSI of the isolated variable in the simulated scenario."))
    k[1].metric(L("PSI — score", "PSI — score"), f"{psi_s:.3f}",
                help=L("PSI da probabilidade final do modelo entre a base original e a base simulada. "
                       "< 0,10 estável · 0,10–0,25 atenção · > 0,25 população mudou.",
                       "PSI of the model's final probability between the original and simulated data. "
                       "< 0.10 stable · 0.10–0.25 watch closely · > 0.25 population changed."))
    k[2].metric(L("Risco médio previsto", "Mean predicted risk"), f"{p1.mean():.1%}", f"{p1.mean() - p0.mean():+.1%} " + L("vs hoje", "vs today"), delta_color="inverse")
    nivel = (L("🟢 estável", "🟢 stable") if psi_s < 0.10 else
             (L("🟡 atenção", "🟡 watch") if psi_s < 0.25 else L("🔴 população mudou: revisar/retreinar", "🔴 population changed: review/retrain")))
    st.markdown(L(f"Status do score: **{nivel}**. Numa esteira real, esse alerta dispara antes de a inadimplência aparecer, "
                "porque ela só é observada 2 anos depois.",
                f"Score status: **{nivel}**. In a real pipeline, this alert fires before the default even shows up, "
                "because it's only observed 2 years later."))

# ------------------------------------------------------------------ 5 limites
with t6:
    st.subheader(L("O que este case não prova", "What this case doesn't prove"))
    st.markdown(L(
        """
- **Sem data na base.** O split é estratificado aleatório, não temporal. Em produção o certo é treinar no passado e testar em safras futuras; o AUC real tende a ser um pouco menor.
- **Uma base, um país, uma época.** Give Me Some Credit é um dataset de competição, não a carteira de uma instituição brasileira.
- **Custos são hipóteses.** A margem e o prejuízo da aba Decisão são sliders ilustrativos.
- **PSI real ≈ 0** por construção da base (ver aba Monitoramento). O teste de estresse é simulado e está identificado como tal.
- **Sem viés de seleção tratado.** Só há rótulo de quem foi aprovado no passado (reject inference fica fora do escopo).
- **Regulação.** Em crédito de verdade entram ainda validação independente, análise de vieses e política de crédito por cima do modelo.
        """,
        """
- **No date field in the data.** The split is random stratified, not time-based. In production the right approach is to train on the past and test on future cohorts; real-world AUC tends to be a bit lower.
- **One dataset, one country, one era.** Give Me Some Credit is a competition dataset, not a Brazilian institution's real portfolio.
- **Costs are assumptions.** The margin and loss in the Decision tab are illustrative sliders.
- **Real PSI ≈ 0** by construction of the dataset (see the Monitoring tab). The stress test is simulated and labeled as such.
- **No selection bias correction.** There's only a label for whoever was approved in the past (reject inference is out of scope).
- **Regulation.** Real-world credit also involves independent validation, bias analysis, and a credit policy layered on top of the model.
        """,
    ))
    st.markdown(L("Código e pipeline completos no repositório do projeto (`train_credito.py`, `moiras/metrics.py`).",
                "Full code and pipeline in the project's repository (`train_credito.py`, `moiras/metrics.py`)."))

st.divider()
st.caption("Moiras · Gabriela Gatti Rodrigues · [LinkedIn](https://linkedin.com/in/gabriela-gatti-rodrigues) · "
           f"[{L('Dashboard Inteligente', 'Intelligent Dashboard')}]({DASH_URL})")
