"""Página genérica de um case (turnover, aquisição): a disputa entre modelos e o que decorre dela."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from moiras.compare import SLUG
from moiras.lang import L, lang

ROOT = Path(__file__).resolve().parent.parent
AZUL, AZUL_ESC, CINZA, VERM, VERDE = "#1f5fbf", "#0b2a5b", "#9aa5b1", "#d1495b", "#2a9d6f"
COR = {"Regressão logística": AZUL, "Árvore de decisão": "#e0a030", "Random forest": VERDE, "XGBoost": VERM}
DASH_URL = "https://dashboardinteligente.streamlit.app"

MODELO_EN = {"Regressão logística": "Logistic regression", "Árvore de decisão": "Decision tree",
             "Random forest": "Random forest", "XGBoost": "XGBoost"}

# Traduções das strings de negócio que vêm do meta.json (gerado em português por train_cases.py).
META_EN = {
    "turnover": {
        "titulo": "Turnover (Tomoyo): who tends to leave the company?",
        "base": "IBM HR Analytics Employee Attrition (Kaggle) — FICTIONAL dataset created by IBM data scientists",
        "evento": "voluntary exit", "unidade": "employee",
        "validacao": "75/25 stratified split. 5x3 repeated cross-validation, only on the training data, picks the model.",
        "acao": "retention program", "texto_valor": "cost of replacing someone who leaves", "texto_acao": "retention conversation/action",
        "notas": [
            "Fictional dataset: useful for comparing models, not for drawing conclusions about real people.",
            "Only 237 exits out of 1,470 employees: the test set has ~59 events, so test metrics are noisy.",
            "In People Analytics, the correct use is aggregated risk by team/profile, not a decision about one individual.",
        ],
    },
    "aquisicao": {
        "titulo": "Product acquisition: who accepts a term deposit?",
        "base": "Bank Marketing (UCI, Portuguese bank, 2008–2010, phone campaigns)",
        "evento": "accepted the deposit", "unidade": "customer",
        "validacao": "80/20 stratified split. 5-fold cross-validation, only on the training data, picks the model.",
        "acao": "sales call", "texto_valor": "margin from one conversion", "texto_acao": "cost of one call",
        "notas": [
            "The `duration` column (call length) was dropped: it only exists after the call and would inflate the model.",
            "Macro variables (euribor, employment, confidence) were also dropped: they identify the campaign period, not the customer.",
            "I tested an out-of-time validation (training on the older 80%, testing on the most recent 20%): performance "
            "collapses and swings from 0.45 to 0.75 AUC across models, because the conversion rate goes from 6% to 31% "
            "and interest rates fall outside the training range. The random split overestimates the performance you'd "
            "get in a future campaign.",
            "2008–2010 dataset from Portugal: doesn't represent today's reality, nor Brazil's.",
        ],
    },
}


def _m(meta: dict, caso: str, campo: str):
    return META_EN.get(caso, {}).get(campo, meta[campo]) if lang() == "en" else meta[campo]


@st.cache_data
def carregar(caso: str) -> dict:
    d = ROOT / "artifacts" / caso
    r = lambda n: pd.read_csv(d / n)
    rf_dots = [(d / f"rf_arvore_{i}.dot").read_text(encoding="utf-8") for i in range(6) if (d / f"rf_arvore_{i}.dot").exists()]
    return {
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")), "cmp": r("comparacao.csv"), "scores": r("scores_teste.csv"),
        "imp": r("importancia.csv"), "coef": r("logistica_coef.csv"), "dt_dot": (d / "arvore_dt.dot").read_text(encoding="utf-8"),
        "dec": {m: r(f"decis_{s}.csv") for m, s in SLUG.items()},
        "cal": {m: r(f"calibracao_{s}.csv") for m, s in SLUG.items()},
        "rf_ranking": r("rf_ranking.csv"), "rf_dots": rf_dots,
    }


def mnome(m: str) -> str:
    return MODELO_EN.get(m, m) if lang() == "en" else m


def render(caso: str) -> None:
    D = carregar(caso)
    meta, cmp_, sc = D["meta"], D["cmp"], D["scores"]
    esc = meta["escolhido"]
    e = cmp_.set_index("modelo")
    unidade, evento = _m(meta, caso, "unidade"), _m(meta, caso, "evento")

    st.title(_m(meta, caso, "titulo"))
    st.markdown(L(f"**Base:** {_m(meta, caso, 'base')}. **Como validei:** {_m(meta, caso, 'validacao')}",
                  f"**Data:** {_m(meta, caso, 'base')}. **How I validated it:** {_m(meta, caso, 'validacao')}"))
    st.page_link("conceitos.py", label=L("📚 Não conhece os termos abaixo (KS, AUC...)? Veja o glossário",
                                          "📚 Don't know the terms below (KS, AUC...)? See the glossary"), icon="📚")
    n_tr, n_te = f"{meta['n_treino']:,}".replace(",", "."), f"{meta['n_teste']:,}".replace(",", ".")
    k = st.columns(4)
    k[0].metric(L("Modelo escolhido", "Chosen model"), mnome(esc),
                help=L("O modelo vencedor pela regra de 1 erro-padrão (veja a aba 1 · A disputa e a aba Conceitos).",
                       "The winning model under the 1-standard-error rule (see the 1 · The contest tab and the Concepts tab)."))
    k[1].metric(L("AUC no teste", "AUC on test"), f"{e.loc[esc, 'teste_auc']:.3f}",
                f"{e.loc[esc, 'teste_auc'] - e.loc['XGBoost', 'teste_auc']:+.3f} vs XGBoost" if esc != "XGBoost" else None,
                help=L("Chance de, pegando um caso positivo e um negativo ao acaso, o modelo apontar risco maior para "
                       "o positivo. 50% = chutando; 100% = acerta sempre. Veja a aba Conceitos.",
                       "Chance that, picking one positive and one negative case at random, the model gives a higher "
                       "risk to the positive one. 50% = guessing; 100% = always right. See the Concepts tab."))
    k[2].metric(L("KS no teste", "KS on test"), f"{e.loc[esc, 'teste_ks']:.3f}",
                help=L("Maior distância entre a curva acumulada de positivos e negativos. Acima de 30–40% já é "
                       "considerado forte. Veja a aba Conceitos.",
                       "Largest gap between the cumulative curves of positives and negatives. Above 30–40% is "
                       "already considered strong. See the Concepts tab."))
    k[3].metric(L(f"Taxa de {evento}", f"Rate of {evento}"), f"{meta['taxa_treino']:.1%}",
                help=L(f"Fração da base de treino em que ocorreu \"{evento}\" — a taxa-base contra a qual "
                       "todo o resto é comparado.",
                       f"Share of the training data where \"{evento}\" happened — the base rate everything else "
                       "is compared against."))
    st.caption(L(f"Treino {n_tr} · teste {n_te} ({unidade}s que o modelo nunca viu).",
                 f"Train {n_tr} · test {n_te} ({unidade}s the model never saw)."))

    abas = ["1 · The contest", "2 · Decision", "3 · Explanation", "4 · Limitations"] if lang() == "en" else \
           ["1 · A disputa", "2 · Decisão", "3 · Explicação", "4 · Limites"]
    t1, t2, t3, t4 = st.tabs(abas)

    # ------------------------------------------------------------ disputa
    with t1:
        st.subheader(L("Quatro famílias, uma régua só", "Four families, one single ruler"))
        st.markdown(L(
            f"**Como ler esta página, em miúdos:** cada uma das 4 famílias tentou adivinhar, para cada {unidade}, "
            f"a chance de {evento}. Testamos cada uma várias vezes, embaralhando quem entra no treino e quem "
            "entra na validação, e medimos o quanto cada uma acerta com a **AUC** (uma nota de 50%, que é chutar cara ou "
            "coroa, a 100%, que é acertar sempre). Isso dá uma nota com uma margem de erro (o **± EP**, erro-padrão): "
            "duas notas que se sobrepõem dentro dessa margem são, no fundo, um empate.",
            f"**How to read this page, plainly:** each of the 4 families tried to guess, for every {unidade}, "
            f"the chance of {evento}. We tested each one several times, reshuffling who's in training and who's in "
            "validation, and measured how well each does with **AUC** (a grade from 50%, a coin flip, to 100%, "
            "always right). That gives a grade with a margin of error (the **± SE**, standard error): two grades "
            "that overlap within that margin are, in practice, a tie.",
        ))
        st.markdown(L(
            "**A regra de escolha, definida antes de rodar:** entre os modelos empatados com o melhor (dentro de 1 erro-padrão), "
            "vence o **mais simples** (logística < árvore < random forest < XGBoost) — porque um modelo mais simples é mais "
            "fácil de explicar e de manter, e não vale a pena pagar esse preço só por um empate técnico. "
            "O teste final (a coluna com o losango ◆ no gráfico) é dado que nenhum modelo usou para se ajustar; ele só confirma "
            "a escolha, nunca decide.",
            "**The selection rule, set before running anything:** among models tied with the best one (within 1 "
            "standard error), the **simplest** wins (logistic < decision tree < random forest < XGBoost) — because "
            "a simpler model is easier to explain and maintain, and it's not worth paying that price for a merely "
            "technical tie. The final test set (the column with the ◆ diamond on the chart) is data no model used "
            "to fit itself; it only confirms the choice, never decides it.",
        ))
        f = go.Figure()
        f.add_bar(x=[mnome(m) for m in cmp_.modelo], y=cmp_.cv_auc, error_y=dict(type="data", array=cmp_.cv_ep),
                  name=L("Validação cruzada (± 1 EP)", "Cross-validation (± 1 SE)"), marker_color=[COR[m] for m in cmp_.modelo])
        f.add_scatter(x=[mnome(m) for m in cmp_.modelo], y=cmp_.teste_auc, mode="markers",
                      name=L("Teste (nunca visto)", "Test (never seen)"), marker=dict(symbol="diamond", size=13, color="black"))
        lo = max(0.5, float(min(cmp_.cv_auc.min(), cmp_.teste_auc.min())) - 0.05)
        f.update_layout(height=360, margin=dict(t=10), yaxis=dict(range=[lo, min(1.0, cmp_.cv_auc.max() + 0.06)], title="AUC"),
                        showlegend=True, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(f, width="stretch")
        best = cmp_.loc[cmp_.melhor_cv].iloc[0]
        if esc == best.modelo:
            st.success(L(
                f"**Por que {esc} venceu:** foi o modelo com a melhor nota de validação ({best.cv_auc:.3f} de AUC) "
                "e, de quebra, nenhum modelo mais simples chegou perto o suficiente dele para tomar o lugar pela regra. "
                "Venceu limpo, sem precisar da regra de desempate.",
                f"**Why {mnome(esc)} won:** it was the model with the best validation score ({best.cv_auc:.3f} AUC), "
                "and on top of that no simpler model came close enough to take its place under the rule. A clean "
                "win, no need for the tiebreaker.",
            ))
        else:
            st.info(L(
                f"**Por que {esc} venceu, e não {best.modelo}:** {best.modelo} teve a nota mais alta na validação "
                f"({best.cv_auc:.3f} de AUC), mas {esc} chegou perto o suficiente ({e.loc[esc, 'cv_auc']:.3f}, dentro de "
                f"1 erro-padrão) e é mais simples de explicar e manter. Pela regra combinada antes de rodar, o empate "
                f"técnico vai para o mais simples: fica com {esc}.",
                f"**Why {mnome(esc)} won, and not {mnome(best.modelo)}:** {mnome(best.modelo)} had the highest "
                f"validation score ({best.cv_auc:.3f} AUC), but {mnome(esc)} came close enough "
                f"({e.loc[esc, 'cv_auc']:.3f}, within 1 standard error) and is simpler to explain and maintain. "
                f"Under the rule agreed on before running anything, a technical tie goes to the simpler model: it "
                f"stays with {mnome(esc)}.",
            ))
        show = cmp_[["modelo", "cv_auc", "cv_ep", "teste_auc", "teste_ks", "teste_brier", "escolhido"]].copy()
        show["escolhido"] = np.where(show.escolhido, "✓", "")
        if lang() == "en":
            show["modelo"] = show["modelo"].map(mnome)
            show.columns = ["Model", "Validation AUC", "± SE", "Test AUC", "Test KS", "Test Brier", "Chosen"]
        else:
            show.columns = ["Modelo", "AUC validação", "± EP", "AUC teste", "KS teste", "Brier teste", "Escolhido"]
        st.dataframe(show.style.format({show.columns[1]: "{:.4f}", show.columns[2]: "{:.4f}", show.columns[3]: "{:.4f}",
                                        show.columns[4]: "{:.3f}", show.columns[5]: "{:.4f}"}), width="stretch", hide_index=True)
        a, b = st.columns(2)
        with a:
            st.markdown(L("**Lift por decil de risco** (1 = maior risco previsto)", "**Lift by risk decile** (1 = highest predicted risk)"))
            fig = go.Figure()
            for m in cmp_.modelo:
                fig.add_scatter(x=D["dec"][m].decil, y=D["dec"][m].lift, mode="lines+markers", name=mnome(m), line_color=COR[m],
                                line_width=4 if m == esc else 1.5)
            fig.add_hline(y=1, line_dash="dot", line_color="black")
            fig.update_layout(height=320, margin=dict(t=10), xaxis_title=L("Decil", "Decile"), yaxis_title="Lift")
            st.plotly_chart(fig, width="stretch")
        with b:
            st.markdown(L("**Calibração**: probabilidade prevista × taxa real", "**Calibration**: predicted probability × real rate"))
            fig = go.Figure()
            mx = max(max(D["cal"][m].p_medio.max(), D["cal"][m].taxa_real.max()) for m in cmp_.modelo)
            fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dot", color="black"), name=L("Perfeita", "Perfect"))
            for m in cmp_.modelo:
                fig.add_scatter(x=D["cal"][m].p_medio, y=D["cal"][m].taxa_real, mode="lines+markers", name=mnome(m), line_color=COR[m],
                                line_width=4 if m == esc else 1.5)
            fig.update_layout(height=320, margin=dict(t=10), xaxis_title=L("Prob. média prevista", "Mean predicted prob."),
                              yaxis_title=L("Taxa real", "Real rate"))
            st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------ decisão
    acao = _m(meta, caso, "acao")
    texto_valor = _m(meta, caso, "texto_valor")
    texto_acao = _m(meta, caso, "texto_acao")
    with t2:
        st.subheader(L(f"Vale a pena agir? Quem priorizar para a {acao}", f"Is it worth acting? Who to prioritize for the {acao}"))
        st.caption(L("⚠️ Valores são **hipóteses ilustrativas ajustáveis**, não dados reais de nenhuma empresa.",
                     "⚠️ Values are **illustrative, adjustable assumptions**, not real data from any company."))
        st.markdown(L(
            "O modelo só ordena quem é mais ou menos propenso; para virar uma decisão de **até onde vale a pena "
            "agir**, alguém precisa dizer quanto vale acertar, qual a chance da ação funcionar, e quanto custa agir. "
            "É isso que os três controles abaixo representam.",
            "The model only ranks who's more or less likely; turning that into a decision of **how far it's worth "
            "acting** requires someone to say how much a hit is worth, what the chance of the action working is, "
            "and how much acting costs. That's what the three controls below represent.",
        ))
        a, b, c = st.columns(3)
        v = a.number_input(L(f"Valor de um evento evitado/ganho: {texto_valor} (R$)", f"Value of an avoided/won event: {texto_valor} ($)"),
                            10, 500000, int(meta["valor_evento"]), 10,
                            help=L("Quanto vale, em R$, evitar (ou conquistar) um caso — por exemplo, o custo de repor "
                                   "um funcionário que sai, ou a receita de um cliente que compra o produto.",
                                   "How much it's worth, in $, to avoid (or win) a case — for example, the cost of "
                                   "replacing an employee who leaves, or the revenue from a customer who buys the product."))
        s = b.slider(L("Chance da ação funcionar", "Chance the action works"), 0.05, 1.0, float(meta["sucesso_acao"]), 0.05,
                      help=L("Mesmo abordando a pessoa certa, a ação nem sempre funciona (nem toda retenção evita a "
                             "saída, nem toda oferta converte). Essa é a chance de dar certo quando se tenta.",
                             "Even reaching out to the right person, the action doesn't always work (not every "
                             "retention effort prevents the exit, not every offer converts). This is the chance it "
                             "works when tried."))
        cu = c.number_input(L(f"Custo por pessoa abordada: {texto_acao} (R$)", f"Cost per person reached: {texto_acao} ($)"),
                             1, 50000, int(meta["custo_acao"]), 1,
                             help=L("Quanto custa abordar uma pessoa, tentando ou não dar certo — tempo do time, "
                                    "desconto oferecido, ligação feita, etc.",
                                    "How much it costs to reach out to a person, whether or not it works — team "
                                    "time, discount offered, the call made, etc."))
        y = sc["y"].to_numpy()
        p = sc[esc].to_numpy()
        ordem = np.argsort(-p)
        ganho = np.cumsum(y[ordem] * v * s - cu)
        frac = np.arange(1, len(y) + 1) / len(y)
        aleatorio = frac * len(y) * (y.mean() * v * s - cu)
        kbest = int(np.argmax(ganho))
        f = go.Figure()
        f.add_scatter(x=frac, y=ganho, name=L(f"Priorizando pelo {esc}", f"Prioritizing by {mnome(esc)}"), line_color=AZUL)
        f.add_scatter(x=frac, y=aleatorio, name=L("Abordando ao acaso", "Reaching out at random"), line=dict(color=CINZA, dash="dot"))
        f.add_vline(x=frac[kbest], line_dash="dot", line_color=AZUL)
        f.update_layout(height=340, margin=dict(t=10),
                        xaxis_title=L(f"% dos {unidade}s abordados (do maior para o menor risco)",
                                      f"% of {unidade}s reached (highest to lowest risk)"),
                        yaxis_title=L("Resultado esperado no teste (R$)", "Expected result on test set ($)"),
                        xaxis_tickformat=".0%", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(f, width="stretch")
        st.caption(L(
            "**Como ler este gráfico:** o eixo de baixo é a % de pessoas abordadas, começando pela de maior risco/"
            "propensão segundo o modelo; o eixo da esquerda é o resultado acumulado esperado. A linha azul sobe "
            "enquanto abordar mais gente ainda compensa, atinge um pico (o ponto ótimo, linha pontilhada) e desce "
            "depois — abordar quem sobra já não vale o custo. A linha cinza pontilhada mostra o que aconteceria "
            "abordando pessoas ao acaso, para comparação.",
            "**How to read this chart:** the bottom axis is the % of people reached, starting with the highest "
            "risk/propensity per the model; the left axis is the expected cumulative result. The blue line rises "
            "while reaching out to more people still pays off, peaks (the optimal point, dotted line), then falls "
            "— reaching the rest isn't worth the cost anymore. The dotted gray line shows what would happen "
            "reaching out at random, for comparison.",
        ))
        kk = st.columns(4)
        kk[0].metric(L("Abordar até", "Reach out up to"), f"{frac[kbest]:.0%}",
                     help=L("Fração da lista, ordenada do maior para o menor risco pelo modelo, que maximiza o "
                            "resultado esperado com as hipóteses ao lado.",
                            "Share of the list, ranked from highest to lowest risk by the model, that maximizes the "
                            "expected result under the assumptions above."))
        kk[1].metric(L("Capturaria dos eventos", "Would capture of events"), f"{y[ordem][:kbest + 1].sum() / y.sum():.0%}",
                     help=L("Dos casos que realmente aconteceriam, que fração estaria dentro dessa fatia priorizada "
                            "pelo modelo.",
                            "Of the cases that would actually happen, what share would fall inside this slice "
                            "prioritized by the model."))
        kk[2].metric(L("Resultado (teste)", "Result (test)"),
                     (f"$ {ganho[kbest]:,.0f}" if lang() == "en" else f"R$ {ganho[kbest]:,.0f}".replace(",", ".")),
                     help=L("Resultado esperado acumulado até esse ponto de corte, nos dados de teste.",
                            "Expected cumulative result up to this cutoff, on the test data."))
        kk[3].metric(L("Ao acaso, mesmo alcance", "At random, same reach"),
                     (f"$ {aleatorio[kbest]:,.0f}" if lang() == "en" else f"R$ {aleatorio[kbest]:,.0f}".replace(",", ".")),
                     help=L("Resultado esperado se, em vez de usar o modelo, você abordasse a mesma quantidade de "
                            "pessoas escolhidas ao acaso.",
                            "Expected result if, instead of using the model, you reached out to the same number of "
                            "randomly chosen people."))
        st.markdown(L(
            f"**O que esses números querem dizer:** o modelo ordena os {unidade}s do mais para o menos arriscado/propenso. "
            f"\"Abordar até {frac[kbest]:.0%}\" significa: comece pelo topo dessa lista e pare quando chegar em "
            f"{frac[kbest]:.0%} do total — passar disso, o próximo da lista já não compensa mais (custa mais do que "
            "o resultado esperado dele). Fazendo isso, você pegaria "
            f"**{y[ordem][:kbest + 1].sum() / y.sum():.0%} de todos os casos de {evento}** abordando só "
            f"{frac[kbest]:.0%} das pessoas — bem menos trabalho do que abordar todo mundo, e bem melhor do que abordar "
            "ao acaso (linha pontilhada no gráfico). Ranquear é o que o modelo entrega; decidir **onde parar** depende de "
            "quanto vale acertar e quanto custa abordar — mexa nos valores acima e veja o ponto ótimo andar.",
            f"**What these numbers mean:** the model ranks the {unidade}s from most to least risky/likely. \"Reach "
            f"out up to {frac[kbest]:.0%}\" means: start from the top of that list and stop once you hit "
            f"{frac[kbest]:.0%} of the total — past that point, the next person on the list no longer pays off "
            "(costs more than their expected result). Doing this, you'd capture "
            f"**{y[ordem][:kbest + 1].sum() / y.sum():.0%} of all {evento} cases** while reaching out to only "
            f"{frac[kbest]:.0%} of the people — far less work than reaching everyone, and much better than random "
            "outreach (dotted line on the chart). Ranking is what the model delivers; deciding **where to stop** "
            "depends on how much a hit is worth and how much reaching out costs — move the values above and watch "
            "the optimal point shift.",
        ))

    # ------------------------------------------------------------ explicação
    with t3:
        st.subheader(L("O que move o risco?", "What drives the risk?"))
        a, b = st.columns(2)
        with a:
            st.markdown(L(f"**Importância por permutação** ({esc}, no teste): quanto a AUC cai ao embaralhar a variável",
                        f"**Permutation importance** ({mnome(esc)}, on test): how much AUC drops when the variable is shuffled"))
            imp = D["imp"].head(12).iloc[::-1]
            fig = go.Figure(go.Bar(x=imp.queda_auc, y=imp.variavel, orientation="h", marker_color=AZUL))
            fig.update_layout(height=420, margin=dict(t=10), xaxis_title=L("Queda na AUC", "Drop in AUC"), yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        with b:
            st.markdown(L("**Regressão logística**: razão de chances por desvio-padrão (>1 aumenta o risco)",
                        "**Logistic regression**: odds ratio per standard deviation (>1 increases risk)"))
            co = D["coef"].head(12).iloc[::-1]
            fig = go.Figure(go.Bar(x=co.odds_ratio - 1, y=co.variavel, orientation="h", base=1,
                                   marker_color=[VERM if o > 1 else AZUL for o in co.odds_ratio]))
            fig.update_layout(height=420, margin=dict(t=10), xaxis_title=L("Razão de chances", "Odds ratio"), yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        st.caption(L("Não conhece \"importância por permutação\" ou \"razão de chances\"? Veja a aba **Conceitos** no menu.",
                   "Don't know \"permutation importance\" or \"odds ratio\"? See the **Concepts** tab in the menu."))
        st.divider()
        dtm = meta["arvore_dt"]
        st.markdown(L(
            f"**A árvore de decisão que competiu de verdade** (profundidade {dtm['profundidade']}, "
            f"{dtm['folhas']} folhas — não é uma versão simplificada à parte, é a mesma da tabela e do gráfico acima)",
            f"**The decision tree that actually competed** (depth {dtm['profundidade']}, {dtm['folhas']} leaves — "
            "not a separate simplified version, it's the same one from the table and chart above)",
        ))
        st.caption(L("Lê-se de cima para baixo: em cada caixa, siga o **sim** ou o **não** até chegar numa folha (oval). "
                   "Cor mais forte = maior concentração do evento; os números mostram a fração de casos que passa por ali "
                   "e a taxa de fica/evento nessa fração.",
                   "Read top to bottom: at each box, follow **yes** or **no** until you reach a leaf (oval). Stronger "
                   "color = higher concentration of the event; the numbers show the share of cases passing through "
                   "there and the stay/event rate in that share."))
        st.graphviz_chart(D["dt_dot"], width="stretch")
        st.caption(L(
            f"Sozinha, essa árvore tem AUC {dtm['auc']:.3f} no teste" +
            (" — foi a **vencedora** desta disputa." if esc == "Árvore de decisão"
             else f", contra {e.loc[esc, 'teste_auc']:.3f} do modelo escolhido ({esc})."),
            f"On its own, this tree has AUC {dtm['auc']:.3f} on the test set" +
            (" — it was the **winner** of this contest." if esc == "Árvore de decisão"
             else f", against {e.loc[esc, 'teste_auc']:.3f} for the chosen model ({mnome(esc)})."),
        ))

        st.divider()
        rf_r = D["rf_ranking"]
        n_rf = meta.get("rf_n_arvores", len(rf_r))
        st.markdown(L(f"**Random forest: um bando de {n_rf} árvores, cada uma vota**",
                    f"**Random forest: a pack of {n_rf} trees, each one votes**"))
        st.caption(L("Nenhuma árvore do random forest decide sozinha: cada uma vê uma parte diferente dos dados, dá seu palpite, "
                   "e a resposta final é a média de todas. Aqui embaixo você pode olhar dentro de algumas delas, da que "
                   "sozinha teve o melhor desempenho no teste até a 6ª melhor — o restante segue o mesmo padrão.",
                   "No single random forest tree decides on its own: each one sees a different slice of the data, "
                   "gives its own guess, and the final answer is the average of all of them. Below you can look "
                   "inside a few of them, from the one that alone did best on the test set to the 6th best — the "
                   "rest follows the same pattern."))
        pos = st.slider(L("Árvore (nº 0 = a melhor sozinha)", "Tree (# 0 = best alone)"), 0, len(D["rf_dots"]) - 1, 0, key=f"rf_{caso}")
        st.graphviz_chart(D["rf_dots"][pos], width="stretch")
        auc_i = float(rf_r.loc[pos, "auc_individual"])
        st.caption(L(
            f"Essa árvore sozinha (mostrando só os 3 primeiros níveis, para caber na tela) tem AUC {auc_i:.3f} no teste. "
            f"O random forest inteiro, com a média de todas as {n_rf}, "
            f"chega a {e.loc['Random forest', 'teste_auc']:.3f} — uma árvore sozinha erra mais; o bando acerta mais "
            "porque os erros de cada uma, em geral, não se repetem nas outras.",
            f"This tree alone (showing only the first 3 levels, to fit the screen) has AUC {auc_i:.3f} on the test "
            f"set. The whole random forest, averaging all {n_rf}, reaches {e.loc['Random forest', 'teste_auc']:.3f} "
            "— a single tree makes more mistakes; the pack does better because each tree's mistakes generally "
            "don't repeat across the others.",
        ))

    # ------------------------------------------------------------ limites
    with t4:
        st.subheader(L("O que este case não prova", "What this case doesn't prove"))
        notas = _m(meta, caso, "notas") if caso in META_EN and lang() == "en" else meta["notas"]
        st.markdown("\n".join(f"- {n}" for n in notas))
        st.markdown(L(
            "- **A regra de escolha é minha decisão de método.** Com outra regra (por exemplo, sempre o de maior AUC) a escolha poderia mudar; "
            "a tabela na primeira aba mostra os números para você julgar.\n"
            "- **Ajuste de hiperparâmetros é enxuto** (grades pequenas), igual para todas as famílias; um ajuste mais pesado pode aproximar os modelos.",
            "- **The selection rule is my own methodological choice.** With a different rule (e.g., always the highest AUC) the "
            "outcome could change; the table on the first tab shows the numbers so you can judge for yourself.\n"
            "- **Hyperparameter tuning is lean** (small grids), the same for every family; heavier tuning could bring the models closer together.",
        ))
