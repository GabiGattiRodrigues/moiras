"""Blocos de tela das divergências (PSI, KL, Jensen-Shannon) e do IV, compartilhados pelas páginas dos cases."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from moiras.lang import L, lang
from moiras.metrics import FALTANTE, OUTRO, piso_ruido_psi, tabela_faixas

AZUL, AZUL_ESC, CINZA, VERM = "#1f5fbf", "#0b2a5b", "#9aa5b1", "#d1495b"
FORCA_EN = {"sem poder": "no power", "fraco": "weak", "médio": "medium", "forte": "strong", "suspeito": "suspicious"}
FORCA_COR = {"sem poder": CINZA, "fraco": "#8fb3e8", "médio": AZUL, "forte": AZUL_ESC, "suspeito": VERM}


def _faixa(f: str) -> str:
    if lang() != "en":
        return f
    return {FALTANTE: "missing", OUTRO: "new value", "tudo": "all"}.get(f, f)


def _num(v: float, casas: int = 4) -> str:
    s = f"{v:.{casas}f}"
    return s if lang() == "en" else s.replace(".", ",")


# ----------------------------------------------------------------------------- monitoramento
def explicacao() -> None:
    st.markdown(L(
        "Três réguas para a mesma pergunta — *a base de hoje ainda parece com a do treino?* — todas comparando as "
        "duas distribuições faixa a faixa:\n"
        "- **KL(hoje ‖ treino)**: pesa quando aparece gente em faixas onde o treino tinha pouca. É o modelo pontuando "
        "um perfil que ele viu pouco.\n"
        "- **KL(treino ‖ hoje)**: pesa quando some gente de faixas onde o treino tinha muita.\n"
        "- **PSI** é a soma das duas: a divergência KL simétrica. Régua de mercado: `< 0,10` estável · "
        "`0,10–0,25` atenção · `> 0,25` população mudou.\n"
        "- **Jensen-Shannon (JS)**: outra versão simétrica, que vai de 0 a 0,693 e não explode quando uma faixa "
        "fica vazia. Para mudanças pequenas, JS ≈ PSI ÷ 8.\n\n"
        "Explicação completa na aba **Conceitos**.",
        "Three yardsticks for the same question — *does today's data still look like the training data?* — all "
        "comparing the two distributions bin by bin:\n"
        "- **KL(today ‖ train)**: weighs heavily when people show up in bins where training had few. That's the "
        "model scoring a profile it barely saw.\n"
        "- **KL(train ‖ today)**: weighs heavily when bins that were full in training empty out.\n"
        "- **PSI** is the sum of both: the symmetric KL divergence. Market rule of thumb: `< 0.10` stable · "
        "`0.10–0.25` watch closely · `> 0.25` population changed.\n"
        "- **Jensen-Shannon (JS)**: another symmetric version, ranging from 0 to 0.693, that doesn't blow up when "
        "a bin goes empty. For small changes, JS ≈ PSI ÷ 8.\n\n"
        "Full explanation on the **Concepts** tab.",
    ))


def tabela_psi(ps: pd.DataFrame, nome: Callable[[str], str], n_ref: int, n_atual: int,
               texto_atual: tuple[str, str]) -> None:
    """Tabela PSI/KL/JS por variável + o piso de ruído para o tamanho das bases comparadas."""
    t = ps.copy()
    t["variavel"] = t.variavel.map(nome)
    cols = (["Variable", "PSI", "KL(today ‖ train)", "KL(train ‖ today)", "Jensen-Shannon"] if lang() == "en" else
            ["Variável", "PSI", "KL(hoje ‖ treino)", "KL(treino ‖ hoje)", "Jensen-Shannon"])
    t = t[["variavel", "psi", "kl_hoje_treino", "kl_treino_hoje", "js"]]
    t.columns = cols
    st.dataframe(t.style.format({c: "{:.5f}" for c in cols[1:]}), width="stretch", hide_index=True, height=280)
    piso = piso_ruido_psi(n_ref, n_atual)
    n_r, n_a = f"{n_ref:,}", f"{n_atual:,}"
    if lang() != "en":
        n_r, n_a = n_r.replace(",", "."), n_a.replace(",", ".")
    st.caption(L(
        f"Confira: PSI = KL(hoje ‖ treino) + KL(treino ‖ hoje) em todas as linhas. **Piso de ruído:** mesmo sem "
        f"mudança nenhuma, só por sorteio, o PSI entre {n_r} {texto_atual[0]} do treino e {n_a} {texto_atual[1]} "
        f"fica em torno de **{_num(piso)}** (≈ 9 × (1/{n_r} + 1/{n_a}), com 10 faixas). Abaixo disso é ruído de "
        "amostra, não mudança de população.",
        f"Check: PSI = KL(today ‖ train) + KL(train ‖ today) on every row. **Noise floor:** even with no change "
        f"at all, just from sampling, the PSI between {n_r} training {texto_atual[0]} and {n_a} {texto_atual[1]} "
        f"sits around **{_num(piso)}** (≈ 9 × (1/{n_r} + 1/{n_a}), with 10 bins). Below that it's sampling "
        "noise, not a population change.",
    ))


def decomposicao(p_ref, p_atual, key: str) -> None:
    """Teste de estresse: abre o PSI do score nas duas KL, mostra o JS e onde (em que faixa) está a mudança."""
    t = tabela_faixas(p_atual, p_ref)
    psi_, kl_ht, kl_th, js = t.psi.sum(), t.kl_pq.sum(), t.kl_qp.sum(), t.js.sum()
    st.markdown(L("**Abrindo o PSI do score: de que lado vem a mudança?**",
                  "**Opening up the score's PSI: which side does the change come from?**"))
    st.markdown(L(f"PSI do score **{_num(psi_, 3)}** = KL(hoje ‖ treino) **{_num(kl_ht, 3)}** + "
                  f"KL(treino ‖ hoje) **{_num(kl_th, 3)}**",
                  f"Score PSI **{_num(psi_, 3)}** = KL(today ‖ train) **{_num(kl_ht, 3)}** + "
                  f"KL(train ‖ today) **{_num(kl_th, 3)}**"))
    k = st.columns(3)
    k[0].metric(L("KL(hoje ‖ treino)", "KL(today ‖ train)"), f"{kl_ht:.3f}",
                help=L("Cresce quando aparece gente em faixas de score que eram raras na base de referência.",
                       "Grows when people show up in score bins that were rare in the reference data."))
    k[1].metric(L("KL(treino ‖ hoje)", "KL(train ‖ today)"), f"{kl_th:.3f}",
                help=L("Cresce quando faixas de score que eram comuns na base de referência esvaziam.",
                       "Grows when score bins that were common in the reference data empty out."))
    k[2].metric("Jensen-Shannon", f"{js:.4f}",
                help=L(f"Vai de 0 a 0,693. Para mudanças pequenas, JS ≈ PSI ÷ 8 (aqui: {_num(psi_ / 8)}).",
                       f"Ranges from 0 to 0.693. For small changes, JS ≈ PSI ÷ 8 (here: {_num(psi_ / 8)})."))
    if psi_ < 0.01:
        msg = L("Sem mudança relevante no score.", "No meaningful change in the score.")
    elif kl_ht > 1.2 * kl_th:
        msg = L(f"Pesa mais a **KL(hoje ‖ treino)** ({kl_ht / psi_:.0%} do PSI): apareceu gente em faixas de score que "
                "eram raras antes. O modelo está pontuando um perfil que viu pouco no treino.",
                f"**KL(today ‖ train)** weighs more ({kl_ht / psi_:.0%} of PSI): people showed up in score bins that "
                "used to be rare. The model is scoring a profile it barely saw in training.")
    elif kl_th > 1.2 * kl_ht:
        msg = L(f"Pesa mais a **KL(treino ‖ hoje)** ({kl_th / psi_:.0%} do PSI): faixas de score que eram comuns "
                "esvaziaram. A população saiu de onde o modelo estava acostumado a vê-la.",
                f"**KL(train ‖ today)** weighs more ({kl_th / psi_:.0%} of PSI): score bins that used to be common "
                "emptied out. The population left where the model was used to seeing it.")
    else:
        msg = L("As duas direções pesam parecido: a distribuição se deslocou como um todo, sem criar nem esvaziar "
                "faixas de forma desigual.",
                "Both directions weigh about the same: the distribution shifted as a whole, without unevenly "
                "creating or emptying bins.")
    st.markdown(msg)
    st.caption(L("Quando a base só desliza (todo mundo um pouco mais endividado, por exemplo), as duas metades ficam "
                 "parecidas. Elas se separam quando um perfil raro passa a aparecer muito (pesa a primeira) ou quando "
                 "uma faixa comum some (pesa a segunda). Compare os cenários \"Público novo\" e \"Pré-filtro\" "
                 "na intensidade máxima.",
                 "When the data just slides (everyone a bit more indebted, for example), both halves stay similar. "
                 "They split apart when a rare profile starts showing up a lot (the first one weighs more) or when "
                 "a common bin empties out (the second one weighs more). Compare the \"New audience\" and "
                 "\"Pre-filter\" scenarios at maximum intensity."))
    x = [str(i + 1) for i in range(len(t))]
    hover = [f"{_faixa(f)}<br>{L('referência', 'reference')}: {a:.1%} · {L('cenário', 'scenario')}: {b:.1%}"
             for f, a, b in zip(t.faixa, t.pct_q, t.pct_p)]
    fig = go.Figure()
    fig.add_bar(x=x, y=t.pct_q, name=L("% na base de referência", "% in reference data"), marker_color=CINZA,
                hovertext=hover, hoverinfo="text")
    fig.add_bar(x=x, y=t.pct_p, name=L("% no cenário", "% in scenario"), marker_color=VERM,
                hovertext=hover, hoverinfo="text")
    fig.add_scatter(x=x, y=t.psi, name=L("contribuição para o PSI", "contribution to PSI"), yaxis="y2",
                    mode="lines+markers", line=dict(color=AZUL_ESC))
    fig.update_layout(barmode="group", height=360, margin=dict(t=10), yaxis_tickformat=".0%",
                      xaxis=dict(type="category", dtick=1,
                                 title=L("Faixa do score (1 = menor risco; faixas = decis da base de referência)",
                                         "Score bin (1 = lowest risk; bins = deciles of the reference data)")),
                      yaxis_title=L("% da base", "% of data"),
                      yaxis2=dict(overlaying="y", side="right", title=L("contribuição para o PSI", "contribution to PSI"),
                                  showgrid=False, rangemode="tozero", tickformat=".3f"),
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch", key=key)
    st.caption(L("Na base de referência cada faixa tem 10% da base por construção. A linha mostra quanto cada faixa "
                 "contribui para o PSI: é ali que a mudança está.",
                 "In the reference data each bin holds 10% by construction. The line shows how much each bin "
                 "contributes to the PSI: that's where the change is."))


# ----------------------------------------------------------------------------- IV
def bloco_iv(iv: pd.DataFrame, faixas: pd.DataFrame, nome: Callable[[str], str], score: str, evento: str,
             key: str) -> None:
    st.markdown(L("**Antes do modelo: quanto cada variável separa sozinha? (Information Value)**",
                  "**Before the model: how much does each variable separate on its own? (Information Value)**"))
    st.caption(L(
        f"O IV é a **mesma conta do PSI** (a KL simétrica), só que comparando o grupo com evento ({evento}) e o grupo "
        "sem evento, em vez de treino e hoje. Quanto mais diferentes as duas distribuições numa variável, mais ela separa. "
        "Régua de bolso: `< 0,02` sem poder · `0,02–0,1` fraco · `0,1–0,3` médio · `0,3–0,5` forte · `> 0,5` suspeito.",
        f"IV is **the same math as PSI** (symmetric KL), only comparing the group with the event ({evento}) to the "
        "group without it, instead of training to today. The more the two distributions differ on a variable, the better it "
        "separates. Rule of thumb: `< 0.02` no power · `0.02–0.1` weak · `0.1–0.3` medium · `0.3–0.5` strong · "
        "`> 0.5` suspicious.",
    ))
    var = iv[iv.variavel != score].reset_index(drop=True)
    sc = iv[iv.variavel == score]
    a, b = st.columns([2, 3])
    with a:
        top = var.head(12).iloc[::-1]
        forca = top.forca.map(FORCA_EN) if lang() == "en" else top.forca
        fig = go.Figure(go.Bar(x=top.iv, y=top.variavel.map(nome), orientation="h",
                               marker_color=top.forca.map(FORCA_COR), hovertext=forca, hoverinfo="x+text"))
        fig.update_layout(height=420, margin=dict(t=10), xaxis_title="IV", yaxis_title="")
        st.plotly_chart(fig, width="stretch", key=f"iv_bar_{key}")
        if len(sc):
            iv_s, iv_1 = float(sc.iv.iloc[0]), float(var.iv.iloc[0])
            st.caption(L(
                f"O **score do modelo** tem IV **{_num(iv_s, 2)}** no teste, contra {_num(iv_1, 2)} da melhor variável "
                "sozinha: juntar as variáveis é o que o modelo acrescenta. (O IV do score é medido no teste porque no "
                "treino o modelo já viu as respostas.)",
                f"The **model's score** has an IV of **{_num(iv_s, 2)}** on test, against {_num(iv_1, 2)} for the best "
                "single variable: combining variables is what the model adds. (The score's IV is measured on test "
                "because the model already saw the answers in training.)"))
    with b:
        opcoes = list(var.variavel)
        esc = st.selectbox(L("Ver a variável faixa a faixa (WoE)", "See the variable bin by bin (WoE)"), opcoes,
                           format_func=nome, key=f"iv_sel_{key}")
        t = faixas[faixas.variavel == esc].sort_values("ordem")
        lab = t.faixa.map(_faixa)
        fig = go.Figure(go.Bar(x=lab, y=t.woe, marker_color=[VERM if w < 0 else AZUL for w in t.woe],
                               customdata=np.c_[t.n, t.taxa_evento],
                               hovertemplate=L("%{x}<br>WoE %{y:.2f}<br>n = %{customdata[0]:,.0f}"
                                               "<br>taxa de evento %{customdata[1]:.1%}<extra></extra>",
                                               "%{x}<br>WoE %{y:.2f}<br>n = %{customdata[0]:,.0f}"
                                               "<br>event rate %{customdata[1]:.1%}<extra></extra>")))
        fig.update_layout(height=330, margin=dict(t=10), yaxis_title="WoE", xaxis_title="",
                          xaxis=dict(type="category"))
        st.plotly_chart(fig, width="stretch", key=f"iv_woe_{key}")
        linha = var[var.variavel == esc].iloc[0]
        forca = FORCA_EN[linha.forca] if lang() == "en" else linha.forca
        st.caption(L(
            f"**{nome(esc)}**: IV {_num(linha.iv, 3)} ({forca}). WoE = ln(% sem evento ÷ % com evento) na faixa. "
            f"**Vermelho (WoE < 0)**: a faixa concentra o grupo com evento ({evento}); azul: o grupo sem. O IV soma, "
            "faixa a faixa, (diferença de %) × WoE.",
            f"**{nome(esc)}**: IV {_num(linha.iv, 3)} ({forca}). WoE = ln(% without event ÷ % with event) in the bin. "
            f"**Red (WoE < 0)**: the bin concentrates the group with the event ({evento}); blue: the group without. IV adds up, bin by "
            "bin, (difference in %) × WoE."))
    if (var.forca == "suspeito").any():
        sus = ", ".join(nome(v) for v in var.loc[var.forca == "suspeito", "variavel"].head(4))
        st.info(L(
            f"**IV acima de 0,5 ({sus}) não é veredito, é alerta.** \"Bom demais\" pede para checar se a variável "
            "não vaza a resposta (ela só existe depois do evento?). Uma variável legítima também pode ser forte "
            "assim — histórico de atraso em crédito, resultado de campanha anterior em vendas.",
            f"**IV above 0.5 ({sus}) isn't a verdict, it's a flag.** \"Too good\" calls for checking whether the "
            "variable leaks the answer (does it only exist after the event?). A legitimate variable can also be this "
            "strong — delinquency history in credit, past campaign outcome in sales."))
