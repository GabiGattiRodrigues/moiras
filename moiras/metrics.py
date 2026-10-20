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


# ----------------------------------------------------------------------------- divergências
# PSI, KL, Jensen-Shannon e IV são a MESMA família de conta: comparar duas distribuições faixa a faixa.
#   KL(P‖Q) = Σ p·ln(p/q)                        -> assimétrica: "surpresa" de quem esperava Q e viu P
#   PSI     = Σ (p-q)·ln(p/q) = KL(P‖Q) + KL(Q‖P) -> KL simétrica (divergência de Jeffreys)
#   JS      = ½KL(P‖M) + ½KL(Q‖M), M = (P+Q)/2   -> simétrica e limitada (0 a ln 2), aguenta faixa vazia
#   IV      = mesma conta do PSI, só que entre bons e maus em vez de treino e hoje
EPS = 1e-4
FALTANTE = "faltante"
OUTRO = "valor novo"


def cortes(ref, bins: int = 10) -> tuple[str, np.ndarray]:
    """Define as faixas UMA vez, na base de referência, e devolve (tipo, valores).

    - poucos valores distintos (binária, contagem curta, dummy): cada valor é uma faixa ("cat");
    - senão: quantis da referência ("num"). Quantis repetidos (massa grande num valor só, ex.: 90% com zero
      atraso) viram um corte só, então "zero" x "mais que zero" continua sendo separado em vez de sumir.
    """
    r = pd.to_numeric(pd.Series(ref), errors="coerce").dropna().to_numpy(dtype=float)
    vals = np.unique(r)
    if len(vals) <= bins:
        return "cat", vals
    return "num", np.unique(np.quantile(r, np.linspace(0, 1, bins + 1))[1:-1])


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def rotulos(corte: tuple[str, np.ndarray]) -> list[str]:
    tipo, v = corte
    if tipo == "cat":
        return [_fmt(x) for x in v]
    if len(v) == 0:
        return ["tudo"]
    return ([f"≤ {_fmt(v[0])}"] + [f"({_fmt(a)}, {_fmt(b)}]" for a, b in zip(v[:-1], v[1:])]
            + [f"> {_fmt(v[-1])}"])


def rotular(x, corte: tuple[str, np.ndarray]) -> pd.Series:
    """Aplica as faixas da referência a qualquer base. Faltante é faixa própria (o % de faltantes mudar também
    é mudança de população); valor que não existia na referência, numa variável categórica, cai em "valor novo"."""
    tipo, v = corte
    xs = pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce")
    nomes = rotulos(corte)
    if tipo == "cat":
        xv = xs.fillna(np.inf).to_numpy()
        if len(v) == 0:
            lab = np.full(len(xv), OUTRO, dtype=object)
        else:
            idx = np.clip(np.searchsorted(v, xv), 0, len(v) - 1)
            lab = np.where(np.isclose(v[idx], xv), np.array(nomes, dtype=object)[idx], OUTRO)
    else:
        idx = np.searchsorted(v, xs.to_numpy(), side="left")  # (a, b]: valor igual ao corte fica na faixa de baixo
        lab = np.array(nomes, dtype=object)[np.clip(idx, 0, len(nomes) - 1)]
    lab = pd.Series(lab, dtype=object)
    lab[xs.isna().to_numpy()] = FALTANTE
    return lab


def tabela_faixas(p_amostra, q_amostra, bins: int = 10, ref=None) -> pd.DataFrame:
    """Distribuição de duas amostras nas MESMAS faixas, e a contribuição de cada faixa para cada divergência.

    p_amostra : a base "de hoje" (ou os maus, no IV)
    q_amostra : a base de referência (treino; ou os bons, no IV)
    ref       : de onde saem as faixas (padrão: q_amostra, a referência). No IV as faixas saem da base inteira.
    """
    corte = cortes(q_amostra if ref is None else ref, bins)
    lp, lq = rotular(p_amostra, corte), rotular(q_amostra, corte)
    ordem = rotulos(corte) + [c for c in (OUTRO, FALTANTE) if c in set(lp) | set(lq)]
    n_p, n_q = lp.value_counts().reindex(ordem, fill_value=0), lq.value_counts().reindex(ordem, fill_value=0)
    t = pd.DataFrame({"faixa": ordem, "n_p": n_p.to_numpy(), "n_q": n_q.to_numpy()})
    t = t[(t.n_p + t.n_q) > 0].reset_index(drop=True)
    t["pct_p"], t["pct_q"] = t.n_p / max(t.n_p.sum(), 1), t.n_q / max(t.n_q.sum(), 1)
    # KL/PSI não aceitam faixa vazia (ln 0): piso de 0,01% e renormaliza. O JS não precisa disso.
    pp, pq = np.clip(t.pct_p, EPS, None), np.clip(t.pct_q, EPS, None)
    pp, pq = pp / pp.sum(), pq / pq.sum()
    t["kl_pq"] = pp * np.log(pp / pq)          # contribuição para KL(P‖Q): pode ser negativa faixa a faixa
    t["kl_qp"] = pq * np.log(pq / pp)          # contribuição para KL(Q‖P)
    t["psi"] = t.kl_pq + t.kl_qp               # = (p-q)·ln(p/q): nunca negativa
    m = (t.pct_p + t.pct_q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(t.pct_p > 0, t.pct_p * np.log(t.pct_p / m), 0.0)
        b = np.where(t.pct_q > 0, t.pct_q * np.log(t.pct_q / m), 0.0)
    t["js"] = (a + b) / 2
    return t


def divergencias(atual, referencia, bins: int = 10) -> dict:
    """Compara a base de hoje com a de referência (treino) nas faixas da referência.

    kl_hoje_treino : KL(hoje‖treino). Pesa quando aparece gente onde o treino tinha pouca (o modelo extrapola).
    kl_treino_hoje : KL(treino‖hoje). Pesa quando some gente de onde o treino tinha muita.
    psi            : a soma das duas (é a KL simétrica). Régua de mercado: <0,10 · 0,10–0,25 · >0,25.
    js             : Jensen-Shannon (nats, máx. ln 2 ≈ 0,693). Para mudanças pequenas, JS ≈ PSI/8.
    """
    t = tabela_faixas(atual, referencia, bins)
    if len(t) < 2:
        return {"psi": 0.0, "kl_hoje_treino": 0.0, "kl_treino_hoje": 0.0, "js": 0.0}
    return {"psi": float(t.psi.sum()), "kl_hoje_treino": float(t.kl_pq.sum()),
            "kl_treino_hoje": float(t.kl_qp.sum()), "js": float(t.js.sum())}


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index = KL(hoje‖treino) + KL(treino‖hoje). Faixas pelos quantis da referência.

    Regra de bolso: <0,10 estável | 0,10-0,25 atenção | >0,25 mudou de verdade.
    """
    return divergencias(actual, expected, bins)["psi"]


def information_value(x, y, bins: int = 10) -> tuple[float, pd.DataFrame]:
    """IV de uma variável: a mesma KL simétrica do PSI, só que entre a distribuição dos BONS (y=0) e a dos
    MAUS (y=1), faixa a faixa. As faixas saem da base inteira. Devolve (iv, tabela de WoE por faixa).

    WoE = ln(%bons / %maus) por faixa (negativo = faixa concentra eventos). IV = Σ (%bons-%maus)·WoE.
    Régua de bolso (Siddiqi): <0,02 sem poder · 0,02–0,1 fraco · 0,1–0,3 médio · 0,3–0,5 forte · >0,5 suspeito.
    """
    x = pd.Series(np.asarray(x, dtype=float))
    y = np.asarray(y).astype(int)
    t = tabela_faixas(x[y == 0], x[y == 1], bins, ref=x)
    t = t.rename(columns={"n_p": "bons", "n_q": "eventos", "pct_p": "pct_bons", "pct_q": "pct_eventos"})
    pb, pm = np.clip(t.pct_bons, EPS, None), np.clip(t.pct_eventos, EPS, None)
    pb, pm = pb / pb.sum(), pm / pm.sum()
    t["n"] = t.bons + t.eventos
    t["taxa_evento"] = t.eventos / t.n.where(t.n > 0, 1)
    t["woe"] = np.log(pb / pm)
    t["iv_parcial"] = (pb - pm) * t.woe
    t = t[["faixa", "n", "bons", "eventos", "taxa_evento", "pct_bons", "pct_eventos", "woe", "iv_parcial"]]
    return (float(t.iv_parcial.sum()) if len(t) > 1 else 0.0), t


def piso_ruido_psi(n_ref: int, n_atual: int, faixas: int = 10) -> float:
    """PSI esperado quando NADA mudou, só por sorteio: ≈ (faixas-1)·(1/n_ref + 1/n_atual).
    Com base pequena, PSI abaixo disso é ruído de amostra, não mudança de população."""
    return (faixas - 1) * (1 / n_ref + 1 / n_atual)


def forca_iv(iv: float) -> str:
    return ("sem poder" if iv < 0.02 else "fraco" if iv < 0.1 else "médio" if iv < 0.3 else
            "forte" if iv < 0.5 else "suspeito")


def exportar_divergencias(Xtr: pd.DataFrame, ytr, Xatual: pd.DataFrame, p_tr, p_atual, out, nome_score: str,
                          p_teste, y_teste, so_continuas: bool = False) -> None:
    """Grava psi.csv (PSI + as duas KL + JS, treino x base atual, por variável e pelo score),
    iv.csv (IV por variável no treino; IV do score no TESTE, porque no treino o modelo já viu as respostas e o
    IV sairia inflado) e iv_faixas.csv (WoE faixa a faixa, para o app)."""
    from pathlib import Path
    out = Path(out)
    cols = [c for c in Xtr.columns if not so_continuas or Xtr[c].nunique() > 2]
    rows = [{"variavel": c, **divergencias(Xatual[c], Xtr[c])} for c in cols]
    rows.append({"variavel": nome_score, **divergencias(p_atual, p_tr)})
    (pd.DataFrame(rows).sort_values("psi", ascending=False)
     [["variavel", "psi", "kl_hoje_treino", "kl_treino_hoje", "js"]].to_csv(out / "psi.csv", index=False))

    ivs, faixas = [], []
    pares = [(c, Xtr[c], ytr) for c in Xtr.columns] + [(nome_score, pd.Series(np.asarray(p_teste)), y_teste)]
    for c, x, y in pares:
        iv, t = information_value(x, y)
        ivs.append({"variavel": c, "iv": iv, "forca": forca_iv(iv)})
        faixas.append(t.assign(variavel=c, ordem=range(len(t))))
    pd.DataFrame(ivs).sort_values("iv", ascending=False).to_csv(out / "iv.csv", index=False)
    pd.concat(faixas)[["variavel", "ordem", "faixa", "n", "bons", "eventos", "taxa_evento", "pct_bons",
                       "pct_eventos", "woe", "iv_parcial"]].to_csv(out / "iv_faixas.csv", index=False)


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


def profit_by_decile_acao(y, p, valor: float, sucesso: float, custo: float) -> pd.DataFrame:
    """Resultado esperado abordando os N decis de MAIOR risco/propensão, em vez de fração contínua da fila.

    Decil 1 = 10% mais arriscado/propenso (mesma convenção de decile_table e profit_by_decile). Cada linha ``n``
    responde: "e se eu abordar os n decis de maior risco (1, 2, ..., n)?" — como uma política real costuma
    priorizar, por faixa, não por casa decimal de probabilidade.
    """
    y, p = np.asarray(y), np.asarray(p)
    ordem = np.argsort(-p)  # do maior risco/propensão para o menor
    yo = y[ordem]
    n_total = len(p)
    rows = []
    for n in range(1, 11):
        k = int(round(n_total * n / 10))
        k = max(k, 1)
        abord_y = yo[:k]
        # corte = menor probabilidade ainda abordada (fronteira do decil abordado)
        corte = float(p[ordem][:k].min()) if k > 0 else 0.0
        eventos = int((abord_y == 1).sum())
        rows.append({"decis_abordados": n, "corte": corte, "fracao": k / n_total,
                     "captura_eventos": eventos / max(int((y == 1).sum()), 1),
                     "lucro": eventos * valor * sucesso - k * custo})
    return pd.DataFrame(rows)
