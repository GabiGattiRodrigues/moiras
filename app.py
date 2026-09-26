"""Moiras — navegação entre a capa e os cases."""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Moiras · disputa de modelos de classificação", page_icon="🧵", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "pt"

with st.sidebar:
    st.image(str(Path(__file__).parent / "assets" / "moiras.png"), width="stretch")
    idioma = st.radio("Idioma / Language", ["🇧🇷 Português", "🇬🇧 English"], horizontal=True,
                       index=0 if st.session_state.lang == "pt" else 1)
    st.session_state.lang = "pt" if idioma.startswith("🇧🇷") else "en"
    pt = st.session_state.lang == "pt"
    st.caption(
        "As Moiras fiam, medem e cortam o fio do destino. Aqui, o destino de cada cliente ou funcionário vira uma "
        "probabilidade." if pt else
        "The Fates spin, measure and cut the thread of destiny. Here, each customer's or employee's fate becomes a "
        "probability."
    )
    st.divider()

    paginas = [
        st.Page("inicio.py", title="Início" if pt else "Home", icon="🧵", default=True),
        st.Page("credito.py", title="Case 1 · Crédito" if pt else "Case 1 · Credit", icon="💳", url_path="credito"),
        st.Page("turnover.py", title="Case 2 · Turnover" if pt else "Case 2 · Turnover", icon="👥", url_path="turnover"),
        st.Page("aquisicao.py", title="Case 3 · Aquisição" if pt else "Case 3 · Acquisition", icon="📞", url_path="aquisicao"),
        st.Page("conceitos.py", title="Conceitos / Concepts", icon="📚", url_path="conceitos"),
    ]
    # position="hidden" tira o menu automático (que o Streamlit sempre trava no topo da barra lateral),
    # para a imagem e o seletor de idioma poderem ficar por cima da lista de páginas, montada à mão logo abaixo.
    pg = st.navigation(paginas, position="hidden")
    for pagina in paginas:
        st.page_link(pagina)

pg.run()
