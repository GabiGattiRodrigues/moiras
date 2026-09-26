"""Idioma global do app: um seletor na barra lateral (compartilhado entre páginas via session_state)."""
import streamlit as st


def L(pt: str, en: str) -> str:
    """Devolve o texto em português ou inglês conforme o idioma escolhido na barra lateral."""
    return en if st.session_state.get("lang") == "en" else pt


def lang() -> str:
    return st.session_state.get("lang", "pt")
