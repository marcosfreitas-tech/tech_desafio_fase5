from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from tab_analise_exploratoria import render_analise_exploratoria_tab
from tab_modelo_preditivo import render_modelo_preditivo_tab


st.set_page_config(
    page_title="Passos Mágicos | Datathon Fase 5",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    button[kind="header"] {display: none;}
    .eda-note {
        background: #F6F2E9;
        border: 1px solid #E9C46A;
        border-left: 6px solid #2A9D8F;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin: 0.6rem 0 1rem 0;
    }
    .eda-note strong { color: #264653; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
PROCESSED_PATH = ROOT / "data" / "processed" / "pede_consolidado_2022_2024.csv"


def build_longitudinal_dataset(df: pd.DataFrame) -> pd.DataFrame:
    base = df.rename(columns={"ano_referencia": "ano_base"}).copy()
    future = df[["ra", "ano_referencia", "ian", "ida", "inde_ano", "ieg", "ips", "ipp", "ipv"]].copy()
    future = future.rename(
        columns={
            "ano_referencia": "ano_base",
            "ian": "ian_prox",
            "ida": "ida_prox",
            "inde_ano": "inde_prox",
            "ieg": "ieg_prox",
            "ips": "ips_prox",
            "ipp": "ipp_prox",
            "ipv": "ipv_prox",
        }
    )
    future["ano_base"] = future["ano_base"] - 1
    merged = base.merge(future, on=["ra", "ano_base"], how="left")
    merged["delta_ida_prox"] = merged["ida_prox"] - merged["ida"]
    merged["delta_inde_prox"] = merged["inde_prox"] - merged["inde_ano"]
    merged["delta_ian_prox"] = merged["ian_prox"] - merged["ian"]
    return merged


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {PROCESSED_PATH}")
    df = pd.read_csv(PROCESSED_PATH)

    numeric_cols = [
        "ano_referencia",
        "idade",
        "defasagem",
        "inde_22",
        "inde_23",
        "inde_ano",
        "ian",
        "ida",
        "ieg",
        "iaa",
        "ips",
        "ipp",
        "ipv",
        "nota_matematica",
        "nota_portugues",
        "nota_ingles",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"nota_matematica", "nota_portugues", "nota_ingles"}.issubset(df.columns):
        df["media_notas"] = df[["nota_matematica", "nota_portugues", "nota_ingles"]].mean(axis=1)
    if {"iaa", "ieg", "ips", "ipp"}.issubset(df.columns):
        df["media_comportamental"] = df[["iaa", "ieg", "ips", "ipp"]].mean(axis=1)

    df_long = build_longitudinal_dataset(df)
    return df, df_long


def render_inicio(df: pd.DataFrame) -> None:
    st.subheader("Contexto")
    st.markdown(
        """
        Dashboard do Datathon Fase 5 da Associação Passos Mágicos, consolidando as análises do notebook
        `1_EDA_e_Storytelling.ipynb` com foco em leitura gerencial objetiva.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{len(df):,}".replace(",", "."))
    c2.metric("Alunos únicos", f"{df['ra'].nunique():,}".replace(",", "."))
    years = sorted([int(y) for y in df["ano_referencia"].dropna().unique().tolist()])
    c3.metric("Anos", " | ".join(map(str, years)))

    st.markdown(
        """
        **Base de storytelling:** documentos oficiais da pasta `doc` + notebook de EDA.
        """
    )


def main() -> None:
    st.title("Datathon Fase 5 - Associação Passos Mágicos")

    try:
        df, df_long = load_data()
    except Exception as error:
        st.error(f"Falha ao carregar os dados: {error}")
        st.stop()

    tab_inicio, tab_eda, tab_modelo = st.tabs(["In\u00edcio", "An\u00e1lise explorat\u00f3ria", "Modelo Preditivo"])

    with tab_inicio:
        render_inicio(df)

    with tab_eda:
        render_analise_exploratoria_tab(df, df_long)

    with tab_modelo:
        render_modelo_preditivo_tab()


if __name__ == "__main__":
    main()
