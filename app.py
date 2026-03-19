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
    .pm-title {
        color: #2563EB;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.08;
        margin: 0;
    }
    .pm-subtitle {
        color: #2563EB;
        font-size: 1rem;
        font-weight: 600;
        opacity: 0.92;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
    .home-hero {
        background: linear-gradient(135deg, #EFF6FF 0%, #FFFFFF 55%, #DBEAFE 100%);
        border: 1px solid #BFDBFE;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        margin: 0.4rem 0 1.1rem 0;
    }
    .home-hero-title {
        color: #1D4ED8;
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }
    .home-hero-text {
        color: #334155;
        font-size: 0.98rem;
        line-height: 1.6;
        margin: 0;
    }
    .home-stat {
        background: #FFFFFF;
        border: 1px solid #DBEAFE;
        border-radius: 16px;
        padding: 0.95rem 1rem;
        min-height: 122px;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.08);
    }
    .home-stat-label {
        color: #2563EB;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.45rem;
    }
    .home-stat-value {
        color: #0F172A;
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.2rem;
    }
    .home-stat-text {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.45;
        margin: 0;
    }
    .home-card {
        background: #FFFFFF;
        border: 1px solid #DBEAFE;
        border-radius: 16px;
        padding: 1rem 1.05rem;
        height: 100%;
    }
    .home-card-title {
        color: #2563EB;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .home-card-text {
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.55;
        margin: 0;
    }
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
LOGO_PATH = ROOT / "doc" / "Passos-magicos-icon-cor.png"
TEAM_MEMBERS = [
    "Alisson Cordeiro Nóbrega",
    "Lucas Benevides Miranda",
    "Marcos Vinícius Fernandes de Freitas",
    "Rodrigo Mallet e Ribeiro de Carvalho"
]


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
    years = sorted([int(y) for y in df["ano_referencia"].dropna().unique().tolist()])

    st.markdown(
        """
        <div class="home-hero">
            <div class="home-hero-title">Objetivo do projeto</div>
            <p class="home-hero-text">
                Esta aplicação organiza a análise gerencial e a solução preditiva do Datathon Fase 5
                para apoiar a Associação Passos Mágicos na identificação de sinais de defasagem,
                priorização de acompanhamento e compreensão mais profunda da trajetória educacional dos estudantes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### :blue[Contexto da Associação]")
    st.markdown(
        """
        A Passos Mágicos tem uma trajetória de mais de três décadas na transformação da vida de crianças e jovens
        em situação de vulnerabilidade social por meio da educação. A iniciativa começou em 1992, em Embu-Guaçu,
        e foi ampliada em 2016 para um modelo social e educacional mais estruturado, combinando educação de qualidade,
        apoio psicológico e psicopedagógico, ampliação de repertório e protagonismo.

        Neste projeto, o foco não é apenas descrever indicadores. A proposta é usar dados para apoiar decisões
        sobre onde estão os principais pontos de atenção, quais dimensões mais pressionam a defasagem e como a ONG
        pode agir de forma mais preventiva com cada estudante.
        """
    )

    st.markdown("### :blue[Desafio Proposto]")
    st.markdown(
        """
        O desafio proposto pela pós-tech parte de uma pergunta central: como transformar a base PEDE de 2022, 2023 e 2024
        em leitura gerencial útil para a Passos Mágicos? A resposta foi organizada em duas frentes complementares.

        A primeira frente investiga a trajetória dos indicadores educacionais e socioemocionais para entender
        defasagem, desempenho acadêmico, engajamento, autoavaliação, aspectos psicossociais, aspectos psicopedagógicos,
        ponto de virada e desempenho global. A segunda frente transforma esses sinais em uma previsão operacional,
        estimando a probabilidade de um aluno ou aluna entrar em risco de defasagem.
        """
    )

    n1, n2 = st.columns(2)
    with n1:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-title">Notebook 1: EDA e Storytelling</div>
                <p class="home-card-text">
                    <code>scripts/1_EDA_e_Storytelling.ipynb</code><br><br>
                    Consolida a análise exploratória e responde às perguntas de negócio do desafio, com foco
                    em interpretar o comportamento dos indicadores ao longo do tempo e gerar recomendações úteis
                    para a atuação da ONG.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with n2:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-title">Notebook 2: Modelo Preditivo</div>
                <p class="home-card-text">
                    <code>scripts/2_Modelo_Preditivo.ipynb</code><br><br>
                    Estrutura a preparação dos dados, a lógica de modelagem e a etapa preditiva que foi incorporada
                    à aplicação para estimar a chance de entrada em risco de defasagem.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### :blue[O que esta aplicação entrega]")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-title">Início</div>
                <p class="home-card-text">
                    Contextualiza o caso da Passos Mágicos, a base usada no desafio e a estrutura técnica do projeto.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-title">Análise Exploratória</div>
                <p class="home-card-text">
                    Reúne os gráficos e interpretações do storytelling para apoiar leitura analítica, priorização
                    de problemas e entendimento dos principais padrões dos estudantes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-title">Modelo Preditivo</div>
                <p class="home-card-text">
                    Permite simular cenários e estimar, de forma operacional, a probabilidade de risco de defasagem
                    para apoiar acompanhamento preventivo.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="home-stat">
                <div class="home-stat-label">Base Consolidada</div>
                <div class="home-stat-value">{len(df):,}</div>
                <p class="home-stat-text">registros integrando o acompanhamento educacional e os indicadores do PEDE.</p>
            </div>
            """.replace(",", "."),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="home-stat">
                <div class="home-stat-label">Estudantes</div>
                <div class="home-stat-value">{df['ra'].nunique():,}</div>
                <p class="home-stat-text">alunos e alunas únicos acompanhados na base disponibilizada para o desafio.</p>
            </div>
            """.replace(",", "."),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="home-stat">
                <div class="home-stat-label">Recorte Temporal</div>
                <div class="home-stat-value">{' | '.join(map(str, years))}</div>
                <p class="home-stat-text">anos cobertos pela base consolidada usada no storytelling e na modelagem.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### :blue[Equipe Responsável]")
    st.markdown("\n".join([f"- {member}" for member in TEAM_MEMBERS]))


def main() -> None:
    title_col, logo_col = st.columns([0.88, 0.12])
    with title_col:
        st.title('📚 :blue[Datathon Fase 5 - Associação Passos Mágicos]')
        st.write(
            ':blue[Análise gerencial e solução preditiva para apoiar decisões educacionais com foco em impacto social.]',
            unsafe_allow_html=True,
        )
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=250)

    try:
        df, df_long = load_data()
    except Exception as error:
        st.error(f"Falha ao carregar os dados: {error}")
        st.stop()

    tab_inicio, tab_eda, tab_modelo = st.tabs(["Início", "Análise exploratória", "Modelo Preditivo"])

    with tab_inicio:
        render_inicio(df)

    with tab_eda:
        render_analise_exploratoria_tab(df, df_long)

    with tab_modelo:
        render_modelo_preditivo_tab(df)


if __name__ == "__main__":
    main()
