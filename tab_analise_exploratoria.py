from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

px.defaults.template = "plotly_white"
COLOR_SEQUENCE = ["#1D3557", "#457B9D", "#2A9D8F", "#E9C46A", "#E76F51", "#A8DADC"]
DIVERGING_SCALE = "RdBu"
SEQUENTIAL_SCALE = ["#F1FAEE", "#A8DADC", "#1D3557"]
YEAR_COLOR_MAP = {"2022": "#1D3557", "2023": "#457B9D", "2024": "#2A9D8F"}
px.defaults.color_discrete_sequence = COLOR_SEQUENCE
def render_analysis_header(question: str, importance: str, approach: str) -> None:
    st.markdown(f"**Pergunta orientadora:** {question}")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Por que importa:** {importance}")
    c2.markdown(f"**Como foi analisado:** {approach}")


def render_exec_note(message: str, implication: str) -> None:
    st.markdown(
        f"""
        <div class="eda-note">
        <strong>Conclusão</strong><br>
        {message}<br><br>
        <strong>Implicação prática</strong><br>
        {implication}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_corr(df: pd.DataFrame) -> None:
    st.subheader("Matriz de correlação")
    corr_features = [
        "inde_ano",
        "ian",
        "ida",
        "ieg",
        "iaa",
        "ips",
        "ipp",
        "ipv",
        "defasagem",
        "nota_matematica",
        "nota_portugues",
        "nota_ingles",
    ]
    corr_features = [c for c in corr_features if c in df.columns]
    indicator_name_map = {
        "inde_ano": "INDE anual",
        "ian": "IAN - adequacao de nivel",
        "ida": "IDA - desempenho academico",
        "ieg": "IEG - engajamento",
        "iaa": "IAA - autoavaliacao",
        "ips": "IPS - indice psicossocial",
        "ipp": "IPP - indice psicopedagogico",
        "ipv": "IPV - ponto de virada",
        "defasagem": "Defasagem escolar",
        "nota_matematica": "Nota de matematica",
        "nota_portugues": "Nota de portugues",
        "nota_ingles": "Nota de ingles",
    }
    corr_matrix = df[corr_features].corr(method="spearman", min_periods=30)
    corr_matrix = corr_matrix.rename(index=indicator_name_map, columns=indicator_name_map)

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=DIVERGING_SCALE,
        zmin=-1,
        zmax=1,
        title="Matriz de correlacao de Spearman - indicadores consolidados",
    )
    fig_corr.update_layout(coloraxis_colorbar={"title": "rho (Spearman)"})
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Obs.: cores mais intensas indicam relações mais fortes.")


def render_q1(df: pd.DataFrame) -> None:
    st.subheader("Q1 - IAN: defasagem e evolução temporal")
    ian_summary = (
        df.dropna(subset=["ian"])
        .groupby("ano_referencia", as_index=False)
        .agg(
            ian_medio=("ian", "mean"),
            percentual_defasagem_alta=("ian", lambda s: (s <= 5).mean() * 100),
            alunos=("ra", "nunique"),
        )
        .sort_values("ano_referencia")
    )

    fig_ian_line = px.line(
        ian_summary,
        x="ano_referencia",
        y="ian_medio",
        markers=True,
        text=ian_summary["ian_medio"].round(2),
        title="Evolucao anual do IAN medio (adequacao de nivel)",
        labels={
            "ian_medio": "IAN medio",
            "ano_referencia": "Ano de referencia",
        },
    )
    fig_ian_line.update_traces(textposition="top center")
    fig_ian_line.update_xaxes(type="category")
    st.plotly_chart(fig_ian_line, use_container_width=True)
    st.caption("O IAN médio mostra tendência de melhoria no período.")

    def classificar_faixa_ian(value: float) -> str:
        if pd.isna(value):
            return np.nan
        if value <= 5:
            return "Defasagem alta (IAN <= 5)"
        if value <= 7:
            return "Atencao (5 < IAN <= 7)"
        return "Adequado (IAN > 7)"

    ian_mix = df[["ano_referencia", "ian"]].copy()
    ian_mix["faixa_ian"] = ian_mix["ian"].apply(classificar_faixa_ian)
    ian_mix = (
        ian_mix.dropna(subset=["faixa_ian"])
        .groupby(["ano_referencia", "faixa_ian"], as_index=False)
        .size()
        .rename(columns={"size": "quantidade_alunos"})
    )
    ian_mix["percentual"] = ian_mix.groupby("ano_referencia")["quantidade_alunos"].transform(lambda s: 100 * s / s.sum())
    ian_mix["ano_referencia"] = ian_mix["ano_referencia"].astype(int).astype(str)
    ian_mix["faixa_ian"] = pd.Categorical(
        ian_mix["faixa_ian"],
        categories=[
            "Defasagem alta (IAN <= 5)",
            "Atencao (5 < IAN <= 7)",
            "Adequado (IAN > 7)",
        ],
        ordered=True,
    )
    ian_mix = ian_mix.sort_values(["ano_referencia", "faixa_ian"])

    fig_ian_mix = px.bar(
        ian_mix,
        x="ano_referencia",
        y="percentual",
        color="faixa_ian",
        barmode="stack",
        title="Composicao percentual de perfis de IAN por ano",
        labels={
            "percentual": "Percentual de alunos (%)",
            "ano_referencia": "Ano de referencia",
            "faixa_ian": "Perfil de adequacao de nivel (IAN)",
        },
        color_discrete_map={
            "Defasagem alta (IAN <= 5)": "#E76F51",
            "Atencao (5 < IAN <= 7)": "#E9C46A",
            "Adequado (IAN > 7)": "#2A9D8F",
        },
    )
    fig_ian_mix.update_traces(texttemplate="%{y:.1f}%", textposition="inside")
    fig_ian_mix.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=sorted(ian_mix["ano_referencia"].unique().tolist()),
    )
    fig_ian_mix.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_ian_mix, use_container_width=True)
    st.caption("A redução da faixa de defasagem alta e o avanço da faixa adequada são sinais de evolução estrutural.")

    ian_by_gender = (
        df.dropna(subset=["ian", "genero"])
        .groupby(["ano_referencia", "genero"], as_index=False)
        .agg(
            ian_medio=("ian", "mean"),
            percentual_defasagem_alta=("ian", lambda s: (s <= 5).mean() * 100),
            alunos=("ra", "nunique"),
        )
        .sort_values(["ano_referencia", "genero"])
    )
    ian_by_gender["ano_referencia"] = ian_by_gender["ano_referencia"].astype(int).astype(str)

    fig_ian_gender = px.bar(
        ian_by_gender,
        x="ano_referencia",
        y="percentual_defasagem_alta",
        color="genero",
        barmode="group",
        text="percentual_defasagem_alta",
        title="Defasagem alta no IAN por sexo e ano",
        labels={
            "ano_referencia": "Ano de referencia",
            "percentual_defasagem_alta": "Alunos com IAN <= 5 (%)",
            "genero": "Sexo",
        },
        color_discrete_map={"Feminino": "#2A9D8F", "Masculino": "#E76F51"},
    )
    fig_ian_gender.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig_ian_gender.update_xaxes(type="category")
    st.plotly_chart(fig_ian_gender, use_container_width=True)
    st.caption("O recorte por sexo ajuda a identificar grupos com maior necessidade de apoio pedagógico.")

    ian_gender_pivot = ian_by_gender.pivot(index="genero", columns="ano_referencia", values="percentual_defasagem_alta")
    variation_rows = []
    for gender_name in ian_gender_pivot.index:
        row = ian_gender_pivot.loc[gender_name]
        for start_year, end_year in [("2022", "2023"), ("2023", "2024"), ("2022", "2024")]:
            if start_year in row.index and end_year in row.index and pd.notna(row[start_year]) and pd.notna(row[end_year]):
                variation_rows.append(
                    {
                        "genero": gender_name,
                        "periodo": f"{start_year} -> {end_year}",
                        "variacao_defasagem_pp": row[end_year] - row[start_year],
                    }
                )

    gender_variation_df = pd.DataFrame(variation_rows)
    if not gender_variation_df.empty:
        gender_variation_df["rotulo_variacao_pp"] = gender_variation_df["variacao_defasagem_pp"].round(0).map(lambda v: f"{v:+.0f}")

        fig_gender_variation = px.bar(
            gender_variation_df,
            x="periodo",
            y="variacao_defasagem_pp",
            color="genero",
            barmode="group",
            text="rotulo_variacao_pp",
            title="Variacao da defasagem alta no IAN por sexo (pontos percentuais)",
            labels={
                "periodo": "Periodo comparado",
                "variacao_defasagem_pp": "Variacao da taxa de IAN <= 5 (p.p.)",
                "genero": "Sexo",
            },
            color_discrete_map={"Feminino": "#2A9D8F", "Masculino": "#E76F51"},
        )
        fig_gender_variation.update_traces(
            texttemplate="%{text} p.p.",
            textposition="outside",
            hovertemplate="Periodo comparado=%{x}<br>Variacao da taxa de IAN <= 5 (p.p.)=%{y:+.0f}<extra></extra>",
        )
        fig_gender_variation.add_hline(y=0, line_dash="dash", line_color="#264653")
        st.plotly_chart(fig_gender_variation, use_container_width=True)

    age_analysis = df.dropna(subset=["ian", "idade"]).copy()
    age_analysis["faixa_etaria"] = pd.cut(age_analysis["idade"], bins=[0, 10, 13, 16, 30], labels=["7-10 anos", "11-13 anos", "14-16 anos", "17+ anos"])
    ian_by_age = (
        age_analysis.dropna(subset=["faixa_etaria"])
        .groupby(["ano_referencia", "faixa_etaria"], as_index=False)
        .agg(
            ian_medio=("ian", "mean"),
            percentual_defasagem_alta=("ian", lambda s: (s <= 5).mean() * 100),
            alunos=("ra", "nunique"),
        )
    )
    ian_by_age["ano_referencia"] = ian_by_age["ano_referencia"].astype(int).astype(str)

    fig_ian_age = px.bar(
        ian_by_age,
        x="percentual_defasagem_alta",
        y="faixa_etaria",
        color="ano_referencia",
        barmode="group",
        orientation="h",
        text="percentual_defasagem_alta",
        title="Defasagem alta no IAN por faixa etaria",
        labels={
            "percentual_defasagem_alta": "Alunos com IAN <= 5 (%)",
            "faixa_etaria": "Faixa etaria",
            "ano_referencia": "Ano de referencia",
        },
        category_orders={"ano_referencia": ["2024", "2023", "2022"]},
        color_discrete_map={"2024": "#2A9D8F", "2023": "#457B9D", "2022": "#1D3557"},
    )
    fig_ian_age.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_ian_age.update_xaxes(range=[0, 100])
    fig_ian_age.update_yaxes(categoryorder="array", categoryarray=["17+ anos", "14-16 anos", "11-13 anos", "7-10 anos"])
    st.plotly_chart(fig_ian_age, use_container_width=True)
    st.caption("A segmentação por faixa etária orienta ações direcionadas por ciclo de aprendizagem.")


def render_q2(df: pd.DataFrame) -> None:
    st.subheader("Q2 - IDA: melhora, estagnação ou queda")
    ida_summary = (
        df.dropna(subset=["ida"])
        .groupby("ano_referencia", as_index=False)
        .agg(
            ida_media=("ida", "mean"),
            ida_percentil_25=("ida", lambda s: s.quantile(0.25)),
            ida_percentil_75=("ida", lambda s: s.quantile(0.75)),
            alunos=("ra", "nunique"),
        )
        .sort_values("ano_referencia")
    )

    fig_ida_line = px.line(
        ida_summary,
        x="ano_referencia",
        y="ida_media",
        markers=True,
        text=ida_summary["ida_media"].round(2),
        title="Evolucao anual do IDA medio",
        labels={"ano_referencia": "Ano de referencia", "ida_media": "IDA medio"},
    )
    fig_ida_line.update_traces(textposition="top center")
    fig_ida_line.update_xaxes(type="category")
    st.plotly_chart(fig_ida_line, use_container_width=True)
    st.caption("A série anual revela ganho inicial e necessidade de sustentação no último ano.")

    ida_dist = df.dropna(subset=["ida", "ano_referencia"]).copy()
    ida_dist["ano_referencia"] = ida_dist["ano_referencia"].astype(int).astype(str)
    fig_ida_violin = px.violin(
        ida_dist,
        x="ano_referencia",
        y="ida",
        color="ano_referencia",
        box=False,
        points=False,
        title="Distribuicao do IDA por ano (sem linha de mediana)",
        labels={
            "ano_referencia": "Ano de referencia",
            "ida": "IDA - desempenho academico",
        },
        color_discrete_sequence=["#1D3557", "#457B9D", "#2A9D8F"],
    )
    fig_ida_violin.update_layout(showlegend=False)
    fig_ida_violin.update_xaxes(type="category")
    st.plotly_chart(fig_ida_violin, use_container_width=True)
    st.caption("A distribuição mostra heterogeneidade entre estudantes, além da média global.")


def render_q3(df: pd.DataFrame) -> None:
    st.subheader("Q3 - IEG x IDA e IPV")
    corr_rows = []
    for year, part in df.groupby("ano_referencia"):
        sub = part[["ieg", "ida", "ipv"]].dropna()
        if len(sub) < 10:
            continue
        corr_rows.append(
            {
                "ano_referencia": int(year),
                "corr_ieg_ida": sub["ieg"].corr(sub["ida"], method="spearman"),
                "corr_ieg_ipv": sub["ieg"].corr(sub["ipv"], method="spearman"),
                "corr_ida_ipv": sub["ida"].corr(sub["ipv"], method="spearman"),
            }
        )
    corr_df = pd.DataFrame(corr_rows).sort_values("ano_referencia")
    if corr_df.empty:
        st.info("Sem amostra suficiente para Q3.")
        return

    corr_long = corr_df.melt(
        id_vars=["ano_referencia", "alunos_validos"],
        value_vars=["corr_ieg_ida", "corr_ieg_ipv", "corr_ida_ipv"],
        var_name="par_correlacao",
        value_name="correlacao_spearman",
    )
    pair_name_map = {
        "corr_ieg_ida": "IEG x IDA",
        "corr_ieg_ipv": "IEG x IPV",
        "corr_ida_ipv": "IDA x IPV",
    }
    corr_long["Par de indicadores"] = corr_long["par_correlacao"].map(pair_name_map)
    corr_long["Ano de referencia"] = corr_long["ano_referencia"].astype(str)
    corr_long["rotulo"] = corr_long["correlacao_spearman"].map(lambda v: f"{v:.2f}")
    fig_corr_q3 = px.bar(
        corr_long,
        x="Ano de referencia",
        y="correlacao_spearman",
        color="Par de indicadores",
        barmode="group",
        text="rotulo",
        title="Q3 - Correlacoes IEG x IDA, IEG x IPV e IDA x IPV por ano",
        labels={
            "correlacao_spearman": "rho (Spearman)",
            "Ano de referencia": "Ano de referencia",
        },
        color_discrete_map={
            "IEG x IDA": "#7B3F98",
            "IEG x IPV": "#1B9E77",
            "IDA x IPV": "#386CB0",
        },
    )
    fig_corr_q3.update_traces(textposition="outside")
    fig_corr_q3.update_xaxes(type="category")
    fig_corr_q3.update_yaxes(range=[0, corr_long["correlacao_spearman"].max() + 0.12])
    st.plotly_chart(fig_corr_q3, use_container_width=True)
    st.caption("As correlações positivas sustentam o uso do engajamento como sinal operacional de desempenho.")

    scatter_df = df.dropna(subset=["ieg", "ida", "ipv", "ano_referencia"]).copy()
    scatter_df["ano_referencia"] = scatter_df["ano_referencia"].astype(int).astype(str)
    year_color_map = {"2022": "#1D3557", "2023": "#457B9D", "2024": "#2A9D8F"}

    fig_scatter_ieg_ida = px.scatter(
        scatter_df,
        x="ieg",
        y="ida",
        color="ano_referencia",
        opacity=0.55,
        title="Relacao entre IEG (engajamento) e IDA (desempenho academico)",
        labels={
            "ieg": "IEG - indicador de engajamento",
            "ida": "IDA - indicador de desempenho academico",
            "ano_referencia": "Ano de referencia",
        },
        color_discrete_map=year_color_map,
        hover_data=["ra", "fase_programa", "pedra_ano"],
    )
    for year_value, part in scatter_df.groupby("ano_referencia"):
        if len(part) < 2:
            continue
        slope, intercept = np.polyfit(part["ieg"], part["ida"], 1)
        x_line = np.linspace(part["ieg"].min(), part["ieg"].max(), 80)
        y_line = slope * x_line + intercept
        fig_scatter_ieg_ida.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"Tendencia {year_value} (IEG x IDA)",
                line={"dash": "dash", "width": 2, "color": year_color_map.get(year_value, "#333333")},
            )
        )
    st.plotly_chart(fig_scatter_ieg_ida, use_container_width=True)
    st.caption("A inclinação positiva das linhas de tendência reforça a associação entre engajamento e resultado acadêmico.")

    fig_scatter_ieg_ipv = px.scatter(
        scatter_df,
        x="ieg",
        y="ipv",
        color="ano_referencia",
        opacity=0.55,
        title="Relacao entre IEG (engajamento) e IPV (ponto de virada)",
        labels={
            "ieg": "IEG - indicador de engajamento",
            "ipv": "IPV - indicador de ponto de virada",
            "ano_referencia": "Ano de referencia",
        },
        color_discrete_map=year_color_map,
        hover_data=["ra", "fase_programa", "pedra_ano"],
    )
    for year_value, part in scatter_df.groupby("ano_referencia"):
        if len(part) < 2:
            continue
        slope, intercept = np.polyfit(part["ieg"], part["ipv"], 1)
        x_line = np.linspace(part["ieg"].min(), part["ieg"].max(), 80)
        y_line = slope * x_line + intercept
        fig_scatter_ieg_ipv.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"Tendencia {year_value} (IEG x IPV)",
                line={"dash": "dash", "width": 2, "color": year_color_map.get(year_value, "#333333")},
            )
        )
    st.plotly_chart(fig_scatter_ieg_ipv, use_container_width=True)
    st.caption("O padrão também aparece para IPV, indicando que engajamento antecede pontos de virada.")


def render_q4(df: pd.DataFrame) -> None:
    st.subheader("Q4 - IAA: coerência com IDA e IEG")
    coherence = df.dropna(subset=["iaa", "ida", "ieg"]).copy()
    coherence["media_objetiva_desempenho"] = coherence[["ida", "ieg"]].mean(axis=1)
    coherence["desvio_autoavaliacao"] = coherence["iaa"] - coherence["media_objetiva_desempenho"]

    coherence_plot = coherence.copy()
    coherence_plot["ano_referencia"] = coherence_plot["ano_referencia"].astype(int).astype(str)
    fig_coherence = px.histogram(
        coherence_plot,
        x="desvio_autoavaliacao",
        color="ano_referencia",
        barmode="overlay",
        nbins=40,
        opacity=0.60,
        title="Distribuicao do desvio entre IAA e media objetiva (IDA, IEG)",
        labels={
            "desvio_autoavaliacao": "Desvio da autoavaliacao (IAA - media de IDA e IEG)",
            "ano_referencia": "Ano de referencia",
        },
        color_discrete_sequence=["#1D3557", "#457B9D", "#2A9D8F"],
    )
    fig_coherence.add_vline(x=0, line_dash="dash", line_color="#264653")
    st.plotly_chart(fig_coherence, use_container_width=True)
    st.caption("Distribuições deslocadas para a direita indicam superestimação; para a esquerda, subestimação.")


def render_q5_q6(df_long: pd.DataFrame) -> None:
    st.subheader("Q5 e Q6 - IPS/IPP: antecedência e confirmação com IAN")

    prior = df_long[df_long["ano_base"].isin([2022, 2023])].copy()
    prior = prior.dropna(subset=["ida", "ida_prox", "ieg", "ieg_prox", "ips", "ipp", "ian", "ian_prox"])
    if prior.empty:
        st.info("Sem amostra suficiente para Q5/Q6.")
        return

    prior["queda_ida_relevante"] = (prior["delta_ida_prox"] <= -1.0).astype(int)
    prior["delta_ieg_prox"] = prior["ieg_prox"] - prior["ieg"]
    prior["queda_ieg_relevante"] = (prior["delta_ieg_prox"] <= -1.0).astype(int)
    prior["queda_ian_relevante"] = (prior["delta_ian_prox"] <= -1.0).astype(int)

    risk_summary = []
    for target_col, target_name in [
        ("queda_ida_relevante", "Queda relevante de IDA (>= 1 ponto)"),
        ("queda_ieg_relevante", "Queda relevante de IEG (>= 1 ponto)"),
        ("queda_ian_relevante", "Queda relevante de IAN (>= 1 ponto)"),
    ]:
        grouped = prior.groupby(target_col)[["ips", "ipp"]].mean()
        if {0, 1}.issubset(set(grouped.index)):
            risk_summary.append(
                {
                    "evento_futuro": target_name,
                    "ips_sem_evento": grouped.loc[0, "ips"],
                    "ips_com_evento": grouped.loc[1, "ips"],
                    "diferenca_ips": grouped.loc[1, "ips"] - grouped.loc[0, "ips"],
                    "ipp_sem_evento": grouped.loc[0, "ipp"],
                    "ipp_com_evento": grouped.loc[1, "ipp"],
                    "diferenca_ipp": grouped.loc[1, "ipp"] - grouped.loc[0, "ipp"],
                }
            )

    risk_summary_df = pd.DataFrame(risk_summary)
    if not risk_summary_df.empty:
        st.caption("Diferenças médias de IPS/IPP por evento futuro foram calculadas conforme o notebook de EDA.")

    prior["ips_baixo"] = (prior["ips"] <= prior["ips"].quantile(0.25)).astype(int)
    prior["ipp_baixo"] = (prior["ipp"] <= prior["ipp"].quantile(0.25)).astype(int)

    pattern_table = (
        prior.groupby(["ips_baixo", "ipp_baixo"], as_index=False)
        .agg(
            alunos=("ra", "nunique"),
            prob_queda_ida=("queda_ida_relevante", "mean"),
            prob_queda_ieg=("queda_ieg_relevante", "mean"),
            prob_queda_ian=("queda_ian_relevante", "mean"),
        )
        .sort_values(["ips_baixo", "ipp_baixo"])
    )
    for col in ["prob_queda_ida", "prob_queda_ieg", "prob_queda_ian"]:
        pattern_table[col] = pattern_table[col] * 100

    pattern_table["perfil_ips_ipp"] = np.select(
        [
            (pattern_table["ips_baixo"] == 1) & (pattern_table["ipp_baixo"] == 1),
            (pattern_table["ips_baixo"] == 1) & (pattern_table["ipp_baixo"] == 0),
            (pattern_table["ips_baixo"] == 0) & (pattern_table["ipp_baixo"] == 1),
            (pattern_table["ips_baixo"] == 0) & (pattern_table["ipp_baixo"] == 0),
        ],
        [
            "IPS baixo + IPP baixo",
            "IPS baixo + IPP nao baixo",
            "IPS nao baixo + IPP baixo",
            "IPS nao baixo + IPP nao baixo",
        ],
        default="Perfil nao classificado",
    )

    pattern_plot = pattern_table.melt(
        id_vars=["perfil_ips_ipp", "alunos"],
        value_vars=["prob_queda_ida", "prob_queda_ieg", "prob_queda_ian"],
        var_name="evento",
        value_name="probabilidade",
    )
    event_name_map = {
        "prob_queda_ida": "Queda relevante em IDA (>= 1 ponto)",
        "prob_queda_ieg": "Queda relevante em IEG (>= 1 ponto)",
        "prob_queda_ian": "Queda relevante em IAN (>= 1 ponto)",
    }
    pattern_plot["evento"] = pattern_plot["evento"].map(event_name_map)
    pattern_plot["rotulo"] = pattern_plot["probabilidade"].map(lambda v: f"{v:.1f}%")
    profile_order = [
        "IPS nao baixo + IPP nao baixo",
        "IPS baixo + IPP nao baixo",
        "IPS nao baixo + IPP baixo",
        "IPS baixo + IPP baixo",
    ]
    pattern_plot["perfil_ips_ipp"] = pd.Categorical(pattern_plot["perfil_ips_ipp"], categories=profile_order, ordered=True)
    pattern_plot = pattern_plot.sort_values(["perfil_ips_ipp", "evento"])
    fig_pattern = px.bar(
        pattern_plot,
        x="perfil_ips_ipp",
        y="probabilidade",
        color="evento",
        barmode="group",
        text="rotulo",
        title="Q5 - Probabilidade de queda no ano seguinte por perfil IPS/IPP",
        labels={
            "perfil_ips_ipp": "Perfil combinado de IPS e IPP no ano base",
            "probabilidade": "Probabilidade de queda no ano seguinte (%)",
            "evento": "Evento futuro",
        },
        color_discrete_map={
            "Queda relevante em IDA (>= 1 ponto)": "#1D3557",
            "Queda relevante em IEG (>= 1 ponto)": "#2A9D8F",
            "Queda relevante em IAN (>= 1 ponto)": "#E76F51",
        },
    )
    fig_pattern.update_traces(textposition="outside")
    fig_pattern.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig_pattern, use_container_width=True)
    st.caption("O perfil combinado IPS baixo + IPP baixo concentra maior risco de deterioração no ano seguinte.")

    prior["perfil_ipp_clinico"] = np.where(prior["ipp"] <= 7, "IPP fragil (<= 7)", "IPP adequado (> 7)")
    prior["perfil_ian_clinico"] = np.where(prior["ian"] <= 5, "IAN com defasagem alta (<= 5)", "IAN sem defasagem alta (> 5)")

    ipp_ian_matrix = pd.crosstab(prior["perfil_ipp_clinico"], prior["perfil_ian_clinico"], normalize="index") * 100

    contradiction_rate = (
        ((prior["ipp"] > 7) & (prior["ian"] <= 5)) | ((prior["ipp"] <= 7) & (prior["ian"] > 5))
    ).mean() * 100
    st.caption(f"Taxa de contradição entre IPP e IAN: {contradiction_rate:.1f}%")

    fig_ipp_ian = px.imshow(
        ipp_ian_matrix,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale=SEQUENTIAL_SCALE,
        title="Q6 - Distribuicao percentual de IAN dentro de cada perfil de IPP",
        labels={
            "x": "Perfil de IAN",
            "y": "Perfil de IPP",
            "color": "Percentual de alunos (%)",
        },
    )
    st.plotly_chart(fig_ipp_ian, use_container_width=True)


def render_q7(df: pd.DataFrame) -> dict | None:
    st.subheader("Q7 - Drivers do IPV")

    ipv_features = ["ida", "ieg", "iaa", "ips", "ipp", "ian"]
    ipv_data = df.dropna(subset=ipv_features + ["ipv"]).copy()
    if len(ipv_data) < 100:
        st.info("Amostra insuficiente para estimar importância de IPV com robustez.")
        return None

    rf_ipv = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf_ipv.fit(ipv_data[ipv_features], ipv_data["ipv"])

    feature_name_map = {
        "ida": "IDA - desempenho academico",
        "ieg": "IEG - engajamento",
        "iaa": "IAA - autoavaliacao",
        "ips": "IPS - indice psicossocial",
        "ipp": "IPP - indice psicopedagogico",
        "ian": "IAN - adequacao de nivel",
    }
    ipv_importance = pd.DataFrame({"indicador": ipv_features, "importancia": rf_ipv.feature_importances_}).sort_values("importancia", ascending=False)
    ipv_importance["indicador"] = ipv_importance["indicador"].map(feature_name_map)

    fig_ipv_imp = px.bar(
        ipv_importance,
        x="importancia",
        y="indicador",
        orientation="h",
        text=ipv_importance["importancia"].map(lambda v: f"{v:.3f}"),
        title="Importancia relativa dos indicadores para explicar o IPV",
        labels={
            "importancia": "Importancia relativa",
            "indicador": "Indicador analisado",
        },
        color_discrete_sequence=["#2A9D8F"],
    )
    fig_ipv_imp.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    fig_ipv_imp.update_traces(textposition="outside")
    st.plotly_chart(fig_ipv_imp, use_container_width=True)
    st.caption("Quanto maior a barra, maior a influência do indicador na explicação do ponto de virada.")
    top = ipv_importance.iloc[0]
    return {
        "top_driver": str(top["indicador"]),
        "top_importance": float(top["importancia"]),
        "top3_share": float(ipv_importance.head(3)["importancia"].sum() * 100),
        "n": int(len(ipv_data)),
    }


def render_q8(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Q8 - Multidimensionalidade do INDE")

    multi_features = ["ida", "ieg", "ips", "ipp"]
    multi_df = df.dropna(subset=multi_features + ["inde_ano"]).copy()
    if len(multi_df) < 150:
        st.info("Amostra insuficiente para análise robusta da multidimensionalidade.")
        return pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(
        multi_df[multi_features],
        multi_df["inde_ano"],
        test_size=0.25,
        random_state=42,
    )

    rf_inde = RandomForestRegressor(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=6,
        random_state=42,
        n_jobs=-1,
    )
    rf_inde.fit(X_train, y_train)

    indicator_name_map = {
        "ida": "IDA - desempenho academico",
        "ieg": "IEG - engajamento",
        "ips": "IPS - indice psicossocial",
        "ipp": "IPP - indice psicopedagogico",
    }
    importance_inde = pd.DataFrame({"indicador": multi_features, "importancia": rf_inde.feature_importances_}).sort_values("importancia", ascending=False)
    importance_inde["indicador"] = importance_inde["indicador"].map(indicator_name_map)
    r2_train = rf_inde.score(X_train, y_train)
    r2_test = rf_inde.score(X_test, y_test)

    c1, c2 = st.columns(2)
    c1.metric("R2 treino", f"{r2_train:.3f}")
    c2.metric("R2 teste", f"{r2_test:.3f}")

    fig_import_inde = px.bar(
        importance_inde,
        x="importancia",
        y="indicador",
        orientation="h",
        text=importance_inde["importancia"].map(lambda v: f"{v:.3f}"),
        title="Importancia relativa para explicar o INDE",
        labels={
            "importancia": "Importancia relativa",
            "indicador": "Indicador analisado",
        },
        color_discrete_sequence=["#457B9D"],
    )
    fig_import_inde.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    fig_import_inde.update_traces(textposition="outside")
    st.plotly_chart(fig_import_inde, use_container_width=True)
    st.caption("IDA e IEG tendem a formar o núcleo explicativo do INDE, com IPS/IPP agregando refinamento.")

    profile_df = multi_df.copy()
    for col in multi_features:
        profile_df[f"{col}_faixa"] = pd.qcut(profile_df[col], q=3, labels=["Baixo", "Intermediario", "Alto"], duplicates="drop")

    profile_summary = (
        profile_df.groupby(["ida_faixa", "ieg_faixa", "ips_faixa", "ipp_faixa"], as_index=False)
        .agg(media_inde=("inde_ano", "mean"), alunos=("ra", "nunique"))
        .sort_values(["media_inde", "alunos"], ascending=[False, False])
    )
    profile_summary = profile_summary[profile_summary["alunos"] >= 15]
    st.caption("Top combinações de perfil foram calculadas conforme o notebook de EDA.")
    return profile_summary


def render_q9(df_long: pd.DataFrame) -> dict | None:
    st.subheader("Q9 - Baseline de previsão de risco")

    risk_features = ["ian", "ida", "ieg", "iaa", "ips", "ipp", "ipv", "inde_ano", "defasagem"]
    risk_base = df_long[df_long["ano_base"].isin([2022, 2023])].copy()
    risk_base["risco_defasagem_prox"] = ((risk_base["ian_prox"] <= 5) | (risk_base["delta_ian_prox"] <= -1)).astype(int)
    risk_model_df = risk_base.dropna(subset=risk_features + ["ian_prox"]).copy()

    if len(risk_model_df) < 200 or risk_model_df["risco_defasagem_prox"].nunique() != 2:
        st.info("Amostra insuficiente para baseline de risco nesta visão de EDA.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        risk_model_df[risk_features],
        risk_model_df["risco_defasagem_prox"],
        test_size=0.30,
        random_state=42,
        stratify=risk_model_df["risco_defasagem_prox"],
    )

    rf_risk = RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_risk.fit(X_train, y_train)

    prob_test = rf_risk.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, prob_test)
    pr_auc = average_precision_score(y_test, prob_test)
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC (teste)", f"{roc_auc:.3f}")
    c2.metric("PR-AUC (teste)", f"{pr_auc:.3f}")

    risk_importance = pd.DataFrame({"feature": risk_features, "importance": rf_risk.feature_importances_}).sort_values("importance", ascending=False)
    st.caption("Importâncias de variáveis e calibração foram calculadas conforme o notebook de EDA.")

    test_scored = X_test.copy()
    test_scored["prob_risco"] = prob_test
    test_scored["risco_real"] = y_test.values
    bins = pd.cut(test_scored["prob_risco"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    calibration = (
        test_scored.groupby(bins, as_index=False)
        .agg(alunos=("risco_real", "size"), taxa_real=("risco_real", "mean"), prob_media=("prob_risco", "mean"))
    )
    calibration["taxa_real"] = calibration["taxa_real"] * 100
    calibration["prob_media"] = calibration["prob_media"] * 100
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "top_feature": str(risk_importance.iloc[0]["feature"]),
        "top_feature_importance": float(risk_importance.iloc[0]["importance"]),
        "n": int(len(risk_model_df)),
    }


def render_q10(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Q10 - Efetividade por fases e coortes")

    program = df.dropna(subset=["ra", "ano_referencia", "pedra_ano"]).copy()
    program = program[program["pedra_ano"].isin(["Quartzo", "Agata", "Ametista", "Topazio"])]

    phase_metrics = (
        program.groupby(["pedra_ano", "ano_referencia"], as_index=False)
        .agg(
            alunos=("ra", "nunique"),
            media_inde=("inde_ano", "mean"),
            media_ida=("ida", "mean"),
            media_ieg=("ieg", "mean"),
            percentual_ian_defasado=("ian", lambda s: (s <= 5).mean() * 100),
        )
        .sort_values(["pedra_ano", "ano_referencia"])
    )

    phase_plot = phase_metrics.copy()
    phase_plot["ano_referencia"] = phase_plot["ano_referencia"].astype(int).astype(str)
    phase_order = ["Quartzo", "Agata", "Ametista", "Topazio"]
    phase_color_map = {
        "Quartzo": "#A4B3D3",
        "Agata": "#8092AC",
        "Ametista": "#08286F",
        "Topazio": "#00164D",
    }
    phase_plot["pedra_ano"] = pd.Categorical(phase_plot["pedra_ano"], categories=phase_order, ordered=True)
    phase_plot = phase_plot.sort_values(["ano_referencia", "pedra_ano"])

    fig_phase_inde = px.bar(
        phase_plot,
        x="ano_referencia",
        y="media_inde",
        color="pedra_ano",
        barmode="group",
        text="media_inde",
        title="Efetividade por fase: evolucao do INDE",
        labels={
            "ano_referencia": "Ano de referencia",
            "media_inde": "INDE medio",
            "pedra_ano": "Fase do programa (pedra)",
        },
        category_orders={"pedra_ano": phase_order, "ano_referencia": ["2022", "2023", "2024"]},
        color_discrete_map=phase_color_map,
    )
    fig_phase_inde.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_phase_inde.update_xaxes(type="category")
    fig_phase_inde.update_yaxes(range=[0, phase_plot["media_inde"].max() + 1.0])
    st.plotly_chart(fig_phase_inde, use_container_width=True)

    fig_phase_risk = px.bar(
        phase_plot,
        x="ano_referencia",
        y="percentual_ian_defasado",
        color="pedra_ano",
        barmode="group",
        text="percentual_ian_defasado",
        title="Efetividade por fase: risco de defasagem (IAN <= 5)",
        labels={
            "ano_referencia": "Ano de referencia",
            "percentual_ian_defasado": "Alunos com IAN <= 5 (%)",
            "pedra_ano": "Fase do programa (pedra)",
        },
        category_orders={"pedra_ano": phase_order, "ano_referencia": ["2022", "2023", "2024"]},
        color_discrete_map=phase_color_map,
    )
    fig_phase_risk.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_phase_risk.update_xaxes(type="category")
    fig_phase_risk.update_yaxes(range=[0, phase_plot["percentual_ian_defasado"].max() + 8])
    st.plotly_chart(fig_phase_risk, use_container_width=True)
    st.caption("Comparar nível de INDE com risco de defasagem ajuda a identificar fases com melhor equilíbrio.")

    cohort = program.dropna(subset=["inde_ano"]).copy()
    first_record = (
        cohort.sort_values(["ra", "ano_referencia"]).groupby("ra", as_index=False).first()[["ra", "pedra_ano"]].rename(columns={"pedra_ano": "pedra_inicial"})
    )
    cohort = cohort.merge(first_record, on="ra", how="left")

    cohort_evolution = (
        cohort.groupby(["pedra_inicial", "ano_referencia"], as_index=False)
        .agg(media_inde=("inde_ano", "mean"), alunos=("ra", "nunique"))
    )

    stone_order = ["Quartzo", "Agata", "Ametista", "Topazio"]
    cohort_evolution["pedra_inicial"] = pd.Categorical(cohort_evolution["pedra_inicial"], categories=stone_order, ordered=True)
    cohort_evolution = (
        cohort_evolution
        .sort_values(["pedra_inicial", "ano_referencia"])
    )

    cohort_plot = cohort_evolution.copy()
    cohort_plot["ano_referencia"] = cohort_plot["ano_referencia"].astype(int).astype(str)
    cohort_label_map = {
        "Quartzo": "Quartzo",
        "Agata": "Agata",
        "Ametista": "Ametista",
        "Topazio": "Topazio",
    }
    cohort_plot["coorte_inicial_rotulo"] = cohort_plot["pedra_inicial"].map(cohort_label_map)
    cohort_order_label = ["Quartzo", "Agata", "Ametista", "Topazio"]
    cohort_color_map = {
        "Quartzo": "#A4B3D3",
        "Agata": "#8092AC",
        "Ametista": "#08286F",
        "Topazio": "#00164D",
    }
    cohort_plot["coorte_inicial_rotulo"] = pd.Categorical(
        cohort_plot["coorte_inicial_rotulo"],
        categories=cohort_order_label,
        ordered=True,
    )
    cohort_plot = cohort_plot.sort_values(["ano_referencia", "coorte_inicial_rotulo"])
    fig_cohort = px.bar(
        cohort_plot,
        x="ano_referencia",
        y="media_inde",
        color="coorte_inicial_rotulo",
        barmode="group",
        text="media_inde",
        title="Evolucao do INDE por coorte da pedra inicial",
        labels={
            "ano_referencia": "Ano de referencia",
            "media_inde": "INDE medio",
            "coorte_inicial_rotulo": "Coorte inicial",
        },
        category_orders={"coorte_inicial_rotulo": cohort_order_label, "ano_referencia": ["2022", "2023", "2024"]},
        color_discrete_map=cohort_color_map,
    )
    fig_cohort.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_cohort.update_xaxes(type="category")
    fig_cohort.update_yaxes(range=[0, cohort_plot["media_inde"].max() + 1.0])
    st.plotly_chart(fig_cohort, use_container_width=True)
    st.caption("A leitura por coorte evidencia heterogeneidade de trajetória e orienta ações específicas.")
    return cohort_evolution


def render_q11(df: pd.DataFrame, cohort_evolution: pd.DataFrame) -> None:
    st.subheader("Q11 - Síntese executiva")
    ian_risk = (
        df.dropna(subset=["ian"]).groupby("ano_referencia")["ian"].apply(lambda s: (s <= 5).mean() * 100).sort_index()
    )
    ida_means = df.dropna(subset=["ida"]).groupby("ano_referencia")["ida"].mean().sort_index()
    delta_ida = ida_means.iloc[-1] - ida_means.iloc[0] if len(ida_means) >= 2 else np.nan
    global_corr = df[["ieg", "ida", "ipp", "ian", "ips", "ipv", "inde_ano"]].corr(method="spearman")
    corr_ieg_ida = global_corr.loc["ieg", "ida"]
    corr_ipp_ian = global_corr.loc["ipp", "ian"]

    if cohort_evolution.empty:
        cohort_delta_txt = "Sem dados suficientes para coorte."
    else:
        cohort_delta = (
            cohort_evolution.groupby("pedra_inicial")
            .apply(lambda x: x.sort_values("ano_referencia")["media_inde"].iloc[-1] - x.sort_values("ano_referencia")["media_inde"].iloc[0])
            .dropna()
        )
        cohort_delta_txt = cohort_delta.round(2).to_string()

    st.markdown(
        f"""
        **Resumo para decisão da ONG**
        - Risco de defasagem (IAN <= 5): 2022 = **{ian_risk.get(2022, np.nan):.1f}%**, 2023 = **{ian_risk.get(2023, np.nan):.1f}%**, 2024 = **{ian_risk.get(2024, np.nan):.1f}%**.
        - Tendência de IDA (2022 -> 2024): **{delta_ida:.2f}** pontos.
        - Relação IEG x IDA: **{corr_ieg_ida:.2f}**.
        - Relação IPP x IAN: **{corr_ipp_ian:.2f}**.

        **Implicações práticas**
        1. Manter estratégia de redução de defasagem para perfis críticos (IAN <= 5).
        2. Focar sustentação de desempenho acadêmico, evitando perda após ganhos iniciais.
        3. Usar engajamento e indicadores psicopedagógicos como gatilhos de intervenção precoce.
        4. Acompanhar fases e coortes com planos específicos, não apenas com ação média.

        ```
        {cohort_delta_txt}
        ```
        """
    )


def render_analise_exploratoria_tab(df: pd.DataFrame, df_long: pd.DataFrame) -> None:

    years = sorted([int(y) for y in df["ano_referencia"].dropna().unique().tolist()])
    # c1, c2, c3 = st.columns(3)
    # c1.metric("Registros analisados", f"{len(df):,}".replace(",", "."))
    # c2.metric("Alunos", f"{df['ra'].nunique():,}".replace(",", "."))
    # c3.metric("Janela temporal", " - ".join(map(str, years)))

    st.markdown(
        f"""
        <div class="eda-note">
        <strong>Contexto</strong><br>
        Esta seção apresenta uma análise exploratória sobre a evolução dos alunos
        atendidos pelo programa, correlacionando métricas de desempenho acadêmico, engajamento e fatores socioemocionais.
        O objetivo central é compreender a jornada de aprendizagem de forma ampla, desde o diagnóstico inicial de defasagem
        até a consolidação do desenvolvimento nas fases avançadas da metodologia (Quartzo, Ágata, Ametista e Topázio).
        <ul>
            <li>Registros analisados : {len(df)}</li>
            <li>Alunos: {df['ra'].nunique()}</li>
            <li>Janela temporal: {" - ".join(map(str, years))}</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Bloco 1 - Diagnóstico de desempenho")
    with st.expander("Relações estruturais entre indicadores", expanded=True):
        render_analysis_header(
            question="Quais indicadores caminham junto com o INDE?",
            importance="Define o eixo de priorização das análises seguintes.",
            approach="Correlação de Spearman entre indicadores acadêmicos, comportamentais e de risco.",
        )
        render_corr(df)
        inde_cols = ["inde_ano", "ida", "ieg", "ips", "ipp", "ian", "ipv"]
        inde_cols = [c for c in inde_cols if c in df.columns]
        if len(inde_cols) >= 3:
            corr = df[inde_cols].corr(method="spearman")
            inde_corr = corr.loc["inde_ano"].drop("inde_ano").sort_values(ascending=False)
            top1 = inde_corr.index[0]
            top2 = inde_corr.index[1] if len(inde_corr) > 1 else None
            top2_txt = f"{top2.upper()} ({inde_corr.iloc[1]:.2f})" if top2 is not None else "N/A"
            render_exec_note(
                message=f"Maior associação com INDE: {top1.upper()} ({inde_corr.iloc[0]:.2f}). Segundo fator: {top2_txt}.",
                implication="Priorizar frentes acadêmicas e de engajamento tende a mover o indicador global com maior eficiência.",
            )

    with st.expander("Q1 - IAN: perfil de defasagem e evolução temporal", expanded=True):
        render_analysis_header(
            question="A defasagem de aprendizagem está reduzindo ao longo do tempo?",
            importance="IAN é o termômetro principal de risco pedagógico.",
            approach="Série temporal de média, composição por faixa de risco e recortes por sexo/faixa etária.",
        )
        render_q1(df)
        ian_summary = (
            df.dropna(subset=["ian"])
            .groupby("ano_referencia", as_index=False)
            .agg(
                ian_medio=("ian", "mean"),
                percentual_defasagem_alta=("ian", lambda s: (s <= 5).mean() * 100),
            )
            .sort_values("ano_referencia")
        )
        if len(ian_summary) >= 2:
            ian_start = ian_summary.iloc[0]
            ian_end = ian_summary.iloc[-1]
            render_exec_note(
                message=(
                    f"IAN médio evoluiu de {ian_start['ian_medio']:.2f} para {ian_end['ian_medio']:.2f}. "
                    f"Defasagem alta (IAN <= 5) variou de {ian_start['percentual_defasagem_alta']:.1f}% "
                    f"para {ian_end['percentual_defasagem_alta']:.1f}%."
                ),
                implication="Há melhora estrutural, mas com grupos persistentes de risco que pedem ação segmentada.",
            )

    with st.expander("Q2 - IDA: melhora, estagnação ou queda", expanded=True):
        render_analysis_header(
            question="O desempenho acadêmico (IDA) está subindo de forma consistente?",
            importance="IDA resume aprendizagem e ajuda a medir retorno das intervenções.",
            approach="Comparação anual de média e distribuição para capturar tendência e volatilidade.",
        )
        render_q2(df)
        ida_summary = (
            df.dropna(subset=["ida"])
            .groupby("ano_referencia", as_index=False)["ida"]
            .mean()
            .rename(columns={"ida": "ida_media"})
            .sort_values("ano_referencia")
        )
        if len(ida_summary) >= 2:
            delta = ida_summary.iloc[-1]["ida_media"] - ida_summary.iloc[0]["ida_media"]
            trend = "melhora líquida" if delta > 0 else "queda líquida"
            render_exec_note(
                message=f"Variação acumulada no período: {delta:+.2f} pontos de IDA, com {trend}.",
                implication="A governança pedagógica precisa monitorar turmas/fases com oscilação para evitar regressão.",
            )

    with st.expander("Q3 - IEG x IDA e IPV", expanded=True):
        render_analysis_header(
            question="Engajamento explica desempenho e ponto de virada?",
            importance="Se a relação for forte, IEG pode atuar como sinal de intervenção precoce.",
            approach="Correlações anuais + dispersões com linha de tendência por ano.",
        )
        render_q3(df)
        rows = []
        for year, part in df.groupby("ano_referencia"):
            sub = part[["ieg", "ida", "ipv"]].dropna()
            if len(sub) < 10:
                continue
            rows.append(
                {
                    "ano": int(year),
                    "ieg_ida": sub["ieg"].corr(sub["ida"], method="spearman"),
                    "ieg_ipv": sub["ieg"].corr(sub["ipv"], method="spearman"),
                }
            )
        corr_df = pd.DataFrame(rows)
        if not corr_df.empty:
            render_exec_note(
                message=(
                    f"Correlação média IEG x IDA: {corr_df['ieg_ida'].mean():.2f}. "
                    f"Correlação média IEG x IPV: {corr_df['ieg_ipv'].mean():.2f}."
                ),
                implication="IEG pode ser usado como indicador de monitoramento contínuo para acionar suporte acadêmico.",
            )

    with st.expander("Q4 - IAA: coerência com indicadores objetivos", expanded=True):
        render_analysis_header(
            question="A autoavaliação (IAA) está alinhada ao desempenho observado?",
            importance="Desalinhamento pode gerar percepção incorreta de necessidade de apoio.",
            approach="Distribuição do desvio entre IAA e média de IDA/IEG com medida de super/subestimação.",
        )
        render_q4(df)
        coherence = df.dropna(subset=["iaa", "ida", "ieg"]).copy()
        if not coherence.empty:
            coherence["desvio"] = coherence["iaa"] - coherence[["ida", "ieg"]].mean(axis=1)
            over = (coherence["desvio"] > 0.5).mean() * 100
            under = (coherence["desvio"] < -0.5).mean() * 100
            render_exec_note(
                message=f"Superestimação relevante: {over:.1f}%. Subestimação relevante: {under:.1f}%.",
                implication="IAA agrega contexto socioemocional, mas decisão operacional deve combinar sinais objetivos.",
            )

    st.markdown("### Bloco 2 - Sinais antecedentes de risco")
    with st.expander("Q5 e Q6 - IPS/IPP: alerta antecipado e confirmação com IAN", expanded=True):
        render_analysis_header(
            question="IPS e IPP antecipam quedas futuras e confirmam risco pedagógico?",
            importance="Identificar risco cedo reduz custo e aumenta efetividade da intervenção.",
            approach="Análise longitudinal com eventos de queda no ano seguinte e matriz de coerência IPP x IAN.",
        )
        render_q5_q6(df_long)
        prior = df_long[df_long["ano_base"].isin([2022, 2023])].copy()
        prior = prior.dropna(subset=["ida", "ida_prox", "ieg", "ieg_prox", "ips", "ipp", "ian", "ian_prox"])
        if not prior.empty:
            prior["queda_ida_relevante"] = (prior["delta_ida_prox"] <= -1.0).astype(int)
            prior["ips_baixo"] = (prior["ips"] <= prior["ips"].quantile(0.25)).astype(int)
            prior["ipp_baixo"] = (prior["ipp"] <= prior["ipp"].quantile(0.25)).astype(int)
            base_rate = prior["queda_ida_relevante"].mean() * 100
            high_risk = prior[(prior["ips_baixo"] == 1) & (prior["ipp_baixo"] == 1)]
            high_rate = high_risk["queda_ida_relevante"].mean() * 100 if not high_risk.empty else np.nan
            contradiction_rate = (
                (((prior["ipp"] > 7) & (prior["ian"] <= 5)) | ((prior["ipp"] <= 7) & (prior["ian"] > 5))).mean() * 100
            )
            render_exec_note(
                message=(
                    f"Taxa base de queda relevante de IDA: {base_rate:.1f}%. "
                    f"No perfil IPS baixo + IPP baixo: {high_rate:.1f}%. "
                    f"Contradição entre IPP e IAN: {contradiction_rate:.1f}%."
                ),
                implication="IPS/IPP são bons sinais antecedentes, mas devem ser lidos junto ao histórico acadêmico.",
            )

    st.markdown("### Bloco 3 - Drivers e modelagem exploratória")
    with st.expander("Q7 - O que mais explica IPV", expanded=True):
        render_analysis_header(
            question="Quais variáveis mais influenciam o IPV?",
            importance="Ajuda a priorizar fatores de alavancagem do ponto de virada.",
            approach="Random Forest de regressão para importância relativa dos indicadores.",
        )
        q7_info = render_q7(df)
        if q7_info is not None:
            render_exec_note(
                message=(
                    f"Principal driver: {q7_info['top_driver']} ({q7_info['top_importance']*100:.1f}%). "
                    f"Top 3 drivers concentram {q7_info['top3_share']:.1f}% da explicação "
                    f"(amostra: {q7_info['n']} registros)."
                ),
                implication="Intervenções focadas nos drivers principais tendem a gerar maior ganho marginal no IPV.",
            )

    with st.expander("Q8 - Combinações que explicam melhor o INDE", expanded=True):
        render_analysis_header(
            question="Quais combinações de desempenho e comportamento resultam em melhor INDE?",
            importance="Permite desenhar perfis-alvo para políticas de acompanhamento.",
            approach="Modelo de regressão com variáveis-chave e análise de perfis por faixas (baixo/intermediário/alto).",
        )
        q8_profiles = render_q8(df)
        if isinstance(q8_profiles, pd.DataFrame) and not q8_profiles.empty:
            top_profile = q8_profiles.iloc[0]
            render_exec_note(
                message=(
                    f"Melhor combinação observada: IDA={top_profile['ida_faixa']}, IEG={top_profile['ieg_faixa']}, "
                    f"IPS={top_profile['ips_faixa']}, IPP={top_profile['ipp_faixa']}; "
                    f"INDE médio={top_profile['media_inde']:.2f} (n={int(top_profile['alunos'])})."
                ),
                implication="INDE é multidimensional, com eixo dominante acadêmico + engajamento na maior parte dos perfis.",
            )

    with st.expander("Q9 - Baseline de previsão de risco de defasagem", expanded=True):
        render_analysis_header(
            question="Qual a capacidade inicial de prever risco de defasagem no próximo ano?",
            importance="Oferece base para priorização preventiva de alunos.",
            approach="Baseline supervisionado com split treino/teste e leitura de ROC-AUC, PR-AUC e calibração.",
        )
        q9_info = render_q9(df_long)
        if q9_info is not None:
            render_exec_note(
                message=(
                    f"ROC-AUC={q9_info['roc_auc']:.3f} e PR-AUC={q9_info['pr_auc']:.3f}. "
                    f"Variável de maior peso: {q9_info['top_feature']} "
                    f"({q9_info['top_feature_importance']*100:.1f}%). "
                    f"Base modelada: {q9_info['n']} observações."
                ),
                implication="Há sinal preditivo útil para triagem inicial de risco, com espaço para calibração refinada.",
            )

    with st.expander("Q10 - Efetividade por fase e coorte", expanded=True):
        render_analysis_header(
            question="Quais fases/coortes mostram maior efetividade ao longo do tempo?",
            importance="Suporta alocação de recursos e réplica de práticas bem-sucedidas.",
            approach="Comparação anual por fase e coorte, com foco em INDE e risco de defasagem.",
        )
        cohort_evolution = render_q10(df)
        if isinstance(cohort_evolution, pd.DataFrame) and not cohort_evolution.empty:
            end_year = cohort_evolution["ano_referencia"].max()
            start_year = cohort_evolution["ano_referencia"].min()
            end_view = cohort_evolution[cohort_evolution["ano_referencia"] == end_year].sort_values("media_inde", ascending=False)
            best_phase = end_view.iloc[0]["pedra_inicial"]
            best_inde = end_view.iloc[0]["media_inde"]

            cohort_delta = (
                cohort_evolution.groupby("pedra_inicial")
                .apply(lambda x: x.sort_values("ano_referencia")["media_inde"].iloc[-1] - x.sort_values("ano_referencia")["media_inde"].iloc[0])
                .dropna()
            )
            strongest_growth = cohort_delta.sort_values(ascending=False).index[0] if not cohort_delta.empty else "N/A"
            strongest_growth_value = cohort_delta.max() if not cohort_delta.empty else np.nan

            render_exec_note(
                message=(
                    f"Melhor patamar de INDE no ano final ({int(end_year)}): {best_phase} ({best_inde:.2f}). "
                    f"Maior ganho entre {int(start_year)} e {int(end_year)}: "
                    f"{strongest_growth} ({strongest_growth_value:+.2f})."
                ),
                implication="O impacto do programa é heterogêneo; a gestão deve combinar estratégia comum com ação por fase/coorte.",
            )
        else:
            cohort_evolution = pd.DataFrame()

    with st.expander("Q11 - Síntese executiva e plano de ação", expanded=True):
        render_analysis_header(
            question="Qual a síntese final para decisão executiva?",
            importance="Consolida os principais achados para orientar plano de ação.",
            approach="Resumo integrado dos indicadores de risco, desempenho e evolução por coorte.",
        )
        render_q11(df, cohort_evolution)

