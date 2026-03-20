from __future__ import annotations

import html
import re

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
_graph_counter = 0

ABBREVIATION_FULL_NAME_MAP = {
    "INDE": "Índice de Desenvolvimento Educacional",
    "IAN": "Índice de Adequação de Nível",
    "IDA": "Indicador de Desempenho Acadêmico",
    "IEG": "Indicador de Engajamento",
    "IAA": "Indicador de Autoavaliação",
    "IPS": "Índice Psicossocial",
    "IPP": "Índice Psicopedagógico",
    "IPV": "Indicador de Ponto de Virada",
}


def expand_abbreviations(text: str | None) -> str | None:
    if text is None:
        return None
    expanded = str(text)
    for abbr, full_name in ABBREVIATION_FULL_NAME_MAP.items():
        expanded = re.sub(rf"\b{abbr}\b(?!\s*\()", f"{abbr} ({full_name})", expanded)
    return expanded


def _expand_sequence_values(values):
    if values is None:
        return values
    if isinstance(values, str):
        return expand_abbreviations(values)
    if isinstance(values, np.ndarray):
        if values.dtype.kind in {"U", "S", "O"}:
            return np.array([expand_abbreviations(v) if isinstance(v, str) else v for v in values], dtype=object)
        return values
    if isinstance(values, (list, tuple, pd.Series)):
        return [expand_abbreviations(v) if isinstance(v, str) else v for v in values]
    return values


def apply_full_names_to_figure(fig: go.Figure) -> None:
    if fig.layout.title is not None and fig.layout.title.text:
        fig.layout.title.text = expand_abbreviations(str(fig.layout.title.text))

    if fig.layout.legend and fig.layout.legend.title and fig.layout.legend.title.text:
        fig.layout.legend.title.text = expand_abbreviations(str(fig.layout.legend.title.text))

    for layout_key in fig.layout:
        if isinstance(layout_key, str) and (layout_key.startswith("xaxis") or layout_key.startswith("yaxis")):
            axis_cfg = fig.layout[layout_key]
            if axis_cfg and axis_cfg.title and axis_cfg.title.text:
                axis_cfg.title.text = expand_abbreviations(str(axis_cfg.title.text))

    if fig.layout.coloraxis and fig.layout.coloraxis.colorbar and fig.layout.coloraxis.colorbar.title and fig.layout.coloraxis.colorbar.title.text:
        fig.layout.coloraxis.colorbar.title.text = expand_abbreviations(str(fig.layout.coloraxis.colorbar.title.text))

    if fig.layout.annotations:
        for ann in fig.layout.annotations:
            if ann.text:
                ann.text = expand_abbreviations(str(ann.text))

    for trace in fig.data:
        if hasattr(trace, "name") and isinstance(trace.name, str):
            trace.name = expand_abbreviations(trace.name)
        if hasattr(trace, "hovertemplate") and isinstance(trace.hovertemplate, str):
            trace.hovertemplate = expand_abbreviations(trace.hovertemplate)
        if hasattr(trace, "texttemplate") and isinstance(trace.texttemplate, str):
            trace.texttemplate = expand_abbreviations(trace.texttemplate)
        if hasattr(trace, "x"):
            trace.x = _expand_sequence_values(trace.x)
        if hasattr(trace, "y"):
            trace.y = _expand_sequence_values(trace.y)
        if hasattr(trace, "text"):
            trace.text = _expand_sequence_values(trace.text)
        if hasattr(trace, "hovertext"):
            trace.hovertext = _expand_sequence_values(trace.hovertext)


def show_subheader(text: str) -> None:
    st.subheader(expand_abbreviations(text))


def show_caption(text: str) -> None:
    st.caption(expand_abbreviations(text))


def inject_eda_report_theme() -> None:
    st.markdown(
        """
        <style>
        .eda-context {
            background: #FDF7EE;
            border: 1px solid #FAC775;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin: 0.35rem 0 1rem 0;
            color: #3D3D3A;
            line-height: 1.58;
        }
        .eda-context strong {
            color: #85500B;
            font-size: 0.94rem;
            letter-spacing: 0.01em;
        }
        .eda-context ul {
            margin: 0.55rem 0 0 1.1rem;
            padding: 0;
        }
        .eda-context li {
            margin-bottom: 0.2rem;
        }
        .eda-header-question {
            color: #1A1A1A;
            font-size: 0.94rem;
            margin: 0.2rem 0 0.25rem 0;
            line-height: 1.5;
        }
        .eda-header-meta {
            color: #3D3D3A;
            font-size: 0.88rem;
            line-height: 1.52;
            margin-bottom: 0.22rem;
        }
        .eda-header-label {
            color: #3D3D3A;
            font-weight: 700;
        }
        .eda-header-rule {
            height: 1px;
            background: #D3D1C7;
            margin: 0.45rem 0 0.6rem 0;
        }
        .eda-note {
            border-radius: 12px;
            padding: 0.95rem 1.05rem;
            margin: 0.58rem 0 1rem 0;
            line-height: 1.58;
            color: #3D3D3A;
            border: 1px solid #D3D1C7;
        }
        .eda-note--analysis {
            background: #EBF7F2;
            border-color: #9FE1CB;
            box-shadow: inset 0 0 0 1px rgba(159, 225, 203, 0.35);
        }
        .eda-note--conclusion {
            background: #FDF7EE;
            border-color: #FAC775;
            box-shadow: inset 0 0 0 1px rgba(250, 199, 117, 0.28);
        }
        .eda-note strong {
            color: #0F6E56;
            font-size: 0.92rem;
            letter-spacing: 0.01em;
        }
        .eda-note--conclusion strong {
            color: #85500B;
        }
        .eda-note-divider {
            border-top: 1px solid #D3D1C7;
            margin: 0.52rem 0 0.62rem 0;
        }
        .eda-note-ref {
            color: #73726C;
            font-size: 0.79rem;
            margin-top: 0.58rem;
            line-height: 1.42;
        }
        .eda-note-ref .eda-note-ref-label {
            color: #73726C;
            font-weight: 700;
        }
        .eda-inline-table-wrap {
            width: 100%;
            overflow-x: auto;
            margin: 0.48rem 0 0.54rem 0;
        }
        .eda-inline-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.83rem;
            color: #3D3D3A;
            background: #FFFFFF;
            border: 1px solid #D3D1C7;
            border-radius: 8px;
            overflow: hidden;
        }
        .eda-inline-table th {
            text-align: left;
            font-weight: 700;
            color: #264653;
            background: #F3F5F4;
            border-bottom: 1px solid #D3D1C7;
            padding: 0.34rem 0.46rem;
            white-space: nowrap;
        }
        .eda-inline-table td {
            padding: 0.33rem 0.46rem;
            border-top: 1px solid #ECEBE4;
            vertical-align: top;
            white-space: nowrap;
        }
        .eda-inline-table tr:first-child td {
            border-top: none;
        }
        .eda-block-title {
            color: #73726C;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin: 0.1rem 0 0.55rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def reset_graph_counter() -> None:
    global _graph_counter
    _graph_counter = 0


def plotly_chart_numbered(
    fig: go.Figure,
    note: str | None = None,
    analysis: str | None = None,
    practical_meaning: str | None = None,
    *,
    apply_full_names: bool = True,
    prefix_title: bool = True,
    analysis_title: str = "Análise do gráfico",
    practical_title: str = "O que isso significa na prática",
    expand_note_text: bool = False,
) -> int:
    if apply_full_names:
        apply_full_names_to_figure(fig)

    global _graph_counter
    _graph_counter += 1
    graph_number = _graph_counter

    if prefix_title:
        current_title = ""
        if fig.layout.title is not None and fig.layout.title.text:
            current_title = str(fig.layout.title.text)
        title_prefix = f"Gráfico {graph_number}"
        fig.update_layout(title=f"{title_prefix} - {current_title}" if current_title else title_prefix)

    st.plotly_chart(fig, use_container_width=True)
    if note:
        show_caption(f"Referência: Gráfico {graph_number}. {note}")
    else:
        show_caption(f"Referência: Gráfico {graph_number}.")
    if analysis or practical_meaning:
        render_graph_note(
            analysis=analysis or "Leitura descritiva não informada.",
            practical_meaning=practical_meaning or "Interpretação prática não informada.",
            analysis_title=analysis_title,
            practical_title=practical_title,
            expand_text=expand_note_text,
        )
    return graph_number




def format_graph_refs(graph_refs: list[int] | None) -> str:
    if not graph_refs:
        return "Sem base gráfica."
    unique_refs = sorted(set(graph_refs))
    return ", ".join([f"Gráfico {n}" for n in unique_refs])


def render_analysis_header(question: str, importance: str, approach: str) -> None:
    q_text = html.escape(str(expand_abbreviations(question)))
    importance_text = html.escape(str(expand_abbreviations(importance)))
    approach_text = html.escape(str(expand_abbreviations(approach)))
    st.markdown(
        f"""
        <div class="eda-header-question">
            <span class="eda-header-label">Pergunta orientadora:</span> {q_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.markdown(
        f"""
        <div class="eda-header-meta">
            <span class="eda-header-label">Por que importa:</span> {importance_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div class="eda-header-meta">
            <span class="eda-header-label">Como foi analisado:</span> {approach_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eda-header-rule"></div>', unsafe_allow_html=True)


def format_card_text(text: str, *, expand_text: bool = False) -> str:
    raw_text = expand_abbreviations(text) if expand_text else text
    return _format_text_with_inline_tables(raw_text)


def _is_tabular_line(line: str) -> bool:
    return line.count("|") >= 2 and any(cell.strip() for cell in line.split("|"))


def _parse_table_rows(table_lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if len(cells) >= 2:
            rows.append(cells)
    return rows


def _render_inline_table(rows: list[list[str]]) -> str:
    max_cols = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]

    header_cells = "".join(f"<th>{html.escape(cell)}</th>" for cell in normalized_rows[0])
    body_rows = []
    for row in normalized_rows[1:]:
        cells_html = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells_html}</tr>")

    tbody_html = f"<tbody>{''.join(body_rows)}</tbody>" if body_rows else ""
    return (
        '<div class="eda-inline-table-wrap">'
        f'<table class="eda-inline-table"><thead><tr>{header_cells}</tr></thead>{tbody_html}</table>'
        "</div>"
    )


def _format_text_with_inline_tables(raw_text: str) -> str:
    lines = raw_text.splitlines()
    output_parts: list[str] = []
    text_buffer: list[str] = []

    def flush_text_buffer() -> None:
        if not text_buffer:
            return
        escaped = html.escape("\n".join(text_buffer)).replace("\n", "<br>")
        output_parts.append(escaped)
        text_buffer.clear()

    i = 0
    while i < len(lines):
        if _is_tabular_line(lines[i]):
            j = i
            while j < len(lines) and _is_tabular_line(lines[j]):
                j += 1
            table_rows = _parse_table_rows(lines[i:j])
            if len(table_rows) >= 2:
                flush_text_buffer()
                output_parts.append(_render_inline_table(table_rows))
            else:
                text_buffer.extend(lines[i:j])
            i = j
            continue

        text_buffer.append(lines[i])
        i += 1

    flush_text_buffer()
    return "".join(output_parts)


def render_exec_note(
    message: str,
    implication: str,
    graph_refs: list[int] | None = None,
    *,
    base_visual_text: str | None = None,
    expand_text: bool = False,
) -> None:
    refs_text = base_visual_text or format_graph_refs(graph_refs)
    st.markdown(
        f"""
        <div class="eda-note eda-note--conclusion">
        <strong>Conclusão</strong><br>
        {format_card_text(message, expand_text=expand_text)}
        <div class="eda-note-divider"></div>
        <strong>Implicação prática</strong><br>
        {format_card_text(implication, expand_text=expand_text)}
        <div class="eda-note-ref">
            <span class="eda-note-ref-label">Base visual</span><br>
            {format_card_text(refs_text)}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_graph_note(
    analysis: str,
    practical_meaning: str,
    *,
    analysis_title: str = "Análise do gráfico",
    practical_title: str = "O que isso significa na prática",
    expand_text: bool = False,
) -> None:
    st.markdown(
        f"""
        <div class="eda-note eda-note--analysis">
        <strong>{format_card_text(analysis_title)}</strong><br>
        {format_card_text(analysis, expand_text=expand_text)}
        <div class="eda-note-divider"></div>
        <strong>{format_card_text(practical_title)}</strong><br>
        {format_card_text(practical_meaning, expand_text=expand_text)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_corr(df: pd.DataFrame) -> int:
    show_subheader("Matriz de correlação")
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
    graph_ref = plotly_chart_numbered(
        fig_corr,
        "Cores mais intensas indicam relações mais fortes.",
        analysis=(
            "As correlações mais fortes concentram-se no bloco acadêmico: IDA lidera com ro=0,78 em relação ao INDE, "
            "seguida por IEG (0,69) e IPV (0,70). As notas de Matemática, Português e Inglês correlacionam fortemente "
            "com IDA (0,83-0,85), revelando que o caminho até o INDE passa por IDA como intermediário, não pelas notas "
            "diretamente. IAN e Defasagem escolar apresentam correlação elevada entre si (0,93), mas ambas têm impacto "
            "fraco sobre o INDE (aprox. 0,41-0,43). IPS e IAA são dimensões praticamente isoladas do restante da estrutura."
        ),
        practical_meaning=(
            "O INDE não responde diretamente às notas, responde ao IDA, que é composto por elas. Monitorar só a nota sem "
            "acompanhar o IDA como variável de controle pode gerar a ilusão de melhora sem efeito sistêmico. Além disso, "
            "IPV correlaciona com IDA (0,56) e IEG (0,55), sugerindo que alunos em ponto de virada já apresentam "
            "deterioração cruzada, tornando o IPV um sinal de alerta precoce, não apenas um marcador de risco tardio."
        ),
        apply_full_names=False,
        prefix_title=False,
    )
    return graph_ref


def render_q1(df: pd.DataFrame) -> dict[str, int]:
    show_subheader("Q1 - IAN: defasagem e evolução temporal")
    graph_refs: dict[str, int] = {}
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
    graph_refs["ian_linha"] = plotly_chart_numbered(
        fig_ian_line,
        "O IAN médio mostra tendência de melhoria no período.",
        analysis=(
            "O IAN médio sobe de forma consistente: 6,43 em 2022, 7,24 em 2023 e 7,68 em 2024, ganho acumulado de "
            "+1,25 pontos em dois anos. O ritmo desacelera (+0,81 de 2022-2023 e +0,44 de 2023-2024).\n\n"
            "D = Fase Efetiva - Fase Ideal | IAN: 10 (em fase) / 5 (moderada) / 2,5 (severa).\n\n"
            "O IAN não é uma nota contínua, é variável discreta com apenas 3 valores possíveis. A média subindo é uma "
            "mudança na distribuição entre esses grupos, não uma curva suave."
        ),
        practical_meaning=(
            "A melhora do IAN indica que mais alunos estão progredindo dentro do nível esperado. No entanto, a "
            "desaceleração entre 2023 e 2024 merece atenção: ganhos fáceis (recuperação de casos mais leves) tendem a "
            "ocorrer primeiro. A média crescente pode estar mascarando uma cauda de alunos que não avançaram."
        ),
    )

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
    graph_refs["ian_mix"] = plotly_chart_numbered(
        fig_ian_mix,
        "A redução da faixa de defasagem alta e o avanço da faixa adequada são sinais de evolução estrutural.",
    )

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
    graph_refs["ian_genero"] = plotly_chart_numbered(
        fig_ian_gender,
        "O recorte por sexo ajuda a identificar grupos com maior necessidade de apoio pedagógico.",
    )

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
        graph_refs["ian_variacao_genero"] = plotly_chart_numbered(
            fig_gender_variation,
            "A variação entre períodos mostra ritmo de melhora distinto entre os sexos.",
        )

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
    graph_refs["ian_faixa_etaria"] = plotly_chart_numbered(
        fig_ian_age,
        "A segmentação por faixa etária orienta ações direcionadas por ciclo de aprendizagem.",
    )
    return graph_refs


def render_q2(df: pd.DataFrame) -> dict[str, int]:
    show_subheader("Q2 - IDA: melhora, estagnação ou queda")
    graph_refs: dict[str, int] = {}
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
    graph_refs["ida_linha"] = plotly_chart_numbered(
        fig_ida_line,
        "A série anual revela ganho inicial e necessidade de sustentação no último ano.",
        analysis=(
            "O IDA médio apresenta um padrão de pico e retração: sobe de 6,10 em 2022 para 6,66 em 2023 (+0,56), "
            "depois recua para 6,35 em 2024 (-0,31). O padrão não é desaceleração, é reversão parcial.\n\n"
            "IDA = (Nota Matemática + Nota Português + Nota Inglês) / 3.\n\n"
            "Com peso de 20% no INDE (Fases 0-7) e 40% na Fase 8, uma queda de 0,31 no IDA se traduz em -0,062 no INDE "
            "nas fases regulares e -0,124 na Fase 8. Como IDA é a alavanca de maior correlação com o INDE (ro=0,78), "
            "essa reversão tem impacto quantificável e imediato no índice global."
        ),
        practical_meaning=(
            "O programa gerou melhora inicial, mas não conseguiu consolidar esse patamar. A questão central não é por que "
            "caiu, mas: o que mudou entre 2023 e 2024 nas condições de aprendizagem ou na intensidade das intervenções? "
            "Alunos na Fase 8 têm peso de IDA dobrado (40%): para esse grupo, a mesma queda de 0,31 gera o dobro do "
            "impacto no INDE."
        ),
    )

    ida_dist = df.dropna(subset=["ida", "ano_referencia"]).copy()
    ida_dist["ano_referencia"] = ida_dist["ano_referencia"].astype(int).astype(str)
    fig_ida_violin = px.violin(
        ida_dist,
        x="ano_referencia",
        y="ida",
        color="ano_referencia",
        box=True,
        points="outliers",
        title="",
        labels={
            "ano_referencia": "Ano de referencia",
            "ida": "IDA - desempenho academico",
        },
        color_discrete_map={"2022": "#1F3E67", "2023": "#3F7BA4", "2024": "#1B9C8A"},
        template="ggplot2",
    )
    fig_ida_violin.update_traces(
        opacity=0.58,
        line_width=2,
        marker={"size": 5, "opacity": 0.7},
        width=0.78,
    )
    fig_ida_violin.update_layout(
        showlegend=False,
        plot_bgcolor="#E5E5E5",
        paper_bgcolor="#E5E5E5",
        margin={"l": 6, "r": 8, "t": 8, "b": 10},
    )
    fig_ida_violin.update_xaxes(type="category", title_text="Ano de referencia", gridcolor="#C7D1DD")
    fig_ida_violin.update_yaxes(title_text="", showticklabels=False, gridcolor="#C7D1DD")
    graph_refs["ida_distribuicao"] = plotly_chart_numbered(
        fig_ida_violin,
        "A distribuição mostra heterogeneidade entre estudantes, além da média global.",
    )
    return graph_refs


def render_q3(df: pd.DataFrame) -> dict[str, int] | None:
    show_subheader("Q3 - IEG x IDA e IPV")
    graph_refs: dict[str, int] = {}
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
                "alunos_validos": len(sub),
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
        title="",
        labels={
            "correlacao_spearman": "rho (Spearman)",
            "Ano de referencia": "Ano de referencia",
        },
        color_discrete_map={
            "IEG x IDA": "#7B3F98",
            "IEG x IPV": "#24A17A",
            "IDA x IPV": "#3F6FAE",
        },
        template="ggplot2",
    )
    fig_corr_q3.update_traces(textposition="outside", textfont={"size": 18, "color": "#4E6C9E"}, cliponaxis=False)
    fig_corr_q3.update_layout(
        showlegend=False,
        bargap=0.2,
        plot_bgcolor="#E5E5E5",
        paper_bgcolor="#E5E5E5",
        margin={"l": 6, "r": 10, "t": 8, "b": 10},
    )
    fig_corr_q3.update_xaxes(type="category", title_text="Ano de referencia", gridcolor="#C7D1DD")
    fig_corr_q3.update_yaxes(
        range=[0, max(0.7, corr_long["correlacao_spearman"].max() + 0.06)],
        title_text="",
        showticklabels=False,
        gridcolor="#C7D1DD",
    )
    fig_corr_q3.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#FF6B57",
        line_width=2,
        annotation_text="Correlação moderada (0,5)",
        annotation_position="top right",
        annotation_font={"color": "#000000", "size": 16},
    )
    graph_refs["q3_correlacoes"] = plotly_chart_numbered(
        fig_corr_q3,
        "As correlações positivas sustentam o uso do engajamento como sinal operacional de desempenho.",
        analysis=(
            "As três correlações se mantêm positivas e moderadas ao longo de todo o período, mas com trajetórias distintas:\n\n"
            "Par de indicadores | 2022 | 2023 | 2024 | Delta 22-24\n"
            "IEG x IDA | 0,51 | 0,45 | 0,52 | +0,01\n"
            "IEG x IPV | 0,54 | 0,49 | 0,55 | +0,01\n"
            "IDA x IPV | 0,62 | 0,55 | 0,55 | -0,07\n\n"
            "Destaque para 2023: todas as correlações caíram simultaneamente e se recuperaram em 2024, exceto IDA x IPV, "
            "que ficou estacionada. Isso sugere que 2023 foi um ano atípico na estrutura relacional entre os indicadores, "
            "não apenas nos valores médios."
        ),
        practical_meaning=(
            "IEG mede tarefas realizadas e registradas (lição de casa, atividades acadêmicas, voluntariado). IPV é uma "
            "avaliação longitudinal feita por educadores. A correlação IEG x IPV de 0,54-0,55 indica que os avaliadores, "
            "ao julgar quem está em ponto de virada, estão capturando o mesmo comportamento que o IEG registra "
            "objetivamente, validando o IEG como proxy rastreável do IPV. Como IEG vale 20% no INDE (igual ao IDA) e "
            "correlaciona com IPV (mais 20%), monitorar o IEG equivale a monitorar indiretamente 40% do INDE ao mesmo tempo."
        ),
    )

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
            "ieg": "IEG",
            "ida": "IDA",
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
    fig_scatter_ieg_ida.update_layout(
        margin={"l": 130, "r": 18, "t": 56, "b": 64},
    )
    fig_scatter_ieg_ida.update_xaxes(automargin=True, title_standoff=12)
    fig_scatter_ieg_ida.update_yaxes(automargin=True, title_standoff=12)
    graph_refs["q3_disp_ieg_ida"] = plotly_chart_numbered(
        fig_scatter_ieg_ida,
        "A inclinação positiva das linhas de tendência reforça a associação entre engajamento e resultado acadêmico.",
        apply_full_names=False,
    )

    fig_scatter_ieg_ipv = px.scatter(
        scatter_df,
        x="ieg",
        y="ipv",
        color="ano_referencia",
        opacity=0.55,
        title="Relacao entre IEG (engajamento) e IPV (ponto de virada)",
        labels={
            "ieg": "IEG",
            "ipv": "IPV",
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
    fig_scatter_ieg_ipv.update_layout(
        margin={"l": 130, "r": 18, "t": 56, "b": 64},
    )
    fig_scatter_ieg_ipv.update_xaxes(automargin=True, title_standoff=12)
    fig_scatter_ieg_ipv.update_yaxes(automargin=True, title_standoff=12)
    graph_refs["q3_disp_ieg_ipv"] = plotly_chart_numbered(
        fig_scatter_ieg_ipv,
        "O padrão também aparece para IPV, indicando que engajamento antecede pontos de virada.",
        apply_full_names=False,
    )
    return graph_refs


def render_q4(df: pd.DataFrame) -> int:
    show_subheader("Q4 - IAA: coerência com IDA e IEG")
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
    graph_ref = plotly_chart_numbered(
        fig_coherence,
        "Distribuições deslocadas para a direita indicam superestimação; para a esquerda, subestimação.",
        analysis=(
            "A distribuição do desvio (IAA menos média de IDA e IEG) está centrada levemente à direita de zero em todos os "
            "anos, a maioria dos alunos se autoavalia acima do que os indicadores objetivos registram. Em 2024, a "
            "distribuição se desloca visivelmente para a direita em relação a 2022.\n\n"
            "Perfil | Percentual | Interpretação\n"
            "Superestimam (IAA > IDA+IEG) | 61,8% | Vínculo com o programa - alavancagem afetiva\n"
            "Alinhamento coerente | ~19% | Referência de calibração\n"
            "Subestimam (IAA < IDA+IEG) | 19,2% | PRIORIDADE - risco de abandono\n\n"
            "IAA = Soma das respostas / 6 perguntas | peso no INDE: 10%.\n\n"
            "O IAA mede percepção de bem-estar e pertencimento sobre 6 dimensões: consigo mesmo, estudos, família, amigos, "
            "Associação e professores, não autopercepção de competência acadêmica."
        ),
        practical_meaning=(
            "O desvio positivo não é sinal de arrogância, pode refletir um aluno que se sente bem na Associação mas ainda "
            "não converteu esse bem-estar em desempenho. IAA alto com IDA baixo é evidência de que o programa criou "
            "pertencimento antes de gerar desempenho. A sequência emocional precede a sequência acadêmica, isso é "
            "pedagogicamente saudável e estrategicamente valioso."
        ),
    )
    return graph_ref


def render_q5_q6(df_long: pd.DataFrame) -> dict[str, int] | None:
    show_subheader("Q5 e Q6 - IPS/IPP: antecedência e confirmação com IAN")
    graph_refs: dict[str, int] = {}

    prior = df_long[df_long["ano_base"].isin([2022, 2023])].copy()
    prior = prior.dropna(subset=["ida", "ida_prox", "ieg", "ieg_prox", "ips", "ipp", "ian", "ian_prox"])
    if prior.empty:
        st.info("Sem amostra suficiente para Q5/Q6.")
        return None

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
        show_caption("Diferenças médias de IPS/IPP por evento futuro foram calculadas conforme o notebook de EDA.")

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
    graph_refs["q5_quedas_por_perfil"] = plotly_chart_numbered(
        fig_pattern,
        "O perfil combinado IPS baixo + IPP baixo concentra maior risco de deterioração no ano seguinte.",
        analysis=(
            "O perfil combinado IPS + IPP determina a probabilidade de queda futura nos três indicadores:\n\n"
            "Perfil combinado | Queda IAN | Queda IDA | Queda IEG | Risco\n"
            "IPS não baixo + IPP não baixo | 13,1% | 39,0% | 30,4% | Menor\n"
            "IPS baixo + IPP não baixo | 2,7% | 34,0% | 31,3% | Atenção\n"
            "IPS não baixo + IPP baixo | 22,2% | 30,4% | 43,7% | Elevado\n"
            "IPS baixo + IPP baixo | 17,1% | 40,0% | 45,7% | MAIOR\n\n"
            "O Gráfico 14 revela: 67,8% dos alunos com IPP frágil (<=7) apresentam IAN com defasagem alta (<=5), sinal "
            "psicopedagógico e acadêmico alinhados. Mas 32,2% desse mesmo grupo tem IAN sem defasagem alta, o sinal "
            "pedagógico existe, mas a defasagem ainda não se instalou."
        ),
        practical_meaning=(
            "IPS é avaliado por psicólogos (comportamental, emocional, social) e IPP por educadores e psicopedagogos "
            "(cognitivo, comportamental, socialização). O detalhe mais revelador: o perfil 'IPS baixo + IPP não baixo' "
            "tem queda de IAN de apenas 2,7%, sugerindo que quando o suporte psicopedagógico está adequado, o risco "
            "psicossocial isolado não se converte em defasagem de nível. Evidência de que IPP funciona como amortecedor "
            "do risco psicossocial."
        ),
        analysis_title="Análise dos gráficos",
    )

    prior["perfil_ipp_clinico"] = np.where(prior["ipp"] <= 7, "IPP fragil (<= 7)", "IPP adequado (> 7)")
    prior["perfil_ian_clinico"] = np.where(prior["ian"] <= 5, "IAN com defasagem alta (<= 5)", "IAN sem defasagem alta (> 5)")

    ipp_ian_matrix = pd.crosstab(prior["perfil_ipp_clinico"], prior["perfil_ian_clinico"], normalize="index") * 100

    contradiction_rate = (
        ((prior["ipp"] > 7) & (prior["ian"] <= 5)) | ((prior["ipp"] <= 7) & (prior["ian"] > 5))
    ).mean() * 100
    show_caption(f"Taxa de contradição entre IPP e IAN: {contradiction_rate:.1f}%")

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
    graph_refs["q6_matriz_ipp_ian"] = plotly_chart_numbered(
        fig_ipp_ian,
        "A leitura conjunta de IPP e IAN indica alinhamentos e contradições entre sinal psicopedagógico e risco de defasagem.",
    )
    return graph_refs


def render_q7(df: pd.DataFrame) -> dict | None:
    show_subheader("Q7 - Drivers do IPV")

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
    graph_ref = plotly_chart_numbered(
        fig_ipv_imp,
        "Quanto maior a barra, maior a influência do indicador na explicação do ponto de virada.",
        analysis=(
            "O modelo Random Forest identifica a importância relativa de cada indicador para explicar o IPV. Três variáveis "
            "respondem por 89,5% da explicação total:\n\n"
            "Indicador | Importância | % do total | Interpretação\n"
            "IPP - Psicopedagógico | 0,531 | 53,1% | Driver dominante - mesmo avaliador\n"
            "IEG - Engajamento | 0,230 | 23,0% | 2o driver - comportamento rastreável\n"
            "IDA - Desempenho | 0,134 | 13,4% | 3o driver - nota não é pré-requisito\n"
            "IPS - Psicossocial | 0,062 | 6,2% | Complementar\n"
            "IAA - Autoavaliação | 0,031 | 3,1% | Baixa influência direta\n"
            "IAN - Adequação nível | 0,012 | 1,2% | Quase irrelevante para IPV\n\n"
            "ALERTA METODOLÓGICO: IPP e IPV são avaliados pela mesma equipe de educadores e psicopedagogos. A importância "
            "de 53,1% do IPP pode refletir parcialmente consistência interna do avaliador, não apenas causalidade real. É "
            "mais preciso dizer que 'quem avalia bem IPP tende a avaliar bem IPV' do que 'IPP causa IPV'."
        ),
        practical_meaning=(
            "IAN aparece com apenas 1,2% de importância, o resultado mais contraintuitivo. Defasagem de nível quase não "
            "explica o ponto de virada. Um aluno defasado pode atingir o ponto de virada; um aluno em fase pode não atingi-lo. "
            "O ponto de virada depende de transformação, não de posição. Isso contradiz a intuição de que 'recuperar "
            "defasagem = ponto de virada'."
        ),
    )
    top = ipv_importance.iloc[0]
    return {
        "grafico_ref": graph_ref,
        "top_driver": str(top["indicador"]),
        "top_importance": float(top["importancia"]),
        "top3_share": float(ipv_importance.head(3)["importancia"].sum() * 100),
        "n": int(len(ipv_data)),
    }


def render_q8(df: pd.DataFrame) -> tuple[pd.DataFrame, int | None]:
    show_subheader("Q8 - Multidimensionalidade do INDE")

    multi_features = ["ida", "ieg", "ips", "ipp"]
    multi_df = df.dropna(subset=multi_features + ["inde_ano"]).copy()
    if len(multi_df) < 150:
        st.info("Amostra insuficiente para análise robusta da multidimensionalidade.")
        return pd.DataFrame(), None

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
    graph_ref = plotly_chart_numbered(
        fig_import_inde,
        "IDA e IEG tendem a formar o núcleo explicativo do INDE, com IPS/IPP agregando refinamento.",
        analysis=(
            "O modelo Random Forest (R2 treino = 0,877; R2 teste = 0,804) explica com alta fidelidade a variação do INDE. "
            "A distribuição de importância é altamente concentrada:\n\n"
            "Indicador | Importância RF | Peso formal INDE (0-7) | Divergência\n"
            "IDA | 58,6% | 20% | RF 2,9x maior - efeito sistêmico\n"
            "IEG | 28,7% | 20% | RF 1,4x maior - alinhado\n"
            "IPS | 6,6% | 10% | RF menor - indicador de triagem\n"
            "IPP | 6,1% | 10% | RF menor - indicador de triagem\n\n"
            "IDA explica 58,6% da variação observada no INDE, mas vale apenas 20% no cálculo formal. Isso não é "
            "incoerência, é sinal de que IDA funciona como proxy sistêmico: quando ele sobe, outros indicadores sobem "
            "junto. A multidimensionalidade do INDE é sistêmica, não apenas estrutural."
        ),
        practical_meaning=(
            "IDA + IEG respondem por 87,3% da explicação do INDE no modelo. A melhor combinação observada, IDA alto + "
            "IEG alto + IPS alto + IPP alto, resulta em INDE médio de 8,74 (n=43, apenas 2,2% da amostra de 1985). "
            "O teto de excelência é estreito: a questão estratégica é como expandir a faixa Ametista, não apenas manter o Topázio."
        ),
    )

    profile_df = multi_df.copy()
    for col in multi_features:
        profile_df[f"{col}_faixa"] = pd.qcut(profile_df[col], q=3, labels=["Baixo", "Intermediario", "Alto"], duplicates="drop")

    profile_summary = (
        profile_df.groupby(["ida_faixa", "ieg_faixa", "ips_faixa", "ipp_faixa"], as_index=False)
        .agg(media_inde=("inde_ano", "mean"), alunos=("ra", "nunique"))
        .sort_values(["media_inde", "alunos"], ascending=[False, False])
    )
    profile_summary = profile_summary[profile_summary["alunos"] >= 15]
    show_caption("Top combinações de perfil foram calculadas conforme o notebook de EDA.")
    return profile_summary, graph_ref


def render_q9(df_long: pd.DataFrame) -> dict | None:
    show_subheader("Q9 - Baseline de previsão de risco")

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
    show_caption("Importâncias de variáveis e calibração foram calculadas conforme o notebook de EDA.")

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


def render_q10(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    show_subheader("Q10 - Efetividade por fases e grupos de entrada")
    graph_refs: dict[str, int] = {}

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
    graph_refs["q10_fase_inde"] = plotly_chart_numbered(
        fig_phase_inde,
        "A comparação por fase mostra diferenças de nível de INDE ao longo do tempo.",
        analysis=(
            "Três gráficos compõem esta análise, cada um revela uma camada distinta da efetividade:\n\n"
            "Pedra (faixa PEDE) | INDE 2022 | INDE 2024 | Delta INDE | Risco IAN 2022 | Risco IAN 2024 | Delta Risco\n"
            "Quartzo (2,4-5,5) | 5,24 | 5,40 | +0,16 | 88,5% | 70,4% | -18,1 pp\n"
            "Ágata (5,5-6,9) | 6,61 | 6,60 | -0,01 | 85,6% | 67,1% | -18,5 pp\n"
            "Ametista (6,9-8,2) | 7,53 | 7,54 | +0,01 | 65,5% | 54,5% | -11,0 pp\n"
            "Topázio (8,2-9,3) | 8,37 | 8,47 | +0,10 | 32,3% | 23,4% | -8,9 pp\n\n"
            "Trajetória por grupo de entrada (Gráfico 19):\n"
            "Grupo de entrada | INDE 2022 | INDE 2024 | Ganho total | Interpretação\n"
            "Quartzo | 5,24 | 5,88 | +0,64 | Maior ganho - programa mais efetivo aqui\n"
            "Ágata | 6,61 | 6,64 | +0,03 | ESTAGNAÇÃO - gap estratégico prioritário\n"
            "Ametista | 7,53 | 7,50 | -0,03 | Quase estável - manutenção\n"
            "Topázio | 8,37 | 8,25 | -0,12 | Queda - programa tem dificuldade no topo\n\n"
            "PARADOXO CENTRAL: o Topázio tem o maior INDE absoluto (8,47 em 2024), mas o grupo que entrou como Topázio "
            "perdeu -0,12 pontos. O programa eleva mais quem está mais baixo e mantém, com dificuldade, quem já está no "
            "topo. Isso é coerência com a missão, mas precisa ser explicitado para gestão de expectativas."
        ),
        practical_meaning=(
            "O Gráfico 17 pode dar impressão de estabilidade (barras quase iguais entre 2022 e 2024). O Gráfico 18 mostra "
            "que por baixo dessa estabilidade há redução real de defasagem em TODAS as faixas. O programa avança mais na "
            "adequação de nível do que na elevação do INDE médio, e esse resultado positivo não aparece na leitura isolada "
            "das médias. Adicionalmente: 23,4% dos alunos Topázio têm IAN <= 5. Como IAN vale 10% do INDE, é possível ter "
            "INDE 8,3+ com dois anos de defasagem de nível, invisível nos relatórios de média."
        ),
        analysis_title="Análise dos gráficos",
    )

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
    graph_refs["q10_fase_risco"] = plotly_chart_numbered(
        fig_phase_risk,
        "Comparar nível de INDE com risco de defasagem ajuda a identificar fases com melhor equilíbrio.",
    )

    group_entry_base = program.dropna(subset=["inde_ano"]).copy()
    first_record = (
        group_entry_base.sort_values(["ra", "ano_referencia"])
        .groupby("ra", as_index=False)
        .first()[["ra", "pedra_ano"]]
        .rename(columns={"pedra_ano": "pedra_inicial"})
    )
    group_entry_base = group_entry_base.merge(first_record, on="ra", how="left")

    group_entry_evolution = (
        group_entry_base.groupby(["pedra_inicial", "ano_referencia"], as_index=False)
        .agg(media_inde=("inde_ano", "mean"), alunos=("ra", "nunique"))
    )

    stone_order = ["Quartzo", "Agata", "Ametista", "Topazio"]
    group_entry_evolution["pedra_inicial"] = pd.Categorical(group_entry_evolution["pedra_inicial"], categories=stone_order, ordered=True)
    group_entry_evolution = (
        group_entry_evolution
        .sort_values(["pedra_inicial", "ano_referencia"])
    )

    group_entry_plot = group_entry_evolution.copy()
    group_entry_plot["ano_referencia"] = group_entry_plot["ano_referencia"].astype(int).astype(str)
    group_label_map = {
        "Quartzo": "Quartzo",
        "Agata": "Agata",
        "Ametista": "Ametista",
        "Topazio": "Topazio",
    }
    group_entry_plot["grupo_inicial_rotulo"] = group_entry_plot["pedra_inicial"].map(group_label_map)
    group_order_label = ["Quartzo", "Agata", "Ametista", "Topazio"]
    group_color_map = {
        "Quartzo": "#A4B3D3",
        "Agata": "#8092AC",
        "Ametista": "#08286F",
        "Topazio": "#00164D",
    }
    group_entry_plot["grupo_inicial_rotulo"] = pd.Categorical(
        group_entry_plot["grupo_inicial_rotulo"],
        categories=group_order_label,
        ordered=True,
    )
    group_entry_plot = group_entry_plot.sort_values(["ano_referencia", "grupo_inicial_rotulo"])
    fig_group_entry = px.bar(
        group_entry_plot,
        x="ano_referencia",
        y="media_inde",
        color="grupo_inicial_rotulo",
        barmode="group",
        text="media_inde",
        title="Evolucao do INDE por grupo de entrada da pedra inicial",
        labels={
            "ano_referencia": "Ano de referencia",
            "media_inde": "INDE medio",
            "grupo_inicial_rotulo": "Grupo de entrada",
        },
        category_orders={"grupo_inicial_rotulo": group_order_label, "ano_referencia": ["2022", "2023", "2024"]},
        color_discrete_map=group_color_map,
    )
    fig_group_entry.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_group_entry.update_xaxes(type="category")
    fig_group_entry.update_yaxes(range=[0, group_entry_plot["media_inde"].max() + 1.0])
    graph_refs["q10_grupo_entrada_inde"] = plotly_chart_numbered(
        fig_group_entry,
        "A leitura por grupo de entrada evidencia heterogeneidade de trajetória e orienta ações específicas.",
    )
    return group_entry_evolution, graph_refs


def render_analise_exploratoria_tab(df: pd.DataFrame, df_long: pd.DataFrame) -> None:
    reset_graph_counter()
    inject_eda_report_theme()
    years = sorted([int(y) for y in df["ano_referencia"].dropna().unique().tolist()])
    # c1, c2, c3 = st.columns(3)
    # c1.metric("Registros analisados", f"{len(df):,}".replace(",", "."))
    # c2.metric("Alunos", f"{df['ra'].nunique():,}".replace(",", "."))
    # c3.metric("Janela temporal", " - ".join(map(str, years)))

    st.markdown(
        f"""
        <div class="eda-context">
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

    st.markdown('<div class="eda-block-title">Bloco 1 - Diagnóstico de desempenho</div>', unsafe_allow_html=True)
    with st.expander("Relações estruturais entre indicadores", expanded=True):
        render_analysis_header(
            question="Quais indicadores caminham junto com o INDE?",
            importance="Define o eixo de priorização das análises seguintes.",
            approach="Correlação de Spearman entre indicadores acadêmicos, comportamentais e de risco.",
        )
        render_corr(df)
        render_exec_note(
            message=(
                "Alavancas primárias\n"
                "IDA (0,78), IEG (0,69) e IPV (0,70) - alta correlação com o INDE, efeito transversal confirmado.\n\n"
                "Indicador precoce\n"
                "IPV sinaliza deterioração antes que o INDE caia, pela sua correlação cruzada com IDA e IEG.\n\n"
                "Ponto de atenção\n"
                "IAN e Defasagem têm correlação mútua de 0,93, mas impacto fraco sobre o INDE, parecem urgentes, mas "
                "não são alavancas do índice global.\n\n"
                "Dimensões isoladas\n"
                "IPS (0,26) e IAA (0,36) - relevantes por outras razões, mas sem retorno sistêmico esperado no INDE."
            ),
            implication=(
                "A ordem de priorização sugere: (1) fortalecer IDA como variável de controle principal; (2) usar o IPV "
                "como triagem antecipada; (3) tratar IEG como alavanca paralela com efeito direto no índice; (4) não "
                "confundir a alta correlação IAN-Defasagem com impacto sobre o INDE."
            ),
            base_visual_text="Gráfico 1",
        )

    with st.expander("Q1 - IAN: perfil de defasagem e evolução temporal", expanded=True):
        render_analysis_header(
            question="A defasagem de aprendizagem está reduzindo ao longo do tempo?",
            importance="IAN é o termômetro principal de risco pedagógico.",
            approach="Série temporal de média, composição por faixa de risco e recortes por sexo/faixa etária.",
        )
        render_q1(df)
        render_exec_note(
            message=(
                "Positivo\n"
                "Trajetória ascendente sem reversão - sinal de consistência programática, não de flutuação pontual.\n\n"
                "Atenção\n"
                "A desaceleração em 2024 (+0,44 vs +0,81 no ano anterior) pode indicar que os ganhos mais acessíveis já "
                "foram capturados, os próximos exigirão intervenções mais intensivas.\n\n"
                "Limite da análise\n"
                "A média crescente não informa sobre a distribuição: alunos na faixa de risco elevado podem estar sendo "
                "puxados pela melhora dos demais sem avançar eles mesmos."
            ),
            implication=(
                "A ONG deve preservar as ações que sustentaram essa trajetória. Para o próximo ciclo, a prioridade "
                "analítica é desagregar o IAN por faixa de risco para verificar se alunos com defasagem severa estão de "
                "fato avançando. Isso define se a estratégia é manutenção ou intensificação."
            ),
            base_visual_text="Gráfico 2",
        )

    with st.expander("Q2 - IDA: melhora, estagnação ou queda", expanded=True):
        render_analysis_header(
            question="O desempenho acadêmico (IDA) está subindo de forma consistente?",
            importance="IDA resume aprendizagem e ajuda a medir retorno das intervenções.",
            approach="Comparação anual de média e distribuição para capturar tendência e volatilidade.",
        )
        render_q2(df)
        render_exec_note(
            message=(
                "Ponto positivo\n"
                "O valor de 2024 (6,35) ainda supera 2022 (6,10), o programa não zerou o ganho, apenas não o manteve.\n\n"
                "Sinal de alerta\n"
                "A inflexão em 2023 sugere intervenção pontual sem mecanismo de manutenção. O programa teve um pico de "
                "efetividade que não se repetiu.\n\n"
                "Risco estratégico\n"
                "Se o IDA continuar recuando em 2025, o efeito sobre o INDE será direto e proporcional, dado o peso dessa "
                "correlação, uma queda sustentada compromete o indicador global."
            ),
            implication=(
                "Prioridade imediata: diagnosticar o que sustentou o pico de 2023 e o que mudou em 2024. Desagregar o IDA "
                "por disciplina (Matemática, Português, Inglês) e por fase para identificar onde ocorreu a queda. O "
                "contraste com o IAN também é revelador: como a adequação de nível melhora enquanto o desempenho "
                "acadêmico piora?"
            ),
            base_visual_text="Gráfico 7",
        )

    with st.expander("Q3 - IEG x IDA e IPV", expanded=True):
        render_analysis_header(
            question="Engajamento explica desempenho e ponto de virada?",
            importance="Se a relação for forte, IEG pode atuar como sinal de intervenção precoce.",
            approach="Correlações anuais + dispersões com linha de tendência por ano.",
        )
        q3_refs = render_q3(df)
        if q3_refs:
            render_exec_note(
                message=(
                    "Relação mais forte\n"
                    "IDA x IPV (0,55-0,62) - desempenho e ponto de virada são os mais coesos. O IPV incorpora desempenho "
                    "na sua avaliação longitudinal.\n\n"
                    "Relação estável\n"
                    "IEG x IPV (0,54-0,55) - engajamento observável e consistentemente relacionado ao julgamento qualitativo "
                    "de virada. Essa correlação não caiu em 2024, ao contrário de IDA x IPV.\n\n"
                    "Atenção ao par IDA x IPV\n"
                    "Em 2022 era o mais forte (0,62), mas caiu para 0,55 e estabilizou. Pode indicar que o IPV passou a "
                    "depender menos de notas e mais de outros fatores."
                ),
                implication=(
                    "Estabelecer um limiar de alerta no IEG, por exemplo, queda de 1 ponto em relação à média da turma "
                    "nas últimas 3 semanas, que acione revisão de acompanhamento antes da avaliação formal. Um aluno com "
                    "queda no IEG sinaliza risco simultâneo em desempenho e ponto de virada antes que qualquer nota formal caia."
                ),
                base_visual_text="Gráfico 9",
            )

    with st.expander("Q4 - IAA: coerência com indicadores objetivos", expanded=True):
        render_analysis_header(
            question="A autoavaliação (IAA) está alinhada ao desempenho observado?",
            importance="Desalinhamento pode gerar percepção incorreta de necessidade de apoio.",
            approach="Distribuição do desvio entre IAA e média de IDA/IEG com medida de super/subestimação.",
        )
        render_q4(df)
        render_exec_note(
            message=(
                "Viés de superestimação dominante\n"
                "61,8% dos alunos se avaliam acima de IDA + IEG, mas o IAA mede bem-estar e pertencimento, não nota. "
                "Não é incoerência do aluno, é limitação da comparação direta entre os instrumentos.\n\n"
                "Sinal mais crítico\n"
                "Os 19,2% que subestimam são pedagogicamente mais preocupantes: alunos que performam bem mas se sentem "
                "mal consigo mesmos. Esse grupo pode abandonar antes de atingir o ponto de virada por baixa autoestima acadêmica.\n\n"
                "Peso limitado no INDE\n"
                "IAA representa apenas 10% do INDE, o menor peso. Seu valor diagnóstico é maior do que seu impacto "
                "quantitativo no índice."
            ),
            implication=(
                "Cruzar o desvio negativo do IAA com o IPV para identificar alunos com bom desempenho mas baixa "
                "autopercepção, e acionar acompanhamento socioemocional antes que o desengajamento se instale. A "
                "prioridade são os subestimadores, não os superestimadores."
            ),
            base_visual_text="Gráfico 12",
        )

    st.markdown('<div class="eda-block-title">Bloco 2 - Sinais antecedentes de risco</div>', unsafe_allow_html=True)
    with st.expander("Q5 e Q6 - IPS/IPP: alerta antecipado e confirmação com IAN", expanded=True):
        render_analysis_header(
            question="IPS e IPP antecipam quedas futuras e confirmam risco pedagógico?",
            importance="Identificar risco cedo reduz custo e aumenta efetividade da intervenção.",
            approach="Análise longitudinal com eventos de queda no ano seguinte e matriz de coerência IPP x IAN.",
        )
        q5_q6_refs = render_q5_q6(df_long)
        if q5_q6_refs:
            render_exec_note(
                message=(
                    "Perfil crítico duplo\n"
                    "IPS baixo + IPP baixo concentra as maiores probabilidades de queda em IDA (40,0%) e IEG (45,7%). "
                    "Com pesos combinados de 30% no INDE (IPS 10% + IPP 10% + IEG 20%), esse perfil sinaliza risco de "
                    "impacto sistêmico no índice global.\n\n"
                    "Janela de intervenção\n"
                    "Os 32,2% com IPP frágil mas IAN ainda sem defasagem alta têm a maior relação custo-benefício de "
                    "intervenção: o dano ainda não é irreversível. O momento de agir é agora.\n\n"
                    "IPP como protetor de IAN\n"
                    "O perfil 'IPS baixo + IPP não baixo' tem queda de IAN de apenas 2,7%, a menor de todos. Suporte "
                    "psicopedagógico adequado funciona como amortecedor do risco psicossocial."
                ),
                implication=(
                    "Triagem em duas etapas: (1) identificar IPS baixo + IPP baixo para protocolo de intervenção integrada; "
                    "(2) priorizar dentro desse grupo os 32,2% com IPP frágil mas IAN ainda sem defasagem, maior relação "
                    "custo-benefício. Decisões baseadas em IPS ou IPP isoladamente perdem essa camada de precisão."
                ),
                base_visual_text="Gráficos 13 e 14",
            )

    st.markdown('<div class="eda-block-title">Bloco 3 - Drivers e modelagem exploratória</div>', unsafe_allow_html=True)
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
                    "Driver dominante\n"
                    "IPP (53,1%) - avaliação psicopedagógica é o fator mais preditivo. Pelo peso do IPP no INDE (10%), "
                    "seu impacto sobre o IPV (20%) é desproporcional ao seu peso direto, opera como amplificador indireto.\n\n"
                    "Segundo driver\n"
                    "IEG (23,0%) - engajamento em tarefas registradas é o componente comportamental mais relevante. "
                    "Combinado com IPP, os dois respondem por 76,1% da explicação do IPV.\n\n"
                    "Terceiro driver\n"
                    "IDA (13,4%) - desempenho acadêmico contribui, mas é o terceiro fator. Um aluno engajado e bem avaliado "
                    "pedagogicamente pode estar em virada mesmo com IDA moderado.\n\n"
                    "Irrelevante para IPV\n"
                    "IAN (1,2%) - defasagem de nível praticamente não explica o ponto de virada."
                ),
                implication=(
                    "Para aumentar o IPV, priorizar nessa ordem: (1) qualidade do acompanhamento psicopedagógico, (2) "
                    "engajamento consistente em tarefas registradas, (3) desempenho acadêmico como complemento. Em cenário "
                    "de orçamento limitado, recursos em IPP e IEG geram retorno proporcionalmente maior no IPV, e nos 20% "
                    "do INDE que esse indicador representa."
                ),
                base_visual_text="Gráfico 15",
            )

    with st.expander("Q8 - Combinações que explicam melhor o INDE", expanded=True):
        render_analysis_header(
            question="Quais combinações de desempenho e comportamento resultam em melhor INDE?",
            importance="Permite desenhar perfis-alvo para políticas de acompanhamento.",
            approach="Modelo de regressão com variáveis-chave e análise de perfis por faixas (baixo/intermediário/alto).",
        )
        q8_profiles, q8_ref = render_q8(df)
        if isinstance(q8_profiles, pd.DataFrame) and not q8_profiles.empty and q8_ref is not None:
            render_exec_note(
                message=(
                    "Núcleo explicativo\n"
                    "IDA (58,6%) + IEG (28,7%) = 87,3% da variação do INDE com apenas dois indicadores. Políticas que movem "
                    "esses dois atuam sobre quase 9 em cada 10 pontos de variação no índice global.\n\n"
                    "Refinamento de segundo nível\n"
                    "IPS e IPP têm importância preditiva baixa para o INDE médio, mas alta para perfis de vulnerabilidade. "
                    "São indicadores de triagem de risco, não de alavancagem de média.\n\n"
                    "Tensão modelo x estrutura\n"
                    "IDA vale 20% no cálculo formal, mas explica 58,6% da variação observada. IDA funciona como proxy: "
                    "alunos com IDA alto tendem a ter IEG alto, IPV alto e melhores avaliações de IPP, o que amplifica o "
                    "efeito real de IDA além do seu peso declarado."
                ),
                implication=(
                    "Sequência correta: (1) estabilizar IDA + IEG para elevar o INDE médio da base, 87,3% da explicação com "
                    "dois indicadores; (2) usar IPS + IPP como filtros de triagem preventiva para blindar perfis vulneráveis; "
                    "(3) combinar tudo para elevar o teto de excelência nos casos já consolidados."
                ),
                base_visual_text="Gráfico 16",
            )

    with st.expander("Q9 - Baseline de previsão de risco de defasagem", expanded=True):
        render_analysis_header(
            question="Qual a capacidade inicial de prever risco de defasagem no próximo ano?",
            importance="Oferece base para priorização preventiva de alunos.",
            approach="Baseline supervisionado com split treino/teste e leitura de ROC-AUC, PR-AUC e calibração.",
        )
        q9_info = render_q9(df_long)
        if q9_info is not None:
            render_graph_note(
                analysis=(
                    "Métrica | Valor | Interpretação\n"
                    "ROC-AUC (teste) | 0,838 | Discriminação geral - acima do limiar 'bom' (>=0,80)\n"
                    "PR-AUC (teste) | 0,771 | Precisão nos casos de risco - métrica operacional real\n"
                    "Variável de maior peso | IPP (33,1%) | Consistente com análise de drivers do IPV\n"
                    "Base modelada | 690 observações | Piso mínimo - expandir com anos anteriores do PEDE\n\n"
                    "ROC-AUC de 0,838 significa que o modelo distingue corretamente alunos em risco em 83,8% dos pares "
                    "comparados. Em bases desbalanceadas, onde alunos em risco são minoria, o ROC-AUC pode ser otimista. "
                    "O PR-AUC de 0,771 é a métrica honesta: mede a precisão do modelo especificamente nos casos positivos, "
                    "que tendem a ser minoria na base.\n\n"
                    "ALERTA DE CALIBRAÇÃO: um modelo com bom AUC mas mal calibrado pode atribuir probabilidade de 70% a "
                    "casos que na realidade têm 30% de chance. Para uso operacional em triagem, a calibração de "
                    "probabilidade (Platt scaling ou isotonic regression) é etapa obrigatória antes da produção."
                ),
                practical_meaning=(
                    "Um ROC-AUC de 0,838 em base de 690 observações é resultado expressivo para um modelo baseline, sem "
                    "engenharia de features avançada, sem dados externos. Os indicadores do PEDE, mesmo sem transformações, "
                    "já carregam sinal preditivo suficiente para construir uma fila de prioridade operacional. Identificar "
                    "corretamente 8 em cada 10 alunos em risco antes da queda representa uma mudança de paradigma de "
                    "reativo para preventivo."
                ),
                analysis_title="Desempenho do modelo baseline",
            )
            render_exec_note(
                message=(
                    "Viabilidade confirmada\n"
                    "ROC-AUC = 0,838 com apenas os indicadores existentes. Não é necessário coletar novos dados para "
                    "construir um sistema de triagem funcional.\n\n"
                    "IPP como âncora preditiva\n"
                    "Com 33,1% de importância no modelo de risco e 53,1% no modelo do IPV, o IPP emerge consistentemente "
                    "como o indicador mais preditivo do sistema. A qualidade e frequência da avaliação psicopedagógica "
                    "impacta diretamente a capacidade do modelo de identificar risco.\n\n"
                    "Limitação crítica da base\n"
                    "690 observações é o piso mínimo. Com a série histórica completa (2020-2023), features de trajetória "
                    "(Delta IDA, Delta IEG ano a ano) provavelmente elevariam o ROC-AUC para acima de 0,88.\n\n"
                    "PR-AUC é a métrica que importa\n"
                    "Em bases desbalanceadas, o ROC-AUC pode ser otimista. O PR-AUC de 0,771 é a métrica honesta para "
                    "triagem com recursos limitados: mede o quanto o modelo é preciso quando aciona o alerta."
                ),
                implication=(
                    "Três melhorias incrementais sem custo de coleta: (1) incluir features de trajetória longitudinal "
                    "(Delta IDA, Delta IEG entre anos); (2) calibrar as probabilidades para uso operacional real; (3) "
                    "expandir a base com anos anteriores do PEDE para aumentar a amostra de treino de 690 para "
                    "potencialmente 2.000+ observações."
                ),
                base_visual_text="Sem base gráfica - métricas do modelo baseline (ROC-AUC e PR-AUC)",
            )

    with st.expander("Q10 - Efetividade por fase e grupos de entrada", expanded=True):
        render_analysis_header(
            question="Quais fases e grupos de entrada mostram maior efetividade ao longo do tempo?",
            importance="Suporta alocação de recursos e réplica de práticas bem-sucedidas.",
            approach="Comparação anual por fase e por grupo de entrada, com foco em INDE e risco de defasagem.",
        )
        group_entry_evolution, q10_refs = render_q10(df)
        if isinstance(group_entry_evolution, pd.DataFrame) and not group_entry_evolution.empty and q10_refs:
            render_exec_note(
                message=(
                    "Impacto real e verificável\n"
                    "Todas as faixas reduziram o percentual de alunos com defasagem alta, evidência de efetividade distribuída, "
                    "não concentrada numa faixa.\n\n"
                    "Programa de recuperação, não de excelência\n"
                    "O maior ganho de INDE é do grupo Quartzo (+0,64). O Topázio perdeu (-0,12). O programa é mais efetivo "
                    "em elevar os mais vulneráveis, coerência com a missão.\n\n"
                    "Alerta: INDE alto com IAN baixo\n"
                    "23,4% dos alunos Topázio têm defasagem alta. INDE elevado não garante adequação pedagógica, a gestão "
                    "precisa conhecer essa limitação estrutural do instrumento.\n\n"
                    "Gap estratégico: grupo Ágata\n"
                    "O grupo Ágata cresceu apenas +0,03 em dois anos. Com INDE médio de 6,64 e limiar Ametista em 6,9, "
                    "pequenos ganhos em IDA e IEG podem desbloquear migração de faixa para dezenas de alunos."
                ),
                implication=(
                    "Três frentes: (1) manter intervenções que reduzem defasagem em Quartzo e Ágata; (2) criar estratégia "
                    "específica para o grupo Ágata próximo do limiar Ametista; (3) monitorar 'Topázio com IAN <= 5' para "
                    "blindar sustentabilidade de longo prazo."
                ),
                base_visual_text="Gráficos 17, 18 e 19",
            )

    st.markdown('<div class="eda-block-title">Bloco 4 - Insights</div>', unsafe_allow_html=True)
    with st.expander("Q11 - Insights", expanded=True):
        render_analysis_header(
            question=(
                "Você pode adicionar mais insights e pontos de vista não abordados nas perguntas, utilizando a "
                "criatividade e a análise dos dados para trazer sugestões para a Passos Mágicos?"
            ),
            importance="Enriquecer a análise com perspectivas transversais não cobertas pelas questões anteriores.",
            approach="Síntese cruzada de todos os blocos analisados e raciocínio estratégico sobre o programa.",
        )
        render_graph_note(
            analysis=(
                "1. Alerta semanal via IEG\n"
                "O IEG é mensurável semanalmente, notas e IDA só aparecem em avaliações formais. Proposta: calcular a "
                "variação do engajamento individual em relação à média da turma nas últimas 3 semanas. Queda de 1+ ponto "
                "dispara notificação para o educador responsável, antes que qualquer nota caia. Custo de implementação: "
                "zero coleta adicional, apenas processamento dos dados já existentes. Sinal semanal: IEG -> Sinal "
                "bimestral: IDA -> Sinal anual: INDE.\n\n"
                "2. Alunos entre faixas Pedra\n"
                "O maior retorno marginal está nos alunos próximos dos limiares de transição. Um aluno com INDE 6,70 "
                "precisa de apenas +0,20 para cruzar para Ametista (6,9). Com IDA pesando 20% no INDE, uma melhora de "
                "1 ponto nas três notas eleva o INDE em 0,20, suficiente para migrar de faixa. Proposta: criar lista de "
                "'alunos de limiar' com INDE entre -0,3 e 0 do próximo threshold, e priorizar para reforço acadêmico "
                "pontual. Gestão de mobilidade, não de média.\n\n"
                "3. Avaliador: IPP e IPV são a mesma voz\n"
                "IPP (10% do INDE) e IPV (20% do INDE) são avaliados pela mesma equipe, juntos valem 30% do INDE. O "
                "Random Forest mostrou que IPP explica 53,1% do IPV, em parte porque é o mesmo avaliador respondendo "
                "sobre o mesmo aluno. Isso cria inflação artificial de 30% do índice com fonte única. Proposta: "
                "introduzir avaliadores distintos para IPP e IPV, ou aplicar os instrumentos com intervalo mínimo de 2 "
                "semanas. Criar checklist de evidências observáveis para ancoragem do IPV reduz dependência da impressão subjetiva.\n\n"
                "4. Topázio com defasagem: os invisíveis do índice global\n"
                "23,4% dos alunos Topázio têm IAN <= 5. São excelentes no índice, mas com defasagem estrutural que pode "
                "comprometer a trajetória fora da Associação (vestibular, ENEM, mercado de trabalho). Proposta: criar o "
                "perfil 'Topázio com defasagem' como categoria de monitoramento específica. A intervenção não é de "
                "emergência (o INDE está alto), mas de sustentabilidade: a defasagem de nível precisa ser trabalhada "
                "antes que o aluno transite para ambientes onde o suporte da Associação não estará disponível.\n\n"
                "5. IAA como termômetro de pertencimento\n"
                "IAA alto com IDA baixo não é superestimação, é evidência de que o programa criou pertencimento antes de "
                "gerar desempenho. A sequência emocional precede a sequência acadêmica, isso é pedagogicamente saudável e "
                "estrategicamente valioso. Esses alunos já confiam na Associação, já se sentem seguros: falta converter "
                "esse pertencimento em desempenho. São os casos de maior alavancagem afetiva disponível para intervenção acadêmica.\n\n"
                "6. O modelo preditivo como produto operacional\n"
                "O baseline (ROC-AUC = 0,838) já funciona bem o suficiente para triagem. O gap é de implementação, não de "
                "qualidade técnica. Proposta: painel de triagem preventiva simples, os 20% de alunos de maior probabilidade "
                "de risco, com indicadores-chave e driver principal de risco para cada um (IPP frágil? IEG em queda? IDA "
                "abaixo da média da fase?). Atualizado a cada ciclo avaliativo. Custo: zero coleta adicional."
            ),
            practical_meaning=(
                "A Passos Mágicos opera um programa de recuperação de trajetória, não de aceleração de excelência. Os "
                "dados mostram isso com clareza: o maior ganho de INDE é do grupo Quartzo (+0,64), a maior redução de "
                "defasagem é nas faixas mais baixas, e o programa tem dificuldade em manter os melhores no topo. Isso não "
                "é fraqueza, é coerência com a missão. Mas tem uma implicação estratégica direta: as métricas de sucesso "
                "devem medir mobilidade de faixa e redução de defasagem, não INDE médio.\n\n"
                "PROPOSTA FINAL: adotar a TAXA DE MIGRAÇÃO DE FAIXA PEDRA como KPI primário, percentual de alunos que "
                "sobem pelo menos uma faixa por ciclo anual. Esse indicador captura o que o programa realmente faz bem, "
                "torna visível o impacto que as médias escondem, e orienta alocação de recursos para onde o retorno é mais alto."
            ),
            analysis_title="Insights",
            practical_title="Síntese final",
        )

