from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Passos Magicos | Risco de Defasagem",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "modelo_risco_defasagem.pkl"


def normalize_column_name(col_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(col_name)).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", normalized).strip("_").lower()
    return normalized


def simplify_feature_name(feature_name: str) -> str:
    return feature_name.replace("num__", "").replace("cat__", "")


def pretty_feature_name(feature_name: str) -> str:
    labels = {
        "idade": "Idade",
        "inde_22": "INDE 2022",
        "inde_23": "INDE 2023",
        "inde_ano": "INDE atual",
        "ian": "IAN",
        "ida": "IDA",
        "ieg": "IEG",
        "iaa": "IAA",
        "ips": "IPS",
        "ipp": "IPP",
        "ipv": "IPV",
        "defasagem": "Defasagem",
        "nota_matematica": "Nota de matematica",
        "nota_portugues": "Nota de portugues",
        "nota_ingles": "Nota de ingles",
        "media_notas": "Media de notas",
        "media_comportamental": "Media comportamental",
        "desalinhamento_autoavaliacao": "Desalinhamento de autoavaliacao",
        "delta_inde_hist": "Delta historico de INDE",
        "genero": "Genero",
        "instituicao_ensino": "Instituicao de ensino",
        "fase_programa": "Fase do programa",
        "turma": "Turma",
        "pedra_ano": "Pedra do ano",
        "ativo_inativo": "Status ativo",
    }
    normalized = simplify_feature_name(feature_name)
    return labels.get(normalized, normalized.replace("_", " ").strip().title())


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan, "INCLUIR": np.nan, "incluir": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_age_series(series: pd.Series) -> pd.Series:
    numeric_age = coerce_numeric_series(series)
    date_values = pd.to_datetime(series, errors="coerce")
    age_from_date = np.where(
        date_values.notna() & (date_values.dt.year == 1900) & (date_values.dt.month == 1),
        date_values.dt.day,
        np.nan,
    )
    result = pd.Series(numeric_age, index=series.index)
    mask = result.isna() & ~pd.isna(age_from_date)
    result.loc[mask] = age_from_date[mask]
    return result.where(result.between(6, 30))


def classify_risk(probability: float, risk_bands: dict | None = None) -> str:
    risk_bands = risk_bands or {}
    low_max = float(risk_bands.get("baixo_max", 0.33))
    high_min = float(risk_bands.get("alto_min", 0.66))
    if probability < low_max:
        return "Baixo"
    if probability < high_min:
        return "Moderado"
    return "Alto"


def build_gauge(probability: float, threshold: float, risk_bands: dict | None = None) -> go.Figure:
    risk_bands = risk_bands or {}
    low_max = float(risk_bands.get("baixo_max", 0.33))
    high_min = float(risk_bands.get("alto_min", 0.66))
    value = float(probability * 100)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": "Probabilidade de risco de defasagem"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, low_max * 100], "color": "#2ca02c"},
                    {"range": [low_max * 100, high_min * 100], "color": "#ffbf00"},
                    {"range": [high_min * 100, 100], "color": "#d62728"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.8,
                    "value": float(threshold * 100),
                },
            },
        )
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=70, b=20))
    return fig


def alias_map() -> dict[str, list[str]]:
    return {
        "ra": ["ra", "registro_aluno", "matricula", "registro"],
        "idade": ["idade", "idade_22", "idade_do_aluno"],
        "inde_ano": [
            "inde_ano",
            "inde_2024",
            "inde_2023",
            "inde_2022",
            "inde_23",
            "inde_22",
            "inde",
            "inde_atual",
            "inde_do_ano_passado",
            "inde_de_2_anos_atras",
        ],
        "ian": ["ian", "ind_adequacao_nivel_ian", "indicador_de_adequacao_de_nivel_ian"],
        "ida": ["ida", "indicador_de_desempenho_acad_ida"],
        "ieg": ["ieg", "ind_engajamento_ieg"],
        "iaa": ["iaa", "ind_autoavaliacao_iaa"],
        "ips": ["ips", "ind_psicossocial_ips"],
        "ipp": ["ipp", "ind_psicopedagogico_ipp"],
        "ipv": ["ipv", "indicador_de_ponto_de_virada_ipv"],
        "defasagem": ["defasagem", "defas"],
        "nota_matematica": ["nota_matematica", "mat", "matem", "matematica", "matematica_mat"],
        "nota_portugues": ["nota_portugues", "por", "portug", "portugues", "portugues_por"],
        "nota_ingles": ["nota_ingles", "ing", "ingles", "ingles_ing"],
        "genero": ["genero"],
        "instituicao_ensino": ["instituicao_ensino", "instituicao_de_ensino", "escola", "instituicao"],
        "fase_programa": ["fase_programa", "fase", "fase_ideal", "fase_ideal_codigo", "fase_ideal_num"],
        "turma": ["turma"],
        "pedra_ano": ["pedra_ano", "pedra_2024", "pedra_2023", "pedra_22", "pedra_23", "pedra"],
        "ativo_inativo": ["ativo_inativo", "ativo_inativo_1"],
    }


def map_uploaded_columns(raw_df: pd.DataFrame, required_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_lookup = {normalize_column_name(c): c for c in raw_df.columns}
    aliases = alias_map()
    mapped = pd.DataFrame(index=raw_df.index)
    report_rows = []

    for target_col in required_columns:
        possible_names = aliases.get(target_col, [target_col])
        chosen_source = None
        for name in possible_names:
            normalized_name = normalize_column_name(name)
            if normalized_name in normalized_lookup:
                chosen_source = normalized_lookup[normalized_name]
                break
        mapped[target_col] = raw_df[chosen_source] if chosen_source else np.nan
        report_rows.append({"feature_modelo": target_col, "coluna_origem": chosen_source or "nao encontrada"})

    report_df = pd.DataFrame(report_rows)
    return mapped, report_df


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    extension = Path(uploaded_file.name).suffix.lower()
    if extension == ".csv":
        return pd.read_csv(uploaded_file)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Formato de arquivo nao suportado: {extension}")


@st.cache_resource
def load_bundle(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em: {model_path.resolve()}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def local_explanations(pipeline: object, input_df: pd.DataFrame, healthy_thresholds: dict, top_n: int = 5) -> pd.DataFrame:
    """
    Tenta explicacao local via SHAP.
    Se nao for possivel, usa heuristica com distancia para thresholds pedagogicos.
    """
    try:
        import shap

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        transformed = preprocessor.transform(input_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed)
            if isinstance(shap_values, list):
                contributions = shap_values[1][0]
            else:
                contributions = shap_values[0]
        else:
            if not hasattr(model, "coef_"):
                raise ValueError("Modelo sem suporte para explicacao local.")
            contributions = transformed[0] * model.coef_[0]

        explanation_df = pd.DataFrame(
            {
                "fator": [simplify_feature_name(f) for f in feature_names],
                "contribuicao": contributions,
            }
        )
        explanation_df["impacto_abs"] = explanation_df["contribuicao"].abs()
        explanation_df = explanation_df.sort_values("impacto_abs", ascending=False).head(top_n)
        explanation_df["direcao"] = np.where(explanation_df["contribuicao"] >= 0, "Aumenta risco", "Reduz risco")
        return explanation_df[["fator", "contribuicao", "direcao"]]
    except Exception:
        row = input_df.iloc[0]
        heuristic_rows = []
        for metric, threshold in healthy_thresholds.items():
            if metric in row.index and pd.notna(row[metric]):
                gap = float(threshold - float(row[metric]))
                if gap > 0:
                    heuristic_rows.append({"fator": metric, "contribuicao": gap, "direcao": "Aumenta risco"})
        if not heuristic_rows:
            heuristic_rows.append({"fator": "perfil_geral", "contribuicao": 0.0, "direcao": "Sem alerta dominante"})
        return pd.DataFrame(heuristic_rows).sort_values("contribuicao", ascending=False).head(top_n)


def suggest_actions(row: pd.Series, risk_level: str) -> list[str]:
    suggestions = []

    if pd.notna(row.get("ian")) and row["ian"] <= 5:
        suggestions.append("Plano de recuperacao de defasagem com metas quinzenais.")
    if pd.notna(row.get("ida")) and row["ida"] <= 6:
        suggestions.append("Reforco academico em leitura/escrita e matematica com monitoria semanal.")
    if pd.notna(row.get("ieg")) and row["ieg"] <= 7:
        suggestions.append("Acao de engajamento com tutor de referencia e check-ins semanais.")
    if pd.notna(row.get("ips")) and row["ips"] <= 7:
        suggestions.append("Encaminhar para acompanhamento psicossocial preventivo.")
    if pd.notna(row.get("ipp")) and row["ipp"] <= 7:
        suggestions.append("Intervencao psicopedagogica para estrategia de aprendizagem.")
    if pd.notna(row.get("ipv")) and row["ipv"] <= 7:
        suggestions.append("Plano de ponto de virada com metas de curto prazo.")
    if risk_level == "Alto":
        suggestions.append("Priorizar este aluno na agenda da equipe multidisciplinar nas proximas 2 semanas.")

    if not suggestions:
        suggestions.append("Manter acompanhamento regular. Perfil atual sem alerta critico.")

    return list(dict.fromkeys(suggestions))


st.title("Dashboard de Risco de Defasagem - Associacao Passos Magicos")
st.caption("Predicao individual e em lote com recomendacoes pedagogicas orientadas por dados.")

try:
    bundle = load_bundle(MODEL_PATH)
except Exception as model_error:
    st.error(str(model_error))
    st.stop()

pipeline = bundle["pipeline"]
threshold = float(bundle.get("threshold", 0.5))
feature_columns = bundle.get("feature_columns", [])
numeric_features = bundle.get("numeric_features", [])
categorical_features = bundle.get("categorical_features", [])
default_inputs = bundle.get("default_inputs", {})
category_options = bundle.get("category_options", {})
healthy_thresholds = bundle.get("healthy_thresholds", {"ian": 5, "ida": 6, "ieg": 7, "ips": 7, "ipp": 7, "ipv": 7})
risk_bands = bundle.get("risk_bands", {"baixo_max": 0.33, "alto_min": 0.66})
training_info = bundle.get("training_info", {})

st.sidebar.header("Configuracao operacional")
operational_threshold = st.sidebar.slider(
    "Threshold de classificacao",
    min_value=0.05,
    max_value=0.95,
    value=float(round(threshold, 2)),
    step=0.01,
)
st.sidebar.caption(f"Modelo treinado: {bundle.get('model_name', 'N/A')}")
if training_info:
    st.sidebar.caption(
        f"Teste temporal: {training_info.get('test_years', ['N/A'])} | "
        f"ROC-AUC: {training_info.get('best_model_test_metrics', {}).get('roc_auc', np.nan):.3f} | "
        f"PR-AUC: {training_info.get('best_model_test_metrics', {}).get('pr_auc', np.nan):.3f}"
    )

st.sidebar.header("Entrada manual de 1 aluno")
manual_data = {}

for col in numeric_features:
    default_value = float(default_inputs.get(col, 0.0)) if pd.notna(default_inputs.get(col, np.nan)) else 0.0
    if col == "idade":
        manual_data[col] = st.sidebar.slider(pretty_feature_name(col), min_value=7, max_value=30, value=int(round(default_value)), step=1)
    elif col == "defasagem":
        manual_data[col] = st.sidebar.slider(pretty_feature_name(col), min_value=-6, max_value=6, value=int(round(default_value)), step=1)
    elif col.startswith("delta_") or col == "desalinhamento_autoavaliacao":
        bounded_default = min(max(default_value, -10.0), 10.0)
        manual_data[col] = st.sidebar.slider(pretty_feature_name(col), min_value=-10.0, max_value=10.0, value=float(round(bounded_default, 2)), step=0.1)
    else:
        bounded_default = min(max(default_value, 0.0), 10.0)
        manual_data[col] = st.sidebar.slider(pretty_feature_name(col), min_value=0.0, max_value=10.0, value=float(round(bounded_default, 2)), step=0.1)

for col in categorical_features:
    options = category_options.get(col, [])
    default_value = str(default_inputs.get(col, "Nao Informado"))
    options = [str(o) for o in options if str(o).strip() != ""]
    if default_value not in options:
        options = [default_value] + options
    if not options:
        options = ["Nao Informado"]
    manual_data[col] = st.sidebar.selectbox(pretty_feature_name(col), options=options, index=0)

tab_manual, tab_batch = st.tabs(["Predicao individual", "Predicao em lote (CSV/Excel)"])

with tab_manual:
    st.subheader("Analise de risco para um aluno")
    predict_clicked = st.button("Calcular risco do aluno", type="primary")

    if predict_clicked:
        single_input = pd.DataFrame([manual_data])[feature_columns]
        probability = float(pipeline.predict_proba(single_input)[0, 1])
        risk_level = classify_risk(probability, risk_bands=risk_bands)
        predicted_label = int(probability >= operational_threshold)

        col1, col2 = st.columns([1.3, 1.0])
        with col1:
            st.plotly_chart(build_gauge(probability, operational_threshold, risk_bands=risk_bands), use_container_width=True)
        with col2:
            st.metric("Probabilidade", f"{probability * 100:.1f}%")
            st.metric("Nivel de risco", risk_level)
            st.metric("Classe (threshold)", f"{predicted_label}")
            st.caption(f"Threshold em uso: {operational_threshold:.3f} | Baixo<{risk_bands.get('baixo_max', 0.33):.2f} | Alto>={risk_bands.get('alto_min', 0.66):.2f}")

        st.markdown("### Principais fatores da previsao")
        factors_df = local_explanations(pipeline, single_input, healthy_thresholds, top_n=6)
        if "fator" in factors_df.columns:
            factors_df["fator"] = factors_df["fator"].apply(pretty_feature_name)
        st.dataframe(factors_df, use_container_width=True)

        st.markdown("### Acao recomendada")
        actions = suggest_actions(single_input.iloc[0], risk_level)
        for action in actions:
            st.write(f"- {action}")

with tab_batch:
    st.subheader("Upload de arquivo para previsao de turma")
    uploaded_file = st.file_uploader("Envie CSV ou Excel com dados dos alunos", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            raw_batch = read_uploaded_table(uploaded_file)
        except Exception as read_error:
            st.error(f"Nao foi possivel ler o arquivo enviado: {read_error}")
            st.stop()

        mapped_batch, mapping_report = map_uploaded_columns(raw_batch, feature_columns)
        recognized_cols = int((mapping_report["coluna_origem"] != "nao encontrada").sum())
        st.caption(f"Mapeamento de colunas: {recognized_cols}/{len(feature_columns)} features encontradas no arquivo.")
        with st.expander("Ver mapeamento de colunas"):
            mapping_view = mapping_report.copy()
            mapping_view["feature_modelo"] = mapping_view["feature_modelo"].apply(pretty_feature_name)
            st.dataframe(mapping_view, use_container_width=True)

        for col in feature_columns:
            if col not in mapped_batch.columns:
                mapped_batch[col] = default_inputs.get(col, np.nan)

        for col in numeric_features:
            if col == "idade":
                mapped_batch[col] = normalize_age_series(mapped_batch[col])
            else:
                mapped_batch[col] = coerce_numeric_series(mapped_batch[col])
            mapped_batch[col] = mapped_batch[col].fillna(float(default_inputs.get(col, 0.0)))

        for col in categorical_features:
            mapped_batch[col] = mapped_batch[col].astype(str).str.strip()
            mapped_batch[col] = mapped_batch[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})
            mapped_batch[col] = mapped_batch[col].fillna(str(default_inputs.get(col, "Nao Informado")))

        probs = pipeline.predict_proba(mapped_batch[feature_columns])[:, 1]
        batch_output = raw_batch.copy()
        batch_output["prob_risco"] = probs
        batch_output["classe_risco_threshold"] = (batch_output["prob_risco"] >= operational_threshold).astype(int)
        batch_output["nivel_risco"] = batch_output["prob_risco"].apply(lambda p: classify_risk(p, risk_bands=risk_bands))

        st.markdown("### Resultado da previsao")
        st.dataframe(batch_output.head(100), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Risco medio da turma", f"{batch_output['prob_risco'].mean() * 100:.1f}%")
        with col_b:
            st.metric("Alunos em risco (classe=1)", int(batch_output["classe_risco_threshold"].sum()))

        st.markdown("### Alunos prioritarios (maior probabilidade)")
        priority_cols = [c for c in ["ra", "nome", "nome_aluno", "turma", "fase_programa"] if c in batch_output.columns]
        priority_view = (
            batch_output.sort_values("prob_risco", ascending=False)
            .loc[:, priority_cols + ["prob_risco", "nivel_risco", "classe_risco_threshold"]]
            .head(20)
        )
        st.dataframe(priority_view, use_container_width=True)

        dist = batch_output["nivel_risco"].value_counts().reindex(["Baixo", "Moderado", "Alto"], fill_value=0).reset_index()
        dist.columns = ["nivel_risco", "quantidade"]

        fig_dist = px.bar(
            dist,
            x="nivel_risco",
            y="quantidade",
            color="nivel_risco",
            color_discrete_map={"Baixo": "#2ca02c", "Moderado": "#ffbf00", "Alto": "#d62728"},
            title="Distribuicao de risco na turma",
            text_auto=True,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        csv_bytes = batch_output.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar resultado em CSV",
            data=csv_bytes,
            file_name="predicoes_risco_turma.csv",
            mime="text/csv",
        )

        st.info("Dica: use o mapeamento de colunas para ajustar seu arquivo e reduzir preenchimento por valores padrao.")
