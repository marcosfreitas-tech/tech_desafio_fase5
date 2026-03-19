from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
N_FOLDS = 5
TARGET_IAN_THRESHOLD = 5.0
TARGET_IDA_THRESHOLD = 6.0
RECALL_OBJECTIVE = 0.80
THRESHOLD_GRID = np.round(np.linspace(0.10, 0.90, 33), 3)

NUMERIC_FEATURES_ALL = [
    "idade",
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
    "defasagem",
    "nota_matematica",
    "nota_portugues",
    "nota_ingles",
    "media_notas",
    "media_comportamental",
    "desalinhamento_autoavaliacao",
    "delta_inde_hist",
]
CATEGORICAL_FEATURES_ALL = ["genero", "instituicao_ensino", "fase_programa", "turma", "pedra_ano", "ativo_inativo"]

NUMERIC_MIN_OBS_RATIO = 0.05
CATEGORICAL_MIN_OBS_RATIO = 0.05
MIN_CATEGORICAL_UNIQUE = 2

FEATURE_LABELS = {
    "idade": "Idade",
    "inde_22": "INDE 2022 (Índice de Desenvolvimento Educacional)",
    "inde_23": "INDE 2023 (Índice de Desenvolvimento Educacional)",
    "inde_ano": "INDE do ano-base (Índice de Desenvolvimento Educacional)",
    "ian": "IAN (Índice de Adequação de Nível)",
    "ida": "IDA (Indicador de Desempenho Acadêmico)",
    "ieg": "IEG (Indicador de Engajamento)",
    "iaa": "IAA (Indicador de Autoavaliação)",
    "ips": "IPS (Índice Psicossocial)",
    "ipp": "IPP (Índice Psicopedagógico)",
    "ipv": "IPV (Indicador de Ponto de Virada)",
    "defasagem": "Defasagem escolar",
    "nota_matematica": "Nota de Matemática",
    "nota_portugues": "Nota de Português",
    "nota_ingles": "Nota de Inglês",
    "media_notas": "Média das notas",
    "media_comportamental": "Média comportamental",
    "desalinhamento_autoavaliacao": "Desalinhamento de autoavaliação",
    "delta_inde_hist": "Variação histórica de INDE",
    "genero": "Gênero",
    "instituicao_ensino": "Instituição de ensino",
    "fase_programa": "Fase do programa",
    "turma": "Turma",
    "pedra_ano": "Pedra do ano",
    "ativo_inativo": "Status (ativo/inativo)",
}

INPUT_GROUPS = {
    "dados_aluno_numeric": ["idade"],
    "dados_aluno_categorical": ["genero", "fase_programa"],
    "notas": ["nota_matematica", "nota_portugues", "nota_ingles"],
    "dimensao_academica_inputs": ["fase_ideal_num", "ieg_total_pontos", "ieg_qtd_tarefas"],
    "dimensao_psicossocial_inputs": ["iaa_soma_respostas", "iaa_qtd_perguntas", "ips_soma_avaliacoes", "ips_qtd_avaliadores"],
    "dimensao_psicopedagogica_inputs": ["ipp_soma_avaliacoes", "ipp_qtd_avaliacoes"],
}

PHASE_OPTIONS = [
    ("ALFA", "Alfa (alfabetização)"),
    ("FASE 1", "Fase 1"),
    ("FASE 2", "Fase 2"),
    ("FASE 3", "Fase 3"),
    ("FASE 4", "Fase 4"),
    ("FASE 5", "Fase 5"),
    ("FASE 6", "Fase 6"),
    ("FASE 7", "Fase 7"),
    ("FASE 8", "Fase 8"),
]
PHASE_VALUE_TO_LABEL = {value: label for value, label in PHASE_OPTIONS}
PHASE_LABEL_TO_VALUE = {label: value for value, label in PHASE_OPTIONS}
PHASE_LABELS = [label for _, label in PHASE_OPTIONS]

DOCUMENTED_LIMITS = {
    "idade": {"min": 6.0, "max": 30.0, "step": 1.0, "format": "%.0f"},
    "nota_matematica": {"min": 0.0, "max": 10.0, "step": 0.1, "format": "%.2f"},
    "nota_portugues": {"min": 0.0, "max": 10.0, "step": 0.1, "format": "%.2f"},
    "nota_ingles": {"min": 0.0, "max": 10.0, "step": 0.1, "format": "%.2f"},
}
MAX_ESCALA_INDICADOR = 10.0
MAX_QTD_AVALIACOES = 50
DIMENSION_INDICATORS = {
    "acadêmica": {
        "label": "dimensão acadêmica",
        "indicators": [
            ("ian", "Índice de Adequação de Nível"),
            ("ida", "Indicador de Desempenho Acadêmico"),
            ("ieg", "Indicador de Engajamento"),
        ],
        "guidance": "priorizar reforço de aprendizagem, acompanhamento da adequação de nível e regularidade nas rotinas acadêmicas.",
    },
    "psicossocial": {
        "label": "dimensão psicossocial",
        "indicators": [
            ("iaa", "Indicador de Autoavaliação"),
            ("ips", "Indicador Psicossocial"),
        ],
        "guidance": "reforçar o acompanhamento emocional, a percepção de si e o suporte relacional no dia a dia do(a) estudante.",
    },
    "psicopedagógica": {
        "label": "dimensão psicopedagógica",
        "indicators": [
            ("ipp", "Indicador Psicopedagógico"),
            ("ipv", "Indicador de Ponto de Virada"),
        ],
        "guidance": "intensificar o acompanhamento psicopedagógico e investigar barreiras de aprendizagem que possam estar travando a evolução.",
    },
}


def _safe_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _evaluate_threshold_grid(y_true: pd.Series, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def _choose_threshold(threshold_df: pd.DataFrame, recall_floor: float) -> float:
    valid = threshold_df[threshold_df["recall"] >= recall_floor].copy()
    if valid.empty:
        valid = threshold_df.copy()
    winner = valid.sort_values(["f1", "precision", "recall", "accuracy"], ascending=False).iloc[0]
    return float(winner["threshold"])


def _prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    expected_cols = [
        "ra",
        "ano_referencia",
        "idade",
        "genero",
        "instituicao_ensino",
        "fase_programa",
        "turma",
        "pedra_ano",
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
        "defasagem",
        "nota_matematica",
        "nota_portugues",
        "nota_ingles",
        "ativo_inativo",
    ]
    for col in expected_cols:
        if col not in data.columns:
            data[col] = np.nan

    numeric_cols = [
        "ano_referencia",
        "idade",
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
        "defasagem",
        "nota_matematica",
        "nota_portugues",
        "nota_ingles",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    base = data.rename(columns={"ano_referencia": "ano_base"}).copy()
    future = data[["ra", "ano_referencia", "ian", "ida", "inde_ano"]].copy()
    future = future.rename(
        columns={
            "ano_referencia": "ano_base",
            "ian": "ian_prox",
            "ida": "ida_prox",
            "inde_ano": "inde_prox",
        }
    )
    future["ano_base"] = future["ano_base"] - 1

    model_df = base.merge(future, on=["ra", "ano_base"], how="left")
    model_df = model_df.sort_values(["ra", "ano_base"]).reset_index(drop=True)

    model_df["delta_ian_prox"] = model_df["ian_prox"] - model_df["ian"]
    model_df["delta_ida_prox"] = model_df["ida_prox"] - model_df["ida"]
    model_df["delta_inde_hist"] = model_df["inde_ano"] - model_df.groupby("ra")["inde_ano"].shift(1)
    model_df["media_notas"] = model_df[["nota_matematica", "nota_portugues", "nota_ingles"]].mean(axis=1)
    model_df["media_comportamental"] = model_df[["iaa", "ieg", "ips", "ipp"]].mean(axis=1)
    model_df["desalinhamento_autoavaliacao"] = model_df["iaa"] - model_df[["ida", "ieg"]].mean(axis=1)

    model_df["target_disponivel"] = model_df["ian_prox"].notna() & model_df["ida_prox"].notna()
    model_df["risco_defasagem_t1"] = (
        (model_df["ian_prox"] <= TARGET_IAN_THRESHOLD)
        | (model_df["ida_prox"] <= TARGET_IDA_THRESHOLD)
        | (model_df["delta_ian_prox"] <= -1.0)
    ).astype("Int64")

    labeled_df = model_df[(model_df["target_disponivel"]) & (model_df["ano_base"].isin([2022, 2023]))].copy()
    return labeled_df


def _feature_name_to_original(feature_name: str, categorical_cols: list[str]) -> str:
    if feature_name.startswith("num__"):
        return feature_name.replace("num__", "", 1)
    if feature_name.startswith("cat__"):
        raw = feature_name.replace("cat__", "", 1)
        for cat_col in sorted(categorical_cols, key=len, reverse=True):
            prefix = f"{cat_col}_"
            if raw.startswith(prefix):
                return cat_col
        return raw
    return feature_name


@st.cache_resource(show_spinner=False)
def train_random_forest_bundle(df: pd.DataFrame) -> dict:
    labeled_df = _prepare_model_dataframe(df)
    if labeled_df.empty:
        raise ValueError("Base rotulada vazia após o preparo dos dados.")

    for col in NUMERIC_FEATURES_ALL + CATEGORICAL_FEATURES_ALL:
        if col not in labeled_df.columns:
            labeled_df[col] = np.nan

    train_df = labeled_df[labeled_df["ano_base"] == 2022].copy()
    test_df = labeled_df[labeled_df["ano_base"] == 2023].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Split temporal inválido: é necessário ter dados de 2022 e 2023.")

    numeric_features: list[str] = []
    for col in NUMERIC_FEATURES_ALL:
        observed_ratio = train_df[col].notna().mean()
        if observed_ratio >= NUMERIC_MIN_OBS_RATIO:
            numeric_features.append(col)

    categorical_features: list[str] = []
    for col in CATEGORICAL_FEATURES_ALL:
        observed_ratio = train_df[col].notna().mean()
        unique_values = train_df[col].dropna().astype(str).nunique()
        if observed_ratio >= CATEGORICAL_MIN_OBS_RATIO and unique_values >= MIN_CATEGORICAL_UNIQUE:
            categorical_features.append(col)

    feature_columns = numeric_features + categorical_features
    if not feature_columns:
        raise ValueError("Nenhuma variável elegível após filtragem de esparsidade.")

    X_train = train_df[feature_columns].copy()
    y_train = train_df["risco_defasagem_t1"].astype(int)
    X_test = test_df[feature_columns].copy()
    y_test = test_df["risco_defasagem_t1"].astype(int)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _safe_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_split=30,
        min_samples_leaf=14,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

    threshold_table = _evaluate_threshold_grid(y_train, oof_prob, THRESHOLD_GRID)
    threshold = _choose_threshold(threshold_table, RECALL_OBJECTIVE)

    pipeline.fit(X_train, y_train)
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "pr_auc": float(average_precision_score(y_test, test_prob)),
        "brier": float(brier_score_loss(y_test, test_prob)),
    }

    default_numeric = {
        col: float(labeled_df[col].median()) if labeled_df[col].notna().any() else 0.0
        for col in numeric_features
    }
    default_categorical = {}
    for col in categorical_features:
        mode_series = labeled_df[col].dropna().astype(str)
        default_categorical[col] = mode_series.mode().iloc[0] if not mode_series.empty else "Não informado"

    category_options = {col: sorted(labeled_df[col].dropna().astype(str).unique().tolist()) for col in categorical_features}

    numeric_limits = {}
    for col in numeric_features:
        valid = labeled_df[col].dropna()
        if valid.empty:
            numeric_limits[col] = {"min": 0.0, "max": 10.0}
            continue
        q01 = float(valid.quantile(0.01))
        q99 = float(valid.quantile(0.99))
        low = min(q01, float(valid.min()))
        high = max(q99, float(valid.max()))
        if low == high:
            high = low + 1.0
        numeric_limits[col] = {"min": low, "max": high}

    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_model = pipeline.named_steps["model"]
    transformed_names = fitted_preprocessor.get_feature_names_out()
    importances = np.asarray(fitted_model.feature_importances_, dtype=float)
    importance_df = pd.DataFrame({"feature_transformada": transformed_names, "importancia": importances})
    importance_df["feature_original"] = importance_df["feature_transformada"].apply(
        lambda value: _feature_name_to_original(value, categorical_features)
    )
    importance_df = (
        importance_df.groupby("feature_original", as_index=False)["importancia"]
        .sum()
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    total_imp = float(importance_df["importancia"].sum())
    importance_df["importancia"] = importance_df["importancia"] / total_imp if total_imp > 0 else 0.0

    risk_bands = {
        "baixo_max": float(max(0.20, threshold - 0.15)),
        "alto_min": float(threshold),
    }

    return {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "risk_bands": risk_bands,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "default_inputs": {**default_numeric, **default_categorical},
        "category_options": category_options,
        "numeric_limits": numeric_limits,
        "feature_importance": importance_df.to_dict(orient="records"),
        "target_rule": "Risco = 1 se (IAN t+1 <= 5,0) ou (IDA t+1 <= 6,0) ou (queda de IAN t+1 <= -1,0).",
        "training_info": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "target_positive_rate_train": float(y_train.mean()),
            "target_positive_rate_test": float(y_test.mean()),
            "metrics_test": metrics,
        },
    }


def _risk_level(probability: float) -> str:
    value = probability * 100
    if value <= 30:
        return "Baixa"
    if value <= 50:
        return "Média"
    return "Alta"


def _risk_message(level: str) -> str:
    if level == "Alta":
        return "Acima de 50%: é extremamente necessário que o(a) estudante tenha acompanhamento psicopedagógico."
    if level == "Média":
        return "Entre 30% e 50%: o(a) estudante necessita de acompanhamento mais intensivo para evitar a defasagem."
    return "Até 30%: o(a) estudante deve manter o padrão atual."


def _valid_score(value: object) -> float | None:
    try:
        score = float(value)
    except Exception:
        return None
    if np.isnan(score):
        return None
    return score


def _format_readable_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} e {items[1]}"
    return f"{', '.join(items[:-1])} e {items[-1]}"


def _dimension_priority_message(indicator_values: dict[str, float | str]) -> str:
    assessments = []
    for dimension_key, config in DIMENSION_INDICATORS.items():
        scores: list[float] = []
        weak_names: list[str] = []
        attention_names: list[str] = []
        for indicator_key, indicator_name in config["indicators"]:
            score = _valid_score(indicator_values.get(indicator_key))
            if score is None:
                continue
            scores.append(score)
            # Regra operacional para leitura gerencial: abaixo de 6 = fragilidade; entre 6 e 7 = atenção.
            if score < 6.0:
                weak_names.append(indicator_name)
            elif score < 7.0:
                attention_names.append(indicator_name)

        if not scores:
            continue

        size = len(scores)
        mean_score = float(np.mean(scores))
        severity = (
            (len(weak_names) / size) * 2.0
            + ((len(weak_names) + len(attention_names)) / size)
            + max(0.0, 7.0 - mean_score) / 4.0
        )
        assessments.append(
            {
                "dimension_key": dimension_key,
                "label": config["label"],
                "guidance": config["guidance"],
                "mean_score": mean_score,
                "severity": severity,
                "weak_names": weak_names,
                "attention_names": attention_names,
            }
        )

    if not assessments:
        return "Não foi possível identificar uma dimensão prioritária com os dados informados."

    assessments = sorted(assessments, key=lambda item: (-item["severity"], item["mean_score"]))
    top = assessments[0]
    second = assessments[1] if len(assessments) > 1 else None

    if top["severity"] < 0.55 and top["mean_score"] >= 7.0:
        return (
            "Os indicadores informados não apontam uma fragilidade dominante em uma dimensão específica. "
            "O foco deve ser manter a consistência atual e acompanhar eventuais oscilações."
        )

    focus_names = top["weak_names"] or top["attention_names"]
    focus_text = _format_readable_list(focus_names)

    if second and abs(top["severity"] - second["severity"]) < 0.20 and second["severity"] >= 0.80:
        second_focus_names = second["weak_names"] or second["attention_names"]
        second_focus_text = _format_readable_list(second_focus_names)
        combined_focus = []
        if focus_text:
            combined_focus.append(focus_text)
        if second_focus_text:
            combined_focus.append(second_focus_text)
        combined_focus_text = _format_readable_list(combined_focus)
        return (
            f"Há maior necessidade de atenção nas {top['label']} e {second['label']}. "
            f"Os sinais mais sensíveis aparecem em {combined_focus_text}. "
            f"Vale {top['guidance']} Também é importante {second['guidance']}"
        )

    if focus_text:
        return (
            f"O principal ponto de atenção está na {top['label']}, especialmente em {focus_text}. "
            f"Vale {top['guidance']}"
        )

    return f"O principal ponto de atenção está na {top['label']}. Vale {top['guidance']}"


def _render_result_card(probability: float, level: str, dimension_message: str) -> None:
    color_map = {
        "Baixa": ("#2A9D8F", "#E8F6F3"),
        "Média": ("#E9C46A", "#FFF8E8"),
        "Alta": ("#E76F51", "#FDEDEC"),
    }
    border_color, bg_color = color_map.get(level, ("#457B9D", "#F4F8FB"))
    risk_message = _risk_message(level)
    st.markdown(
        f"""
        <div style="background:{bg_color}; border:1px solid {border_color}; border-left:6px solid {border_color};
                    border-radius:10px; padding:0.9rem 1rem; margin:0.6rem 0 1rem 0;">
            <div style="font-size:1rem; font-weight:700; color:#264653;">
                Probabilidade de defasagem no próximo ciclo: {probability * 100:.1f}%
            </div>
            <div style="margin-top:0.35rem; color:#264653;">
                Faixa de risco: <strong>{level}</strong>
            </div>
            <div style="margin-top:0.55rem; color:#264653; font-size:0.96rem;">
                {risk_message}
            </div>
            <div style="margin-top:0.65rem; padding-top:0.65rem; border-top:1px solid rgba(38, 70, 83, 0.16);
                        color:#264653; font-size:0.95rem; line-height:1.5;">
                {dimension_message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_feature_importance(bundle: dict) -> None:
    importance_df = pd.DataFrame(bundle["feature_importance"]).head(8).copy()
    if importance_df.empty:
        return

    importance_df["variavel"] = importance_df["feature_original"].map(FEATURE_LABELS).fillna(importance_df["feature_original"])
    importance_df["importancia_pct"] = importance_df["importancia"] * 100
    st.markdown("**Variáveis mais influentes no modelo (Random Forest)**")
    st.bar_chart(
        importance_df.set_index("variavel")["importancia_pct"],
        height=260,
    )


def _has_feature(bundle: dict, feature: str) -> bool:
    return feature in bundle["feature_columns"]


def _safe_int(
    value: object,
    fallback: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    try:
        resolved = int(round(float(value)))
    except Exception:
        resolved = int(fallback)
    if min_value is not None:
        resolved = max(int(min_value), resolved)
    if max_value is not None:
        resolved = min(int(max_value), resolved)
    return resolved


def _normalize_phase_value(value: object) -> str | None:
    phase_num = _extract_phase_code(None if value is None else str(value))
    if phase_num is None:
        return None
    phase_int = int(round(phase_num))
    if phase_int <= 0:
        return "ALFA"
    if 1 <= phase_int <= 8:
        return f"FASE {phase_int}"
    return None


def _render_numeric_input(
    bundle: dict,
    values: dict[str, float | str],
    feature: str,
    *,
    key_prefix: str,
    min_value: float | None = None,
    max_value: float | None = None,
    step: float | None = None,
    value_format: str = "%.2f",
) -> None:
    if feature not in bundle["numeric_features"]:
        return
    limits = bundle["numeric_limits"][feature]
    min_v = float(limits["min"]) if min_value is None else float(min_value)
    max_v = float(limits["max"]) if max_value is None else float(max_value)
    default_v = float(values.get(feature, bundle["default_inputs"].get(feature, min_v)))
    default_v = min(max(default_v, min_v), max_v)
    resolved_step = step if step is not None else (0.1 if max_v <= 20 else 0.5)
    values[feature] = st.number_input(
        FEATURE_LABELS.get(feature, feature),
        min_value=min_v,
        max_value=max_v,
        value=default_v,
        step=resolved_step,
        format=value_format,
        key=f"{key_prefix}_{feature}",
    )


def _render_categorical_input(
    bundle: dict,
    values: dict[str, float | str],
    feature: str,
    *,
    key_prefix: str,
    label: str | None = None,
) -> None:
    if feature not in bundle["categorical_features"]:
        return
    options = bundle["category_options"].get(feature, [])
    default_value = str(values.get(feature, bundle["default_inputs"].get(feature, "Não informado")))
    if not options:
        options = [default_value]
    if default_value not in options:
        options = [default_value] + options
    default_idx = options.index(default_value) if default_value in options else 0
    values[feature] = st.selectbox(
        label or FEATURE_LABELS.get(feature, feature),
        options=options,
        index=default_idx,
        key=f"{key_prefix}_{feature}",
    )


def _extract_phase_code(phase_label: str | None) -> float | None:
    if phase_label is None:
        return None
    raw = str(phase_label).strip().upper()
    if not raw:
        return None
    if "ALFA" in raw:
        return 0.0
    match = re.search(r"(\d+)", raw)
    if match:
        return float(match.group(1))
    return None


def _safe_ratio(total_value: float | int, count_value: float | int) -> float:
    count = max(float(count_value), 1.0)
    return float(total_value) / count


def _compute_ian_from_defasagem(defasagem: float) -> float:
    # Regra operacional coerente com o documento + padrão observado na base histórica.
    if defasagem <= -3:
        return 2.5
    if defasagem <= -1:
        return 5.0
    return 10.0


def _clip_to_model_limits(bundle: dict, feature: str, value: float) -> float:
    limits = bundle.get("numeric_limits", {}).get(feature)
    if not limits:
        return float(value)
    return float(np.clip(value, float(limits["min"]), float(limits["max"])))


def _sync_derived_features(bundle: dict, values: dict[str, float | str]) -> None:
    nota_matematica = float(values.get("nota_matematica", bundle["default_inputs"].get("nota_matematica", 0.0)))
    nota_portugues = float(values.get("nota_portugues", bundle["default_inputs"].get("nota_portugues", 0.0)))
    nota_ingles = float(values.get("nota_ingles", bundle["default_inputs"].get("nota_ingles", 0.0)))

    # IDA = média simples das três notas (conforme documento).
    ida_calc = (nota_matematica + nota_portugues + nota_ingles) / 3.0
    values["ida"] = _clip_to_model_limits(bundle, "ida", ida_calc)

    # IEG = soma das pontuações das tarefas / número de tarefas.
    ieg_total = float(values.get("ieg_total_pontos", values.get("ieg", bundle["default_inputs"].get("ieg", 0.0)) * 10.0))
    ieg_qtd = float(values.get("ieg_qtd_tarefas", 10))
    ieg_calc = _safe_ratio(ieg_total, ieg_qtd)
    values["ieg"] = _clip_to_model_limits(bundle, "ieg", ieg_calc)

    # IAA = soma das respostas / número de perguntas.
    iaa_total = float(values.get("iaa_soma_respostas", values.get("iaa", bundle["default_inputs"].get("iaa", 0.0)) * 10.0))
    iaa_qtd = float(values.get("iaa_qtd_perguntas", 10))
    iaa_calc = _safe_ratio(iaa_total, iaa_qtd)
    values["iaa"] = _clip_to_model_limits(bundle, "iaa", iaa_calc)

    # IPS = soma das pontuações dos avaliadores / número de avaliadores.
    ips_total = float(values.get("ips_soma_avaliacoes", values.get("ips", bundle["default_inputs"].get("ips", 0.0)) * 3.0))
    ips_qtd = float(values.get("ips_qtd_avaliadores", 3))
    ips_calc = _safe_ratio(ips_total, ips_qtd)
    values["ips"] = _clip_to_model_limits(bundle, "ips", ips_calc)

    # IPP = soma das avaliações psicopedagógicas / número de avaliações.
    ipp_total = float(values.get("ipp_soma_avaliacoes", values.get("ipp", bundle["default_inputs"].get("ipp", 0.0)) * 3.0))
    ipp_qtd = float(values.get("ipp_qtd_avaliacoes", 3))
    ipp_calc = _safe_ratio(ipp_total, ipp_qtd)
    values["ipp"] = _clip_to_model_limits(bundle, "ipp", ipp_calc)

    # IAN via defasagem: D = fase efetiva - fase ideal.
    fase_efetiva_num = _extract_phase_code(str(values.get("fase_programa", "")))
    fase_ideal_num = float(values.get("fase_ideal_num", fase_efetiva_num if fase_efetiva_num is not None else 7.0))
    values["fase_efetiva_num"] = fase_efetiva_num if fase_efetiva_num is not None else np.nan
    values["fase_ideal_num"] = fase_ideal_num

    if fase_efetiva_num is not None:
        defasagem_calc = float(fase_efetiva_num - fase_ideal_num)
        values["defasagem"] = _clip_to_model_limits(bundle, "defasagem", defasagem_calc)
        ian_calc = _compute_ian_from_defasagem(defasagem_calc)
        values["ian"] = _clip_to_model_limits(bundle, "ian", ian_calc)

    # IPV (proxy operacional): média de IDA, IEG e IPS.
    ipv_calc = np.mean(
        [
            float(values.get("ida", bundle["default_inputs"].get("ida", 0.0))),
            float(values.get("ieg", bundle["default_inputs"].get("ieg", 0.0))),
            float(values.get("ips", bundle["default_inputs"].get("ips", 0.0))),
        ]
    )
    values["ipv"] = _clip_to_model_limits(bundle, "ipv", float(ipv_calc))

    if _has_feature(bundle, "media_notas"):
        notas = [nota_matematica, nota_portugues, nota_ingles]
        values["media_notas"] = float(np.mean(notas))

    if _has_feature(bundle, "media_comportamental"):
        comp = [
            float(values.get("iaa", bundle["default_inputs"].get("iaa", 0.0))),
            float(values.get("ieg", bundle["default_inputs"].get("ieg", 0.0))),
            float(values.get("ips", bundle["default_inputs"].get("ips", 0.0))),
            float(values.get("ipp", bundle["default_inputs"].get("ipp", 0.0))),
        ]
        values["media_comportamental"] = float(np.mean(comp))

    if _has_feature(bundle, "desalinhamento_autoavaliacao"):
        iaa = float(values.get("iaa", bundle["default_inputs"].get("iaa", 0.0)))
        ida = float(values.get("ida", bundle["default_inputs"].get("ida", 0.0)))
        ieg = float(values.get("ieg", bundle["default_inputs"].get("ieg", 0.0)))
        values["desalinhamento_autoavaliacao"] = _clip_to_model_limits(
            bundle,
            "desalinhamento_autoavaliacao",
            float(iaa - ((ida + ieg) / 2)),
        )


def _render_probability_gauge(probability: float) -> None:
    value = float(np.clip(probability * 100, 0, 100))
    faixa_baixa_fim = 30.0
    faixa_media_fim = 50.0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": "%",
                "valueformat": ".1f",
                "font": {"size": 108, "color": "#7A8094"},
            },
            title={
                "text": "Probabilidade de risco de defasagem",
                "font": {"size": 40, "color": "#72788C"},
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 30, 50, 70, 100],
                    "ticktext": ["0", "30", "50", "70", "100"],
                    "tickwidth": 2,
                    "tickcolor": "#72788C",
                    "tickfont": {"size": 28, "color": "#72788C"},
                },
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 2,
                "bordercolor": "#4F5568",
                "bar": {
                    "color": "#000000",
                    "thickness": 0.40,
                    "line": {"color": "#000000", "width": 1},
                },
                "steps": [
                    {
                        "range": [0, faixa_baixa_fim],
                        "color": "#2DAA31",
                        "thickness": 1.0,
                    },
                    {
                        "range": [faixa_baixa_fim, faixa_media_fim],
                        "color": "#F6BD00",
                        "thickness": 1.0,
                    },
                    {
                        "range": [faixa_media_fim, 100],
                        "color": "#DE2227",
                        "thickness": 1.0,
                    },
                ],
            },
        )
    )

    fig.update_layout(
        height=520,
        margin=dict(l=18, r=18, t=110, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial", "color": "#72788C"},
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _render_technical_popover(bundle: dict) -> None:
    info = bundle["training_info"]
    metrics = info["metrics_test"]

    with st.popover("ℹ️", help="Detalhes técnicos do modelo"):
        st.markdown("**Detalhes técnicos do modelo**")
        st.caption("Seção de apoio para leitura analítica. O uso operacional está no formulário principal.")

        st.markdown("**Regra de definição do risco**")
        st.write(bundle["target_rule"])

        st.markdown("**Amostras usadas no treinamento**")
        st.write(f"Treino (2022): {info['train_rows']:,}".replace(",", "."))
        st.write(f"Teste temporal (2023): {info['test_rows']:,}".replace(",", "."))

        st.markdown("**Desempenho no teste temporal (2023)**")
        c1, c2 = st.columns(2)
        c1.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
        c2.metric("AUC-ROC", f"{metrics['roc_auc']:.3f}")
        c3, c4 = st.columns(2)
        c3.metric("Precisão", f"{metrics['precision'] * 100:.1f}%")
        c4.metric("F1-score", f"{metrics['f1'] * 100:.1f}%")

        _render_feature_importance(bundle)


def render_modelo_preditivo_tab(df: pd.DataFrame) -> None:
    try:
        bundle = train_random_forest_bundle(df)
    except Exception as error:
        st.error(f"Falha ao preparar o modelo preditivo: {error}")
        st.stop()
        return

    header_col, info_col = st.columns([0.92, 0.08])
    with header_col:
        st.subheader("Probabilidade de risco de defasagem")
        st.markdown(
            """
            Preencha os dados do(a) estudante para estimar a chance de entrada em defasagem no próximo ciclo.
            """
        )
    with info_col:
        _render_technical_popover(bundle)

    values: dict[str, float | str] = {
        feature: bundle["default_inputs"].get(feature) for feature in bundle["feature_columns"]
    }

    with st.container():
        st.markdown("### 1. Dados do Aluno")
        c1, c2, c3 = st.columns(3)
        with c1:
            _render_numeric_input(
                bundle,
                values,
                "idade",
                key_prefix="aluno",
                min_value=DOCUMENTED_LIMITS["idade"]["min"],
                max_value=DOCUMENTED_LIMITS["idade"]["max"],
                step=DOCUMENTED_LIMITS["idade"]["step"],
                value_format=DOCUMENTED_LIMITS["idade"]["format"],
            )
        with c2:
            _render_categorical_input(bundle, values, "genero", key_prefix="aluno")
        with c3:
            default_phase_value = _normalize_phase_value(values.get("fase_programa")) or "FASE 5"
            default_phase_label = PHASE_VALUE_TO_LABEL.get(default_phase_value, PHASE_LABELS[0])
            selected_phase_label = st.selectbox(
                "Fase efetiva (programa atual)",
                options=PHASE_LABELS,
                index=PHASE_LABELS.index(default_phase_label),
                key="pred_fase_efetiva",
                help="A fase representa o nível de aprendizado do(a) estudante na Associação Passos Mágicos.",
            )
            values["fase_programa"] = PHASE_LABEL_TO_VALUE[selected_phase_label]

        st.caption(
            "Na Passos Mágicos, a fase representa o nível de aprendizado: "
            "Alfa corresponde à alfabetização e as Fases 1 a 8 representam a progressão educacional."
        )

        f1, f2 = st.columns(2)
        fase_efetiva_default = _extract_phase_code(str(values.get("fase_programa", "")))
        if fase_efetiva_default is None:
            fase_efetiva_default = 7.0
        fase_ideal_default = _normalize_phase_value(values.get("fase_ideal_num")) or _normalize_phase_value(
            fase_efetiva_default
        )
        if fase_ideal_default is None:
            fase_ideal_default = "FASE 7"
        with f1:
            selected_ideal_label = st.selectbox(
                "Fase ideal (esperada para idade/série)",
                options=PHASE_LABELS,
                index=PHASE_LABELS.index(PHASE_VALUE_TO_LABEL[fase_ideal_default]),
                key="pred_fase_ideal",
                help="Usada no cálculo de defasagem: D = fase efetiva - fase ideal.",
            )
            phase_ideal_num = _extract_phase_code(PHASE_LABEL_TO_VALUE[selected_ideal_label])
            values["fase_ideal_num"] = 0.0 if phase_ideal_num is None else float(phase_ideal_num)
        with f2:
            fase_efetiva_num = _extract_phase_code(str(values.get("fase_programa", "")))
            if fase_efetiva_num is None:
                st.info("Não foi possível identificar o número da fase efetiva.")
            else:
                defasagem_preview = fase_efetiva_num - float(values.get("fase_ideal_num", fase_efetiva_num))
                st.metric("Defasagem prévia (D = efetiva - ideal)", f"{defasagem_preview:+.0f}")
                st.caption(
                    f"Fase efetiva selecionada: {PHASE_VALUE_TO_LABEL.get(str(values.get('fase_programa')), '-')}."
                )

        st.markdown("### 2. Notas Acadêmicas")
        n1, n2, n3 = st.columns(3)
        with n1:
            _render_numeric_input(
                bundle,
                values,
                "nota_matematica",
                key_prefix="nota",
                min_value=DOCUMENTED_LIMITS["nota_matematica"]["min"],
                max_value=DOCUMENTED_LIMITS["nota_matematica"]["max"],
                step=DOCUMENTED_LIMITS["nota_matematica"]["step"],
                value_format=DOCUMENTED_LIMITS["nota_matematica"]["format"],
            )
        with n2:
            _render_numeric_input(
                bundle,
                values,
                "nota_portugues",
                key_prefix="nota",
                min_value=DOCUMENTED_LIMITS["nota_portugues"]["min"],
                max_value=DOCUMENTED_LIMITS["nota_portugues"]["max"],
                step=DOCUMENTED_LIMITS["nota_portugues"]["step"],
                value_format=DOCUMENTED_LIMITS["nota_portugues"]["format"],
            )
        with n3:
            _render_numeric_input(
                bundle,
                values,
                "nota_ingles",
                key_prefix="nota",
                min_value=DOCUMENTED_LIMITS["nota_ingles"]["min"],
                max_value=DOCUMENTED_LIMITS["nota_ingles"]["max"],
                step=DOCUMENTED_LIMITS["nota_ingles"]["step"],
                value_format=DOCUMENTED_LIMITS["nota_ingles"]["format"],
            )

        st.markdown("### 3. Bases para cálculo dos indicadores")
        st.caption(
            "Escala adotada no preenchimento: 0 a 10 para notas/pontuações. "
            "Os indicadores são calculados automaticamente pelas fórmulas do PEDE."
        )
        d1, d2, d3 = st.columns(3)

        with d1:
            with st.container(border=True):
                st.markdown("**Dimensão Acadêmica**")
                st.caption("IEG = soma das tarefas / número de tarefas")
                ieg_base = float(values.get("ieg", bundle["default_inputs"].get("ieg", 7.0)))
                values["ieg_qtd_tarefas"] = st.number_input(
                    "Número de tarefas (IEG)",
                    min_value=1,
                    max_value=MAX_QTD_AVALIACOES,
                    value=_safe_int(
                        values.get("ieg_qtd_tarefas", 10),
                        10,
                        min_value=1,
                        max_value=MAX_QTD_AVALIACOES,
                    ),
                    step=1,
                )
                ieg_total_max = float(values["ieg_qtd_tarefas"]) * MAX_ESCALA_INDICADOR
                ieg_total_default = float(values.get("ieg_total_pontos", ieg_base * float(values["ieg_qtd_tarefas"])))
                values["ieg_total_pontos"] = st.number_input(
                    "Soma das pontuações das tarefas (IEG)",
                    min_value=0.0,
                    max_value=float(ieg_total_max),
                    value=float(np.clip(ieg_total_default, 0.0, ieg_total_max)),
                    step=0.1,
                    format="%.2f",
                )

        with d2:
            with st.container(border=True):
                st.markdown("**Dimensão Psicossocial**")
                st.caption("IAA = soma das respostas / número de perguntas")
                iaa_base = float(values.get("iaa", bundle["default_inputs"].get("iaa", 7.0)))
                values["iaa_qtd_perguntas"] = st.number_input(
                    "Número de perguntas da autoavaliação (IAA)",
                    min_value=1,
                    max_value=MAX_QTD_AVALIACOES,
                    value=_safe_int(
                        values.get("iaa_qtd_perguntas", 10),
                        10,
                        min_value=1,
                        max_value=MAX_QTD_AVALIACOES,
                    ),
                    step=1,
                )
                iaa_total_max = float(values["iaa_qtd_perguntas"]) * MAX_ESCALA_INDICADOR
                iaa_total_default = float(values.get("iaa_soma_respostas", iaa_base * float(values["iaa_qtd_perguntas"])))
                values["iaa_soma_respostas"] = st.number_input(
                    "Soma das respostas da autoavaliação (IAA)",
                    min_value=0.0,
                    max_value=float(iaa_total_max),
                    value=float(np.clip(iaa_total_default, 0.0, iaa_total_max)),
                    step=0.1,
                    format="%.2f",
                )
                st.caption("IPS = soma das avaliações dos psicólogos / número de avaliadores")
                ips_base = float(values.get("ips", bundle["default_inputs"].get("ips", 6.0)))
                values["ips_qtd_avaliadores"] = st.number_input(
                    "Número de avaliadores psicossociais (IPS)",
                    min_value=1,
                    max_value=MAX_QTD_AVALIACOES,
                    value=_safe_int(
                        values.get("ips_qtd_avaliadores", 3),
                        3,
                        min_value=1,
                        max_value=MAX_QTD_AVALIACOES,
                    ),
                    step=1,
                )
                ips_total_max = float(values["ips_qtd_avaliadores"]) * MAX_ESCALA_INDICADOR
                ips_total_default = float(
                    values.get("ips_soma_avaliacoes", ips_base * float(values["ips_qtd_avaliadores"]))
                )
                values["ips_soma_avaliacoes"] = st.number_input(
                    "Soma das avaliações psicossociais (IPS)",
                    min_value=0.0,
                    max_value=float(ips_total_max),
                    value=float(np.clip(ips_total_default, 0.0, ips_total_max)),
                    step=0.1,
                    format="%.2f",
                )

        with d3:
            with st.container(border=True):
                st.markdown("**Dimensão Psicopedagógica**")
                st.caption("IPP = soma das avaliações psicopedagógicas / número de avaliações")
                ipp_base = float(values.get("ipp", bundle["default_inputs"].get("ipp", 7.0)))
                values["ipp_qtd_avaliacoes"] = st.number_input(
                    "Número de avaliações psicopedagógicas (IPP)",
                    min_value=1,
                    max_value=MAX_QTD_AVALIACOES,
                    value=_safe_int(
                        values.get("ipp_qtd_avaliacoes", 3),
                        3,
                        min_value=1,
                        max_value=MAX_QTD_AVALIACOES,
                    ),
                    step=1,
                )
                ipp_total_max = float(values["ipp_qtd_avaliacoes"]) * MAX_ESCALA_INDICADOR
                ipp_total_default = float(
                    values.get("ipp_soma_avaliacoes", ipp_base * float(values["ipp_qtd_avaliacoes"]))
                )
                values["ipp_soma_avaliacoes"] = st.number_input(
                    "Soma das avaliações psicopedagógicas (IPP)",
                    min_value=0.0,
                    max_value=float(ipp_total_max),
                    value=float(np.clip(ipp_total_default, 0.0, ipp_total_max)),
                    step=0.1,
                    format="%.2f",
                )
                st.caption("IPV calculado automaticamente por proxy operacional: média de IDA, IEG e IPS.")

        preview_values = dict(values)
        _sync_derived_features(bundle, preview_values)

        def _fmt_indicator(value: object) -> str:
            try:
                number = float(value)
                if np.isnan(number):
                    return "-"
                return f"{number:.2f}".replace(".", ",")
            except Exception:
                return "-"

        st.markdown("### Indicadores calculados automaticamente")

        def _indicator_item(sigla: str, nome: str, value: object) -> str:
            return (
                "<div class='pred-ind-item'>"
                "<div class='pred-ind-meta'>"
                f"<span class='pred-ind-code'>{sigla}</span>"
                f"<span class='pred-ind-name'>{nome}</span>"
                "</div>"
                f"<span class='pred-ind-score'>{_fmt_indicator(value)}</span>"
                "</div>"
            )

        st.markdown(
            """
            <style>
            .pred-ind-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1rem;
                margin: 0.35rem 0 1rem 0;
            }
            .pred-ind-card {
                border-radius: 14px;
                padding: 0.85rem 0.9rem;
                box-shadow: 0 1px 3px rgba(31, 41, 55, 0.08);
            }
            .pred-ind-card-acad {
                background: linear-gradient(180deg, #F3F8FF 0%, #FCFDFF 100%);
                border: 1px solid #CFE0F5;
            }
            .pred-ind-card-psico {
                background: linear-gradient(180deg, #F6FAF3 0%, #FCFEFB 100%);
                border: 1px solid #D6E7CF;
            }
            .pred-ind-card-pedag {
                background: linear-gradient(180deg, #FFF8F2 0%, #FFFDFB 100%);
                border: 1px solid #F2DEC9;
            }
            .pred-ind-title {
                font-size: 0.94rem;
                font-weight: 700;
                color: #1D3557;
                margin-bottom: 0.2rem;
            }
            .pred-ind-subtitle {
                font-size: 0.76rem;
                color: #5E6C84;
                margin-bottom: 0.55rem;
            }
            .pred-ind-item {
                display: grid;
                grid-template-columns: 1fr auto;
                gap: 0.6rem;
                align-items: center;
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #E3EAF4;
                border-radius: 10px;
                padding: 0.5rem 0.6rem;
                margin-bottom: 0.45rem;
            }
            .pred-ind-item:last-child { margin-bottom: 0; }
            .pred-ind-meta {
                display: grid;
                grid-template-columns: auto 1fr;
                gap: 0.45rem;
                align-items: center;
            }
            .pred-ind-code {
                background: #2E5FA7;
                color: #FFFFFF;
                font-size: 0.68rem;
                font-weight: 700;
                border-radius: 999px;
                padding: 0.16rem 0.45rem;
                line-height: 1.25;
                letter-spacing: 0.02em;
            }
            .pred-ind-name {
                font-size: 0.79rem;
                font-weight: 600;
                color: #415066;
                line-height: 1.2;
            }
            .pred-ind-score {
                font-size: 1.08rem;
                font-weight: 700;
                color: #1E293B;
                min-width: 56px;
                text-align: right;
            }
            @media (max-width: 900px) {
                .pred-ind-grid {
                    grid-template-columns: 1fr;
                }
                .pred-ind-item {
                    grid-template-columns: 1fr;
                }
                .pred-ind-score {
                    text-align: left;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Os indicadores abaixo são calculados automaticamente a partir das entradas informadas.")
        st.markdown(
            f"""
            <div class="pred-ind-grid">
                <div class="pred-ind-card pred-ind-card-acad">
                    <div class="pred-ind-title">Dimensão Acadêmica</div>
                    <div class="pred-ind-subtitle">Indicadores de desempenho e adequação escolar</div>
                    {_indicator_item("IAN", "Índice de Adequação de Nível", preview_values.get("ian"))}
                    {_indicator_item("IDA", "Indicador de Desempenho Acadêmico", preview_values.get("ida"))}
                    {_indicator_item("IEG", "Indicador de Engajamento", preview_values.get("ieg"))}
                </div>
                <div class="pred-ind-card pred-ind-card-psico">
                    <div class="pred-ind-title">Dimensão Psicossocial</div>
                    <div class="pred-ind-subtitle">Indicadores de autoavaliação e contexto psicossocial</div>
                    {_indicator_item("IAA", "Indicador de Autoavaliação", preview_values.get("iaa"))}
                    {_indicator_item("IPS", "Indicador Psicossocial", preview_values.get("ips"))}
                </div>
                <div class="pred-ind-card pred-ind-card-pedag">
                    <div class="pred-ind-title">Dimensão Psicopedagógica</div>
                    <div class="pred-ind-subtitle">Indicadores de suporte e ponto de virada educacional</div>
                    {_indicator_item("IPP", "Indicador Psicopedagógico", preview_values.get("ipp"))}
                    {_indicator_item("IPV", "Indicador de Ponto de Virada", preview_values.get("ipv"))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Os cálculos acima atualizam imediatamente. Use o botão abaixo apenas para gerar a predição.")
        submitted = st.button("Calcular probabilidade de risco", type="primary", use_container_width=True)

    if not submitted:
        return

    _sync_derived_features(bundle, values)
    input_row = pd.DataFrame([{feature: values.get(feature) for feature in bundle["feature_columns"]}])
    probability = float(bundle["pipeline"].predict_proba(input_row)[:, 1][0])
    level = _risk_level(probability)
    dimension_message = _dimension_priority_message(values)

    _render_probability_gauge(probability)
    _render_result_card(probability, level, dimension_message)
