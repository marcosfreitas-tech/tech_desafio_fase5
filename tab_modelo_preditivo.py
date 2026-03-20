from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
N_FOLDS = 5
TARGET_IAN_THRESHOLD = 5.0
RECALL_OBJECTIVE = 0.80
THRESHOLD_GRID = np.round(np.linspace(0.10, 0.90, 33), 3)
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_BUNDLE_CANDIDATES = (
    PROJECT_ROOT / "models" / "modelo_risco_defasagem.pkl",
    PROJECT_ROOT / "artifacts" / "modelo_risco_defasagem.pkl",
)

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
        model_df["ian_prox"] <= TARGET_IAN_THRESHOLD
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


def _resolve_model_bundle_path() -> Path:
    for candidate in MODEL_BUNDLE_CANDIDATES:
        if candidate.exists():
            return candidate
    candidate_list = "\n".join([f"- {path}" for path in MODEL_BUNDLE_CANDIDATES])
    raise FileNotFoundError(
        "Artefato do modelo nao encontrado. Gere o modelo no notebook `scripts/2_Modelo_Preditivo.ipynb` "
        "e salve o `modelo_risco_defasagem.pkl` em uma das pastas abaixo:\n"
        f"{candidate_list}"
    )


def _build_numeric_limits(reference_df: pd.DataFrame, numeric_features: list[str]) -> dict[str, dict[str, float]]:
    numeric_limits: dict[str, dict[str, float]] = {}
    for col in numeric_features:
        if col in reference_df.columns:
            valid = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        else:
            valid = pd.Series(dtype="float64")

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
    return numeric_limits


def _normalize_feature_importance(bundle: dict) -> list[dict[str, float | str]]:
    def _normalize_records(records: object) -> list[dict[str, float | str]]:
        if not isinstance(records, list) or not records:
            return []

        importance_df = pd.DataFrame(records).copy()
        if "feature_original" not in importance_df.columns:
            if "feature" in importance_df.columns:
                importance_df["feature_original"] = importance_df["feature"]
            elif "variavel" in importance_df.columns:
                importance_df["feature_original"] = importance_df["variavel"]

        if "importancia" not in importance_df.columns:
            if "importance" in importance_df.columns:
                importance_df["importancia"] = pd.to_numeric(importance_df["importance"], errors="coerce")
            elif "importancia_normalizada" in importance_df.columns:
                importance_df["importancia"] = pd.to_numeric(importance_df["importancia_normalizada"], errors="coerce")

        if "feature_original" not in importance_df.columns or "importancia" not in importance_df.columns:
            return []

        normalized = (
            importance_df[["feature_original", "importancia"]]
            .dropna(subset=["feature_original", "importancia"])
            .copy()
        )
        if normalized.empty:
            return []

        normalized["feature_original"] = normalized["feature_original"].astype(str)
        normalized["importancia"] = pd.to_numeric(normalized["importancia"], errors="coerce").fillna(0.0)
        total = float(normalized["importancia"].sum())
        if total > 0:
            normalized["importancia"] = normalized["importancia"] / total
        return normalized.to_dict(orient="records")

    normalized = _normalize_records(bundle.get("feature_importance"))
    if normalized:
        return normalized
    return _normalize_records(bundle.get("feature_importance_best_model"))


def _normalize_training_info(bundle: dict) -> dict:
    raw_info = bundle.get("training_info")
    if not isinstance(raw_info, dict):
        raw_info = {}

    raw_metrics = raw_info.get("metrics_test")
    if not isinstance(raw_metrics, dict):
        raw_metrics = raw_info.get("best_model_test_metrics")
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}

    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    normalized_metrics = {}
    for key in metric_keys:
        try:
            normalized_metrics[key] = float(raw_metrics.get(key, 0.0))
        except (TypeError, ValueError):
            normalized_metrics[key] = 0.0

    try:
        train_rows = int(raw_info.get("train_rows", 0))
    except (TypeError, ValueError):
        train_rows = 0
    try:
        test_rows = int(raw_info.get("test_rows", 0))
    except (TypeError, ValueError):
        test_rows = 0
    try:
        positive_rate_train = float(raw_info.get("target_positive_rate_train", 0.0))
    except (TypeError, ValueError):
        positive_rate_train = 0.0
    try:
        positive_rate_test = float(raw_info.get("target_positive_rate_test", 0.0))
    except (TypeError, ValueError):
        positive_rate_test = 0.0

    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "target_positive_rate_train": positive_rate_train,
        "target_positive_rate_test": positive_rate_test,
        "metrics_test": normalized_metrics,
    }


@st.cache_resource(show_spinner=False)
def load_model_bundle(df: pd.DataFrame, model_bundle_path: Path, model_bundle_mtime_ns: int) -> dict:
    _ = model_bundle_mtime_ns  # forca invalidacao do cache quando o artefato mudar

    with model_bundle_path.open("rb") as file:
        bundle = pickle.load(file)

    if not isinstance(bundle, dict):
        raise ValueError("Artefato invalido: o arquivo de modelo nao contem um dicionario de configuracao.")

    pipeline = bundle.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "predict_proba"):
        raise ValueError("Artefato invalido: pipeline ausente ou sem suporte a `predict_proba`.")

    model = getattr(pipeline, "named_steps", {}).get("model")
    if model is None:
        model = pipeline

    feature_columns = [str(col) for col in bundle.get("feature_columns", [])]
    numeric_features = [str(col) for col in bundle.get("numeric_features", [])]
    categorical_features = [str(col) for col in bundle.get("categorical_features", [])]
    if not feature_columns:
        raise ValueError("Artefato invalido: `feature_columns` ausente ou vazio.")

    labeled_df = _prepare_model_dataframe(df)
    reference_df = labeled_df if not labeled_df.empty else df.copy()
    for col in feature_columns:
        if col not in reference_df.columns:
            reference_df[col] = np.nan

    default_inputs = bundle.get("default_inputs")
    if not isinstance(default_inputs, dict):
        default_inputs = {}
    default_inputs = default_inputs.copy()
    for col in feature_columns:
        if col in default_inputs:
            continue
        if col in numeric_features:
            series = pd.to_numeric(reference_df[col], errors="coerce").dropna()
            default_inputs[col] = float(series.median()) if not series.empty else 0.0
        else:
            mode_series = reference_df[col].dropna().astype(str)
            default_inputs[col] = mode_series.mode().iloc[0] if not mode_series.empty else "Nao informado"

    category_options = bundle.get("category_options")
    if not isinstance(category_options, dict):
        category_options = {}
    normalized_options: dict[str, list[str]] = {}
    for col in categorical_features:
        options = category_options.get(col, [])
        if not isinstance(options, list):
            options = []
        options = [str(option) for option in options]
        if not options:
            options = sorted(reference_df[col].dropna().astype(str).unique().tolist())
        default_value = str(default_inputs.get(col, "Nao informado"))
        if default_value and default_value not in options:
            options = [default_value] + options
        normalized_options[col] = options

    computed_limits = _build_numeric_limits(reference_df, numeric_features)
    numeric_limits_raw = bundle.get("numeric_limits")
    if not isinstance(numeric_limits_raw, dict):
        numeric_limits_raw = {}
    numeric_limits = {}
    for col in numeric_features:
        raw_limits = numeric_limits_raw.get(col)
        if isinstance(raw_limits, dict) and "min" in raw_limits and "max" in raw_limits:
            try:
                min_v = float(raw_limits["min"])
                max_v = float(raw_limits["max"])
                if min_v < max_v:
                    numeric_limits[col] = {"min": min_v, "max": max_v}
                    continue
            except (TypeError, ValueError):
                pass
        numeric_limits[col] = computed_limits[col]

    feature_importance = _normalize_feature_importance(bundle)
    if not feature_importance and hasattr(model, "feature_importances_"):
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            transformed_names = preprocessor.get_feature_names_out()
            importances = np.asarray(model.feature_importances_, dtype=float)
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
            feature_importance = importance_df.to_dict(orient="records")

    try:
        threshold = float(bundle.get("threshold", 0.5))
    except (TypeError, ValueError):
        threshold = 0.5

    risk_bands_raw = bundle.get("risk_bands")
    if not isinstance(risk_bands_raw, dict):
        risk_bands_raw = {}
    try:
        baixo_max = float(risk_bands_raw.get("baixo_max", max(0.20, threshold - 0.15)))
    except (TypeError, ValueError):
        baixo_max = float(max(0.20, threshold - 0.15))
    try:
        alto_min = float(risk_bands_raw.get("alto_min", threshold))
    except (TypeError, ValueError):
        alto_min = float(threshold)
    risk_bands = {"baixo_max": baixo_max, "alto_min": alto_min}

    target_rule = bundle.get("target_rule")
    if target_rule is None:
        target_rule = "Risco = 1 se (IAN t+1 <= 5,0)."

    raw_model_hyperparameters: dict[str, object] = {}
    if hasattr(model, "get_params"):
        try:
            raw_model_hyperparameters = dict(model.get_params())
        except Exception:
            raw_model_hyperparameters = {}

    model_info = bundle.get("model_info")
    if not isinstance(model_info, dict):
        model_info = {}
    try:
        recall_objective = float(model_info.get("recall_objective", RECALL_OBJECTIVE))
    except (TypeError, ValueError):
        recall_objective = float(RECALL_OBJECTIVE)
    try:
        threshold_grid_min = float(model_info.get("threshold_grid_min", float(np.min(THRESHOLD_GRID))))
    except (TypeError, ValueError):
        threshold_grid_min = float(np.min(THRESHOLD_GRID))
    try:
        threshold_grid_max = float(model_info.get("threshold_grid_max", float(np.max(THRESHOLD_GRID))))
    except (TypeError, ValueError):
        threshold_grid_max = float(np.max(THRESHOLD_GRID))
    model_info = {
        "algorithm": model_info.get("algorithm", type(model).__name__),
        "hyperparameters": model_info.get(
            "hyperparameters",
            raw_model_hyperparameters,
        ),
        "recall_objective": recall_objective,
        "threshold_grid_min": threshold_grid_min,
        "threshold_grid_max": threshold_grid_max,
    }

    popover_info = bundle.get("popover_info")
    if not isinstance(popover_info, dict):
        popover_info = {}

    return {
        "pipeline": pipeline,
        "threshold": threshold,
        "risk_bands": risk_bands,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "default_inputs": default_inputs,
        "category_options": normalized_options,
        "numeric_limits": numeric_limits,
        "feature_importance": feature_importance,
        "target_rule": target_rule,
        "model_info": model_info,
        "training_info": _normalize_training_info(bundle),
        "popover_info": popover_info,
        "model_source_path": str(model_bundle_path),
        "model_name": str(bundle.get("model_name", model_info.get("algorithm", "Modelo"))),
    }

def _risk_level(probability: float) -> str:
    value = probability * 100
    if value <= 30:
        return "Baixa"
    if value <= 50:
        return "Média"
    return "Alta"


def _predict_probability(bundle: dict, values: dict[str, float | str]) -> float:
    input_row = pd.DataFrame([{feature: values.get(feature) for feature in bundle["feature_columns"]}])
    raw_probability = float(bundle["pipeline"].predict_proba(input_row)[:, 1][0])

    note_cols = ("nota_matematica", "nota_portugues", "nota_ingles")
    if not all(_has_feature(bundle, col) for col in note_cols):
        return raw_probability

    notes = np.array([float(values.get(col, 0.0)) for col in note_cols], dtype=float)
    if np.isnan(notes).any():
        return raw_probability

    mean_note = float(np.mean(notes))
    offsets = notes - mean_note
    lower_bound = float(max(-offsets))
    upper_bound = float(min(10.0 - offsets))
    if upper_bound <= lower_bound:
        return raw_probability

    note_grid = np.linspace(lower_bound, upper_bound, 31)
    probability_grid: list[float] = []
    for mean_candidate in note_grid:
        scenario = dict(values)
        for idx, col in enumerate(note_cols):
            scenario[col] = float(np.clip(mean_candidate + offsets[idx], 0.0, 10.0))
        _sync_derived_features(bundle, scenario)
        scenario_row = pd.DataFrame([{feature: scenario.get(feature) for feature in bundle["feature_columns"]}])
        probability_grid.append(float(bundle["pipeline"].predict_proba(scenario_row)[:, 1][0]))

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(note_grid, probability_grid)
    adjusted_probability = float(iso.predict([mean_note])[0])
    return float(np.clip(adjusted_probability, 0.0, 1.0))


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
    required_cols = {"feature_original", "importancia"}
    if importance_df.empty or not required_cols.issubset(set(importance_df.columns)):
        return

    importance_df["variavel"] = importance_df["feature_original"].map(FEATURE_LABELS).fillna(importance_df["feature_original"])
    importance_df["importancia_pct"] = importance_df["importancia"] * 100
    model_label = str(bundle.get("model_name", "")).strip()
    if not model_label:
        model_label = str(bundle.get("model_info", {}).get("algorithm", "Modelo"))
    st.markdown(f"**Variáveis mais influentes no modelo ({model_label})**")
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
    model_info = bundle.get("model_info", {})
    model_hyperparameters = model_info.get("hyperparameters", {})
    popover_info = bundle.get("popover_info", {})
    train_label = str(popover_info.get("train_label", "Treino (2022)"))
    test_label = str(popover_info.get("test_label", "Teste temporal (2023)"))
    threshold_note = popover_info.get("threshold_selection_criterion")
    model_selection_note = popover_info.get("model_selection_criterion")

    with st.popover("ℹ️", help="Detalhes técnicos do modelo"):
        def _format_key_label(raw_key: str) -> str:
            label = str(raw_key).replace("_", " ").strip().title()
            replacements = {
                "Ian": "IAN",
                "Ida": "IDA",
                "Ieg": "IEG",
                "Iaa": "IAA",
                "Ips": "IPS",
                "Ipp": "IPP",
                "Ipv": "IPV",
                "Pr": "PR",
                "Roc": "ROC",
            }
            for old, new in replacements.items():
                label = label.replace(old, new)
            return label

        def _format_value(value: object) -> str:
            if value is None:
                return "-"
            if isinstance(value, bool):
                return "Sim" if value else "Não"
            if isinstance(value, (int, np.integer)):
                return f"{int(value)}"
            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    return "-"
                if float(value).is_integer():
                    return f"{int(value)}"
                return f"{float(value):.4g}"
            return str(value)

        st.markdown("**Detalhes técnicos do modelo**")
        st.caption("Seção de apoio para leitura analítica. O uso operacional está no formulário principal.")

        st.markdown("**Regra de definição do risco**")
        target_rule = bundle.get("target_rule")
        if isinstance(target_rule, dict):
            description = target_rule.get("description")
            if description:
                st.write(str(description))

            threshold_items = []
            for key, value in target_rule.items():
                if str(key).lower() == "description":
                    continue
                if "threshold" in str(key).lower():
                    threshold_items.append((str(key), value))

            if threshold_items:
                cols = st.columns(min(3, len(threshold_items)))
                for idx, (key, value) in enumerate(threshold_items):
                    with cols[idx % len(cols)]:
                        st.metric(_format_key_label(key), _format_value(value))

            for key, value in target_rule.items():
                if str(key).lower() == "description" or (isinstance(key, str) and "threshold" in key.lower()):
                    continue
                st.caption(f"{_format_key_label(str(key))}: {_format_value(value)}")
        else:
            st.write(str(target_rule))

        st.markdown("**Amostras usadas no treinamento**")
        st.write(f"{train_label}: {info['train_rows']:,}".replace(",", "."))
        st.write(f"{test_label}: {info['test_rows']:,}".replace(",", "."))

        st.markdown("**Parâmetros do modelo**")
        model_algorithm = str(model_info.get("algorithm", bundle.get("model_name", "Modelo")))
        c_model, c_threshold = st.columns(2)
        with c_model:
            st.metric("Algoritmo", model_algorithm)
        with c_threshold:
            st.metric("Threshold final", f"{bundle['threshold']:.3f}")

        if model_hyperparameters:
            preferred_order = [
                "n_estimators",
                "max_depth",
                "learning_rate",
                "min_samples_split",
                "min_samples_leaf",
                "subsample",
                "colsample_bytree",
                "max_features",
                "class_weight",
                "random_state",
            ]
            shown_keys: list[str] = [key for key in preferred_order if key in model_hyperparameters]
            if not shown_keys:
                shown_keys = list(model_hyperparameters.keys())[:8]

            c_left, c_right = st.columns(2)
            for idx, key in enumerate(shown_keys):
                col = c_left if idx % 2 == 0 else c_right
                with col:
                    st.metric(_format_key_label(key), _format_value(model_hyperparameters.get(key)))

        if threshold_note:
            st.caption(str(threshold_note))
        elif model_info:
            st.caption(
                f"Threshold escolhido via OOF com objetivo de recall >= {model_info.get('recall_objective', RECALL_OBJECTIVE):.2f}, "
                f"avaliando grade de {model_info.get('threshold_grid_min', float(np.min(THRESHOLD_GRID))):.2f} "
                f"a {model_info.get('threshold_grid_max', float(np.max(THRESHOLD_GRID))):.2f}."
            )
        if model_selection_note:
            st.caption(str(model_selection_note))

        st.markdown("**Desempenho no teste temporal (2023)**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
            st.caption("Sensibilidade: proporção dos casos de risco que o modelo consegue identificar.")
        with c2:
            st.metric("AUC-ROC", f"{metrics['roc_auc']:.3f}")
            st.caption("Capacidade de separar risco vs. não risco em todos os limiares (quanto maior, melhor).")
        c3, c4 = st.columns(2)
        with c3:
            st.metric("Precisão", f"{metrics['precision'] * 100:.1f}%")
            st.caption("Entre os casos marcados como risco, quantos realmente eram risco.")
        with c4:
            st.metric("F1-score", f"{metrics['f1'] * 100:.1f}%")
            st.caption("Média harmônica entre Precisão e Recall: equilíbrio entre detectar e errar menos.")

        _render_feature_importance(bundle)


def render_modelo_preditivo_tab(df: pd.DataFrame) -> None:
    try:
        model_bundle_path = _resolve_model_bundle_path()
        bundle = load_model_bundle(
            df=df,
            model_bundle_path=model_bundle_path,
            model_bundle_mtime_ns=model_bundle_path.stat().st_mtime_ns,
        )
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

    with st.form("predicao_modelo_form", clear_on_submit=False):
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

        submitted = st.form_submit_button(
            "Calcular indicadores e probabilidade de risco de defasagem",
            type="primary",
            width="stretch",
        )

        if not submitted:
            return

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

    _sync_derived_features(bundle, values)
    probability = _predict_probability(bundle, values)
    level = _risk_level(probability)
    dimension_message = _dimension_priority_message(values)

    _render_probability_gauge(probability)
    _render_result_card(probability, level, dimension_message)


