# Datathon Fase 5 | Associacao Passos Magicos

Aplicacao Streamlit e notebooks analiticos para transformar a base PEDE (2022 a 2024) em diagnostico gerencial e previsao de risco de defasagem.

## Objetivo

Apoiar decisoes preventivas da ONG com:

- leitura estruturada da trajetoria dos estudantes;
- identificacao de sinais de risco;
- simulacao individual de probabilidade de defasagem no ciclo seguinte.

## Entregas do repositorio

- `scripts/1_EDA_e_Storytelling.ipynb`: consolidacao de dados + storytelling analitico.
- `scripts/2_Modelo_Preditivo.ipynb`: modelagem, comparacao de modelos, threshold e artefatos.
- `app.py`: aplicacao Streamlit com quatro abas de consumo.
- `tab_analise_exploratoria.py`: interface de analise (Q1 a Q11).
- `tab_modelo_preditivo.py`: inferencia preditiva com formulario operacional.
- `tab_dicionario.py`: visualizacao e download do PDF de referencia.

## Estrutura de pastas

```text
.
|-- app.py
|-- tab_analise_exploratoria.py
|-- tab_modelo_preditivo.py
|-- tab_dicionario.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- BASE DE DADOS PEDE 2024 - DATATHON.xlsx
|   `-- processed/
|       |-- pede_consolidado_2022_2024.csv
|       `-- pede_longitudinal_2022_2024.csv
|-- scripts/
|   |-- 1_EDA_e_Storytelling.ipynb
|   |-- 2_Modelo_Preditivo.ipynb
|   `-- _eda_cells_dump.txt
|-- artifacts/
|   |-- feature_importance.csv
|   |-- feature_importance_all_models.csv
|   |-- model_accuracy_analysis.csv
|   |-- model_comparison_test.csv
|   |-- predicoes_risco_2024.csv
|   |-- scaler_numerico.pkl
|   `-- threshold_curve_best_model.csv
|-- models/
|   |-- features_list.pkl
|   `-- modelo_risco_defasagem.pkl
|-- doc/
|   |-- Dicionario Dados Datathon.pdf
|   |-- PEDE_ Pontos importantes.pdf
|   `-- outros PDFs e ativos visuais
`-- tests/
    |-- resumo.md
    `-- resumo_detalhado.md
```

## Base utilizada

Fonte principal do app: `data/processed/pede_consolidado_2022_2024.csv`

- 3.027 registros
- 1.660 estudantes unicos
- anos: 2022, 2023, 2024
- distribuicao:
  - 2022: 859
  - 2023: 1.013
  - 2024: 1.155

## Aplicacao Streamlit

A aplicacao abre quatro abas:

- `Inicio`: contexto, objetivo, estrutura e visao executiva.
- `Analise exploratoria`: narrativa analitica em blocos (Q1 a Q11).
- `Modelo Preditivo`: simulacao operacional de risco por estudante.
- `Dicionario`: renderizacao e download do documento PEDE.

## Modelo operacional atual

A aba preditiva carrega o bundle em `models/modelo_risco_defasagem.pkl`.

Estado atual do modelo salvo:

- modelo: XGBoost (`XGBClassifier`)
- threshold: 0.25
- regra de target: `risco = 1 se IAN_t+1 <= 5.0`
- treino: 574 linhas (ano base 2022)
- teste temporal: 693 linhas (ano base 2023)
- metricas de teste:
  - accuracy: 0.6869
  - precision: 0.5987
  - recall: 0.8961
  - F1: 0.7178
  - ROC-AUC: 0.8450
  - PR-AUC: 0.8262
  - Brier: 0.1620

Observacao de produto:

- a classificacao visual no card usa faixas: Baixa (<=30%), Media (30%-50%), Alta (>50%).
- o popover tecnico foi removido da interface para simplificar o fluxo.

## Artefatos

Arquivos de apoio em `artifacts/`:

- comparacao de modelos (`model_comparison_test.csv`)
- analise de acuracia e CV (`model_accuracy_analysis.csv`)
- curva de threshold (`threshold_curve_best_model.csv`)
- importancias (`feature_importance.csv` e `feature_importance_all_models.csv`)
- scoring 2024 (`predicoes_risco_2024.csv`)

Distribuicao de risco em 2024 (`predicoes_risco_2024.csv`):

- Alto: 736
- Moderado: 79
- Baixo: 340

## Requisitos

Dependencias em `requirements.txt`:

- pandas
- numpy
- openpyxl
- plotly
- scikit-learn
- xgboost
- shap
- streamlit[pdf]
- pymupdf
- matplotlib
- imbalanced-learn
- statsmodels

## Como executar localmente

1. Criar ambiente virtual:

```powershell
python -m venv .venv
```

2. Ativar ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

4. Rodar o app:

```powershell
streamlit run app.py
```

## Equipe

- Alisson Cordeiro Nobrega
- Lucas Benevides Miranda
- Marcos Vinicius Fernandes de Freitas
- Rodrigo Mallet e Ribeiro de Carvalho
