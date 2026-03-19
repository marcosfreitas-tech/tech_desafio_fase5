# Datathon Fase 5 | Associação Passos Mágicos

Aplicação em Streamlit e notebooks analíticos desenvolvidos para o Datathon da Pós-Tech, com foco em transformar a base PEDE de 2022 a 2024 em leitura gerencial e em uma solução preditiva de risco de defasagem escolar.

O projeto combina duas frentes complementares:

- análise exploratória com storytelling orientado a negócio;
- modelagem preditiva para apoiar priorização preventiva de acompanhamento.

## Objetivo

A proposta do projeto é apoiar a Associação Passos Mágicos na identificação de sinais de risco educacional, entendimento dos principais drivers de desempenho e simulação da probabilidade de entrada em defasagem no próximo ciclo.

## O que a aplicação entrega

- uma aba inicial com o contexto do desafio, visão executiva do projeto e composição da equipe;
- uma aba de análise exploratória com 10 perguntas de negócio organizadas em blocos analíticos;
- uma aba de modelo preditivo com formulário operacional para simulação individual de risco;
- cálculo automático dos indicadores do PEDE a partir das entradas informadas no formulário;
- leitura técnica do modelo com métricas, regra de target e variáveis mais influentes.

## Base utilizada

- 3.027 registros consolidados;
- 1.660 estudantes únicos;
- janela temporal de 2022, 2023 e 2024;
- fonte principal da aplicação: `data/processed/pede_consolidado_2022_2024.csv`.

Distribuição dos registros por ano:

- 2022: 859
- 2023: 1.013
- 2024: 1.155

## Visão funcional

### 1. Análise exploratória

A aba de EDA organiza o storytelling em perguntas de negócio sobre:

- correlações entre indicadores e INDE;
- evolução de IAN e IDA ao longo do tempo;
- relação entre engajamento, desempenho e ponto de virada;
- coerência entre autoavaliação e indicadores objetivos;
- sinais antecedentes de risco via IPS e IPP;
- drivers do IPV;
- combinações que mais explicam o INDE;
- baseline exploratório de previsão de risco;
- efetividade por fase e coorte.

### 2. Modelo preditivo

A aba preditiva treina um `RandomForestClassifier` em tempo de execução a partir da base consolidada e trabalha com uma lógica longitudinal para estimar risco no ciclo seguinte.

Definição da variável-alvo:

- risco = 1 se `IAN t+1 <= 5,0`, ou `IDA t+1 <= 6,0`, ou queda de `IAN <= -1,0`.

Configuração principal observada no app:

- treino temporal com base de 2022;
- teste temporal com base de 2023;
- 574 registros no treino;
- 693 registros no teste;
- threshold selecionado: `0.25`.

Métricas do teste temporal:

- acurácia: `74,6%`
- recall: `98,9%`
- precisão: `72,7%`
- F1-score: `83,8%`
- AUC-ROC: `0,819`
- PR-AUC: `0,901`

Variáveis com maior peso no modelo atual:

- `inde_22`
- `inde_ano`
- `defasagem`
- `ian`
- `media_notas`

Além da probabilidade final, a interface calcula automaticamente indicadores como `IAN`, `IDA`, `IEG`, `IAA`, `IPS`, `IPP` e `IPV` com base nas entradas informadas pelo usuário.

## Estrutura do projeto

A estrutura abaixo considera apenas os arquivos e pastas relevantes do repositório versionado:

```text
.
|-- app.py
|-- tab_analise_exploratoria.py
|-- tab_modelo_preditivo.py
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
|   |-- modelo_risco_defasagem.pkl
|   |-- overfitting_diagnostico.csv
|   |-- overfitting_diagnostico_pos_mitigacao.csv
|   |-- predicoes_risco_2024.csv
|   |-- scaler_numerico.pkl
|   `-- threshold_curve_best_model.csv
|-- models/
|   `-- features_list.pkl
`-- doc/
    |-- Datathon e PDFs de apoio
    `-- identidade visual da Passos Mágicos
```

## Arquivos principais

- `app.py`: ponto de entrada da aplicação Streamlit.
- `tab_analise_exploratoria.py`: renderização das análises e do storytelling.
- `tab_modelo_preditivo.py`: treinamento do modelo, interface de entrada e cálculo da probabilidade de risco.
- `scripts/1_EDA_e_Storytelling.ipynb`: notebook da análise exploratória e geração da base processada.
- `scripts/2_Modelo_Preditivo.ipynb`: notebook de modelagem, comparação de modelos e geração de artefatos.

## Dados e artefatos

### Dados

- `data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`: base bruta disponibilizada no desafio.
- `data/processed/pede_consolidado_2022_2024.csv`: base consolidada usada diretamente pela aplicação.
- `data/processed/pede_longitudinal_2022_2024.csv`: base longitudinal derivada para análises temporais e modelagem.

### Artefatos de modelagem

Os arquivos em `artifacts/` registram saídas produzidas no notebook preditivo, como:

- modelo serializado;
- scaler numérico;
- comparativos entre modelos;
- análise de threshold;
- importâncias das variáveis;
- diagnósticos de overfitting;
- score de risco para 2024.

Observação: a aplicação Streamlit atual recompõe e treina o bundle do modelo em memória a partir do CSV processado. Os artefatos salvos no repositório funcionam como apoio analítico e registro dos experimentos.

## Como executar localmente

### 1. Criar e ativar o ambiente virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar as dependências

```powershell
pip install -r requirements.txt
```

### 3. Iniciar a aplicação

```powershell
streamlit run app.py
```

## Como usar

### Navegação principal

- `Início`: visão executiva do desafio, objetivo do projeto e escopo da solução.
- `Análise exploratória`: leitura gerencial dos indicadores e dos padrões encontrados na base.
- `Modelo Preditivo`: simulação manual de risco de defasagem para um estudante.

### Fluxo da aba preditiva

1. Informar dados do aluno.
2. Preencher notas acadêmicas.
3. Informar as bases de cálculo dos indicadores do PEDE.
4. Revisar os indicadores calculados automaticamente.
5. Acionar o botão para obter a probabilidade de risco e a recomendação de prioridade.

## Dependências principais

- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `scikit-learn`
- `imbalanced-learn`
- `shap`
- `matplotlib`
- `openpyxl`
- `statsmodels`

## Observações importantes

- O app depende da presença do arquivo `data/processed/pede_consolidado_2022_2024.csv`.
- Os notebooks possuem lógica para reconstruir a base processada a partir do Excel bruto, quando necessário.
- Não há dependência explícita de variáveis de ambiente ou secrets para o fluxo principal documentado neste repositório.

## Equipe

- Alisson Cordeiro Nóbrega
- Lucas Benevides Miranda
- Marcos Vinícius Fernandes de Freitas
- Rodrigo Mallet e Ribeiro de Carvalho
