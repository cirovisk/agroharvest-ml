# Predição de Produtividade Agrícola e Classificação de Desempenho em Municípios Brasileiros

Este repositório contém o desenvolvimento do projeto final apresentado à disciplina CET0621 - Aprendizado de Máquina na Análise de Dados da UNICAMP (Faculdade de Tecnologia).

O objetivo do trabalho é aplicar técnicas de aprendizado de máquina supervisionado para estimar a produtividade municipal em kg/ha (regressão) e classificar faixas de desempenho produtivo (classificação multiclasse) para as culturas de soja e milho em todo o território nacional.

---

## Arquitetura e Integração (Metodologia KDD)

Os dados utilizados foram extraídos e unificados a partir do Data Lakehouse portátil Agroharvest-BRv2, baseado em DuckDB e Apache Parquet. A solução consolida quatro fontes públicas distintas:
1. **Produção Agrícola Municipal (PAM/IBGE)**: Dados de produtividade real de 2021.
2. **Zoneamento Agrícola de Risco Climático (ZARC/MAPA)**: Zoneamento de risco de plantio.
3. **Acompanhamento de Safra (CONAB)**: Histórico de safras e preços.
4. **Meteorologia Histórica (API Open-Meteo)**: Dados diários de temperatura e precipitação.

### Integração Geoespacial por Distância de Haversine
Como os dados diários de meteorologia estavam associados a 29 estações meteorológicas físicas de referência regional, implementou-se o cálculo de distância de Haversine para associar cada um dos 5.570 municípios do Brasil à sua estação climatológica mais próxima de forma realista (considerando a curvatura terrestre):

$$d = 2r \arcsin \left( \sqrt{\sin^2\frac{\Delta \phi}{2} + \cos\phi_1 \cos\phi_2 \sin^2\frac{\Delta \lambda}{2}} \right)$$

O dataset final analítico consolidou 7.549 instâncias para a modelagem preditiva, integradas eficientemente por meio do DuckDB e arquivos Parquet.

---

## Resultados Obtidos

Os modelos passaram por validação cruzada 5-fold e sintonia fina de hiperparâmetros via busca em grade exaustiva (GridSearchCV), implementando rígidos controles para prevenção de vazamento de dados (data leakage).

### 1. Estimação de Produtividade (Regressão)
Métricas obtidas na partição de teste para predição de produtividade em kg/ha:

| Modelo | MAE (kg/ha) | RMSE (kg/ha) | R² |
| :--- | :---: | :---: | :---: |
| MLP Regressor (Neural Net) | 812,91 | 1223,53 | 0,6541 |
| **Random Forest Regressor** | **669,23** | **1024,79** | **0,7573** |
| XGBoost Regressor | 693,74 | 1050,66 | 0,7449 |

O estimador Random Forest obteve o melhor desempenho geral (R² = 0,7573), indicando forte habilidade em mapear relações não-lineares sem requerer linearidade estrita das variáveis físicas. A localização geoespacial (latitude/longitude) e a área de plantio foram os atributos de maior importância.

### 2. Classificação de Desempenho Produtivo
A produtividade foi discretizada de forma balanceada e estratificada por cultura em faixas de Baixo, Médio e Alto desempenho.

| Modelo | Acurácia | Precisão Macro | Recall Macro | F1-Macro |
| :--- | :---: | :---: | :---: | :---: |
| Naive Bayes | 50,99% | 50,22% | 50,98% | 0,4451 |
| **k-NN (Manhattan, k=7)** | **70,33%** | **70,26%** | **70,33%** | **0,7029** |
| MLP Classifier | 67,35% | 67,41% | 67,35% | 0,6733 |
| Bagging MLP (Ensemble) | 67,62% | 67,07% | 67,61% | 0,6706 |

O classificador k-NN obteve a melhor acurácia geral (70,33%), comprovando a hipótese agronômica de contiguidade de produtividade regional no espaço euclidiano (municípios vizinhos tendem a ter desempenhos correlatos).

---

## Como Executar o Projeto (Docker)

O ambiente de experimentos de Machine Learning está totalmente conteinerizado utilizando Docker e Docker Compose para garantir a portabilidade e a reprodutibilidade dos resultados.

### Pré-requisitos
* Docker e Docker Compose instalados.
* O banco de dados DuckDB (cultivares.duckdb) deve estar gerado e acessível no diretório configurado em src/01_build_dataset.py.

### Instruções
1. Construa a imagem docker:
   ```bash
   docker compose build
   ```
2. Suba o container e execute os experimentos completos (ETL, pré-processamento, treinamento e avaliação):
   ```bash
   docker compose up -d
   ```
3. Acompanhe a execução dos logs preditivos:
   ```bash
   docker compose logs -f
   ```

Os resultados de desempenho e matrizes de confusão gerados serão salvos na pasta /outputs.

---

## Artigo Completo
O relatório científico completo, formatado no padrão IEEE com a fundamentação conceitual das etapas KDD e discussão estatística das limitações do modelo (como o risco de spatial overfitting), está disponível em:
* docs/artigo_cientifico_agroharvest_ml.pdf

## Contexto Acadêmico e Autoria

Este projeto foi desenvolvido originalmente como um trabalho em grupo para a disciplina CET0621 na UNICAMP pelos alunos Ciro Fraga de Souza, Felippe Dada Gaede, Jonnerson Lopes e Murilo Melo de Oliveira para fins de avaliação acadêmica.

Todo o desenvolvimento do código-fonte, arquitetura do banco de dados DuckDB, cruzamento geoespacial via fórmula de Haversine, conteinerização via Docker e modelagem preditiva contidos neste repositório foram realizados integralmente por mim (Ciro Fraga de Souza).
