import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Desabilitar GUI para execução no Docker
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuração de caminhos
ML_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ML_DIR / "outputs" / "data" / "dataset_ml.csv"
FIGURES_DIR = ML_DIR / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("ETAPA 2: Análise Exploratória de Dados (EDA)")
    print("=" * 60)

    # 1. Carregar dados
    if not INPUT_PATH.exists():
        print(f"Erro: Arquivo {INPUT_PATH} não encontrado. Execute a Fase 1 primeiro.")
        return
        
    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset carregado com sucesso. Formato: {df.shape}")

    # 2. Estatísticas Descritivas Gerais
    print("\n--- Estatísticas Descritivas ---")
    print(df[["area_plantada_ha", "produtividade_kg_ha", "precipitacao_anual_mm", "temp_media_anual", "risco_zarc_medio"]].describe())

    print("\n--- Valores Ausentes ---")
    print(df.isnull().sum())

    print("\n--- Distribuição por Cultura ---")
    print(df["cultura"].value_counts())

    # 3. Gerar Gráficos
    sns.set_theme(style="whitegrid")
    
    # Gráfico 1: Distribuição da Produtividade por Cultura (Histograma)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="produtividade_kg_ha", hue="cultura", kde=True, bins=50, palette="viridis", multiple="stack")
    plt.title("Distribuição da Produtividade Agrícola por Cultura (2021)")
    plt.xlabel("Produtividade (kg/ha)")
    plt.ylabel("Frequência (Municípios)")
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "distribuicao_produtividade.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"✓ Gráfico salvo: {fig1_path}")

    # Gráfico 2: Boxplot de Produtividade por Região e Cultura
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x="regiao", y="produtividade_kg_ha", hue="cultura", palette="Set2")
    plt.title("Produtividade Agrícola por Região e Cultura")
    plt.xlabel("Região")
    plt.ylabel("Produtividade (kg/ha)")
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "boxplot_produtividade_regiao.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"✓ Gráfico salvo: {fig2_path}")

    # Gráfico 3: Matriz de Correlação das Variáveis Numéricas
    plt.figure(figsize=(12, 10))
    cols_corr = [
        "area_plantada_ha", "area_colhida_ha", "qtde_produzida_ton", 
        "precipitacao_anual_mm", "temp_max_media", "temp_min_media", 
        "temp_media_anual", "umidade_media_anual", "risco_zarc_medio", 
        "amplitude_termica_media", "produtividade_kg_ha"
    ]
    corr_matrix = df[cols_corr].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True)
    plt.title("Matriz de Correlação Linear (Pearson)")
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "heatmap_correlacao.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"✓ Gráfico salvo: {fig3_path}")

    # Gráfico 4: Relação Temperatura Média vs Produtividade
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="temp_media_anual", y="produtividade_kg_ha", hue="cultura", alpha=0.5, palette="cool")
    plt.title("Temperatura Média Anual vs Produtividade")
    plt.xlabel("Temperatura Média Anual (°C)")
    plt.ylabel("Produtividade (kg/ha)")
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "scatter_temp_produtividade.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"✓ Gráfico salvo: {fig4_path}")

    print("EDA finalizada com sucesso.")

if __name__ == "__main__":
    main()
