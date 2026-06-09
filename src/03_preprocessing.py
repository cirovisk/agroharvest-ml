import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Configuração de caminhos
ML_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ML_DIR / "outputs" / "data" / "dataset_ml.csv"
OUTPUT_DIR = ML_DIR / "outputs" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_and_split():
    print("=" * 60)
    print("ETAPA 3: Pré-processamento e Divisão dos Dados (KDD)")
    print("=" * 60)

    # 1. Carregar dados
    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset carregado: {df.shape}")

    # Remover variáveis irrelevantes ou 100% nulas
    drop_cols = ["id_municipio", "id_cultura", "ano", "codigo_ibge", "nome", "uf", 
                 "id_estacao_proxima", "distancia_estacao_km", "umidade_media_anual",
                 "valor_producao_mil_reais", "qtde_produzida_ton", "area_colhida_ha"]
    
    df_clean = df.drop(columns=drop_cols)
    print(f"Colunas mantidas para ML: {list(df_clean.columns)}")

    # 2. Criar classes de desempenho produtivo (discretização do target para classificação)
    # Faremos isso calculando os percentis 33.3 e 66.6 dentro de cada cultura para garantir balanceamento estrito
    df_clean["classe_desempenho"] = ""
    
    for cult in df_clean["cultura"].unique():
        mask = df_clean["cultura"] == cult
        prod = df_clean.loc[mask, "produtividade_kg_ha"]
        p33 = np.percentile(prod, 33.3)
        p66 = np.percentile(prod, 66.6)
        
        # Mapeamento:
        # Alta produtividade -> Alto Desempenho Produtivo
        # Média produtividade -> Médio Desempenho Produtivo
        # Baixa produtividade -> Baixo Desempenho Produtivo
        df_clean.loc[mask & (prod <= p33), "classe_desempenho"] = "Baixo"
        df_clean.loc[mask & (prod > p33) & (prod <= p66), "classe_desempenho"] = "Médio"
        df_clean.loc[mask & (prod > p66), "classe_desempenho"] = "Alto"
        
        print(f"Cultura '{cult}' - Limites de Produtividade: Baixo Desempenho <= {p33:.1f} kg/ha | Médio | Alto Desempenho > {p66:.1f} kg/ha")

    print("\nDistribuição das classes de desempenho produtivo geradas:")
    print(df_clean["classe_desempenho"].value_counts())

    # 3. Separar Features (X) e Targets (y)
    y_reg = df_clean["produtividade_kg_ha"]
    
    le = LabelEncoder()
    y_clf = le.fit_transform(df_clean["classe_desempenho"]) # Encode para 0, 1, 2
    
    # Salvar classes do LabelEncoder para podermos decodificar depois
    with open(OUTPUT_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    X_raw = df_clean.drop(columns=["produtividade_kg_ha", "classe_desempenho"])

    # 4. Splits de Treino/Teste (80% treino, 20% teste)
    print("\nRealizando splits Treino/Teste...")
    
    # Split para Regressão (estratificado por cultura)
    X_train_raw, X_test_raw, y_train_reg, y_test_reg = train_test_split(
        X_raw, y_reg, test_size=0.2, random_state=42, stratify=X_raw["cultura"]
    )
    
    # Split para Classificação (estratificado pela classe de desempenho + cultura)
    strat_col = X_raw["cultura"] + "_" + df_clean["classe_desempenho"]
    X_train_clf_raw, X_test_clf_raw, y_train_clf, y_test_clf = train_test_split(
        X_raw, y_clf, test_size=0.2, random_state=42, stratify=strat_col
    )

    cat_cols = ["cultura", "regiao"]
    num_cols = [c for c in X_raw.columns if c not in cat_cols]
    
    print(f"\nFeatures Categóricas: {cat_cols}")
    print(f"Features Numéricas: {num_cols}")

    # --- PROCESSAMENTO PARA REGRESSÃO ---
    ohe_reg = OneHotEncoder(sparse_output=False, drop="first")
    scaler_reg = StandardScaler()
    
    # Fit e transform nos dados de regressão
    X_train_cat_enc = ohe_reg.fit_transform(X_train_raw[cat_cols])
    X_test_cat_enc = ohe_reg.transform(X_test_raw[cat_cols])
    cat_feature_names = ohe_reg.get_feature_names_out(cat_cols)
    
    df_train_cat_enc = pd.DataFrame(X_train_cat_enc, columns=cat_feature_names, index=X_train_raw.index)
    df_test_cat_enc = pd.DataFrame(X_test_cat_enc, columns=cat_feature_names, index=X_test_raw.index)
    
    X_train = pd.concat([X_train_raw[num_cols], df_train_cat_enc], axis=1)
    X_test = pd.concat([X_test_raw[num_cols], df_test_cat_enc], axis=1)
    
    # Scaling
    X_train_num_scaled = scaler_reg.fit_transform(X_train_raw[num_cols])
    X_test_num_scaled = scaler_reg.transform(X_test_raw[num_cols])
    
    df_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=num_cols, index=X_train_raw.index)
    df_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=num_cols, index=X_test_raw.index)
    
    X_train_scaled = pd.concat([df_train_num_scaled, df_train_cat_enc], axis=1)
    X_test_scaled = pd.concat([df_test_num_scaled, df_test_cat_enc], axis=1)
    
    # Salvar o OneHotEncoder e o Scaler de Regressão
    with open(OUTPUT_DIR / "one_hot_encoder.pkl", "wb") as f:
        pickle.dump(ohe_reg, f)
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler_reg, f)

    # --- PROCESSAMENTO PARA CLASSIFICAÇÃO ---
    ohe_clf = OneHotEncoder(sparse_output=False, drop="first")
    scaler_clf = StandardScaler()
    
    # Fit e transform nos dados de classificação
    X_train_clf_cat_enc = ohe_clf.fit_transform(X_train_clf_raw[cat_cols])
    X_test_clf_cat_enc = ohe_clf.transform(X_test_clf_raw[cat_cols])
    cat_feature_names_clf = ohe_clf.get_feature_names_out(cat_cols)
    
    df_train_clf_cat_enc = pd.DataFrame(X_train_clf_cat_enc, columns=cat_feature_names_clf, index=X_train_clf_raw.index)
    df_test_clf_cat_enc = pd.DataFrame(X_test_clf_cat_enc, columns=cat_feature_names_clf, index=X_test_clf_raw.index)
    
    X_train_clf = pd.concat([X_train_clf_raw[num_cols], df_train_clf_cat_enc], axis=1)
    X_test_clf = pd.concat([X_test_clf_raw[num_cols], df_test_clf_cat_enc], axis=1)
    
    # Scaling
    X_train_clf_num_scaled = scaler_clf.fit_transform(X_train_clf_raw[num_cols])
    X_test_clf_num_scaled = scaler_clf.transform(X_test_clf_raw[num_cols])
    
    df_train_clf_num_scaled = pd.DataFrame(X_train_clf_num_scaled, columns=num_cols, index=X_train_clf_raw.index)
    df_test_clf_num_scaled = pd.DataFrame(X_test_clf_num_scaled, columns=num_cols, index=X_test_clf_raw.index)
    
    X_train_clf_scaled = pd.concat([df_train_clf_num_scaled, df_train_clf_cat_enc], axis=1)
    X_test_clf_scaled = pd.concat([df_test_clf_num_scaled, df_test_clf_cat_enc], axis=1)

    # 7. Salvar os datasets splitados
    datasets = {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "X_train_clf": X_train_clf,
        "X_test_clf": X_test_clf,
        "X_train_clf_scaled": X_train_clf_scaled,
        "X_test_clf_scaled": X_test_clf_scaled,
        "y_train_clf": y_train_clf,
        "y_test_clf": y_test_clf
    }
    
    for name, data in datasets.items():
        with open(OUTPUT_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(data, f)
            
    print(f"✓ Todos os datasets splitados e transformados foram salvos em: {OUTPUT_DIR}")
    print(f"Tamanho do treino de Regressão: {X_train.shape}")
    print(f"Tamanho do teste de Regressão: {X_test.shape}")
    print(f"Tamanho do treino de Classificação: {X_train_clf.shape}")

if __name__ == "__main__":
    preprocess_and_split()
