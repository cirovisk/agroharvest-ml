import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Desabilitar interface gráfica para o Docker
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# Configuração de caminhos
ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "outputs" / "data"
MODELS_DIR = ML_DIR / "outputs" / "models"
RESULTS_DIR = ML_DIR / "outputs" / "results"
FIGURES_DIR = ML_DIR / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_pkl(filename):
    with open(DATA_DIR / filename, "rb") as f:
        return pickle.load(f)

def load_model(filename):
    with open(MODELS_DIR / filename, "rb") as f:
        return pickle.load(f)

def main():
    print("=" * 60)
    print("ETAPA 5: Avaliação, Métricas e Geração de Visualizações")
    print("=" * 60)

    # 1. Carregar dados de teste
    X_test_reg = load_pkl("X_test.pkl").copy()
    X_test_reg_scaled = load_pkl("X_test_scaled.pkl").copy()
    y_test_reg = load_pkl("y_test_reg.pkl").copy()

    X_test_clf = load_pkl("X_test_clf.pkl").copy()
    X_test_clf_scaled = load_pkl("X_test_clf_scaled.pkl").copy()
    y_test_clf = load_pkl("y_test_clf.pkl").copy()

    label_encoder = load_pkl("label_encoder.pkl")
    class_names = label_encoder.classes_

    # 2. Carregar modelos de regressão
    mlp_reg = load_model("mlp_regressor.pkl")
    rf_reg = load_model("random_forest_regressor.pkl")
    xgb_reg = load_model("xgboost_regressor.pkl")

    # 3. Carregar modelos de classificação
    nb_clf = load_model("naive_bayes.pkl")
    knn_clf = load_model("k-nn.pkl")
    mlp_clf = load_model("mlp_classifier.pkl")
    bag_clf = load_model("bagging_mlp.pkl")

    sns.set_theme(style="whitegrid")

    # --- VISUALIZAÇÕES DE REGRESSÃO ---

    # Gráfico 1: Real vs Predito do Melhor Regressor (Random Forest)
    print("Gerando gráfico Real vs Predito...")
    y_pred_rf = rf_reg.predict(X_test_reg)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_reg, y_pred_rf, alpha=0.4, color="#2c7bb6")
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], "k--", lw=2)
    plt.title("Produtividade Real vs Predita (Random Forest Regressor)")
    plt.xlabel("Real (kg/ha)")
    plt.ylabel("Predito (kg/ha)")
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "regression_real_vs_predito.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"✓ Salvo: {fig1_path}")

    # Gráfico 2: Importância de Features (Random Forest)
    print("Gerando gráfico de importância das features...")
    importances = rf_reg.feature_importances_
    features = X_test_reg.columns
    df_importances = pd.DataFrame({"Feature": features, "Importância": importances}).sort_values(by="Importância", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_importances, x="Importância", y="Feature", palette="viridis")
    plt.title("Importância das Variáveis (Random Forest Regressor)")
    plt.xlabel("Importância Relativa")
    plt.ylabel("Variável")
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "regression_importance_features.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"✓ Salvo: {fig2_path}")

    # --- VISUALIZAÇÕES DE CLASSIFICAÇÃO ---

    # Gráfico 3: Matriz de Confusão do Melhor Classificador (k-NN)
    print("Gerando matriz de confusão do k-NN...")
    y_pred_knn = knn_clf.predict(X_test_clf_scaled)
    cm_knn = confusion_matrix(y_test_clf, y_pred_knn)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusão - k-NN")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "confusion_matrix_knn.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"✓ Salvo: {fig3_path}")

    # Gráfico 4: Matriz de Confusão do Bagging MLP
    print("Gerando matriz de confusão do Bagging MLP...")
    y_pred_bag = bag_clf.predict(X_test_clf_scaled)
    cm_bag = confusion_matrix(y_test_clf, y_pred_bag)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_bag, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusão - Bagging MLP (Ensemble)")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "confusion_matrix_bagging.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"✓ Salvo: {fig4_path}")

    # 4. Exibir relatórios detalhados no terminal
    print("\n" + "=" * 60)
    print("Relatório de Classificação k-NN:")
    print("=" * 60)
    print(classification_report(y_test_clf, y_pred_knn, target_names=class_names))

    print("\n" + "=" * 60)
    print("Relatório de Classificação Bagging MLP:")
    print("=" * 60)
    print(classification_report(y_test_clf, y_pred_bag, target_names=class_names))

if __name__ == "__main__":
    main()
