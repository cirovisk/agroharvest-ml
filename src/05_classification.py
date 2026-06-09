import os
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuração de caminhos
ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "outputs" / "data"
MODELS_DIR = ML_DIR / "outputs" / "models"
RESULTS_DIR = ML_DIR / "outputs" / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_pkl(filename):
    with open(DATA_DIR / filename, "rb") as f:
        return pickle.load(f)

def main():
    print("=" * 60)
    print("ETAPA 4 (Classificação): Modelagem de Desempenho Produtivo")
    print("=" * 60)

    # 1. Carregar dados
    X_train = load_pkl("X_train_clf.pkl").copy()
    X_test = load_pkl("X_test_clf.pkl").copy()
    X_train_scaled = load_pkl("X_train_clf_scaled.pkl").copy()
    X_test_scaled = load_pkl("X_test_clf_scaled.pkl").copy()
    y_train = load_pkl("y_train_clf.pkl").copy()
    y_test = load_pkl("y_test_clf.pkl").copy()

    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

    # 2. Definição dos Experimentos
    experiments = {
        "Naive Bayes": {
            "model": GaussianNB(),
            "use_scaled": True,
            "grid": {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
            }
        },
        "k-NN": {
            "model": KNeighborsClassifier(),
            "use_scaled": True,
            "grid": {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        },
        "MLP Classifier": {
            "model": MLPClassifier(random_state=42, max_iter=2000),
            "use_scaled": True,
            "grid": {
                "hidden_layer_sizes": [(50,), (50, 50), (100,)],
                "alpha": [0.0001, 0.001],
                "learning_rate_init": [0.001, 0.01]
            }
        },
        "Bagging MLP": {
            "model": BaggingClassifier(
                estimator=MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(50,)),
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": True,
            "grid": {
                "n_estimators": [5, 10],
                "max_samples": [0.7, 0.9, 1.0],
                "max_features": [0.8, 1.0]
            }
        }
    }

    metrics_summary = {}

    for name, exp in experiments.items():
        print(f"\n--- Treinando {name} ---")
        
        # Selecionar dados (classificadores exigidos na proposta utilizam escala)
        X_tr = X_train_scaled if exp["use_scaled"] else X_train
        X_te = X_test_scaled if exp["use_scaled"] else X_test
        
        # A. Avaliação Base (sem tuning)
        print("  Treinando modelo base...")
        base_model = exp["model"]
        base_model.fit(X_tr, y_train)
        y_pred_base = base_model.predict(X_te)
        
        acc_base = accuracy_score(y_test, y_pred_base)
        prec_base = precision_score(y_test, y_pred_base, average="macro", zero_division=0)
        rec_base = recall_score(y_test, y_pred_base, average="macro", zero_division=0)
        f1_base = f1_score(y_test, y_pred_base, average="macro", zero_division=0)
        
        print(f"  [Base] Accuracy: {acc_base:.4f} | F1-Macro: {f1_base:.4f}")
        
        # B. GridSearchCV para ajuste de hiperparâmetros
        print("  Buscando melhores hiperparâmetros (GridSearchCV)...")
        grid_search = GridSearchCV(
            estimator=exp["model"],
            param_grid=exp["grid"],
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_tr, y_train)
        best_model = grid_search.best_estimator_
        
        print(f"  Melhores Parâmetros: {grid_search.best_params_}")
        
        # Avaliação do melhor modelo
        y_pred_best = best_model.predict(X_te)
        acc_best = accuracy_score(y_test, y_pred_best)
        prec_best = precision_score(y_test, y_pred_best, average="macro", zero_division=0)
        rec_best = recall_score(y_test, y_pred_best, average="macro", zero_division=0)
        f1_best = f1_score(y_test, y_pred_best, average="macro", zero_division=0)
        
        print(f"  [Tuned] Accuracy: {acc_best:.4f} | F1-Macro: {f1_best:.4f}")
        
        # Salvar o modelo final
        model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
            
        # Salvar métricas para comparação
        metrics_summary[name] = {
            "base": {"accuracy": acc_base, "precision": prec_base, "recall": rec_base, "f1_macro": f1_base},
            "tuned": {"accuracy": acc_best, "precision": prec_best, "recall": rec_best, "f1_macro": f1_best},
            "best_params": grid_search.best_params_
        }

    # 3. Salvar métricas consolidadas em JSON
    metrics_path = RESULTS_DIR / "classification_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"\n✓ Processo de Classificação finalizado. Métricas salvas em: {metrics_path}")

if __name__ == "__main__":
    main()
