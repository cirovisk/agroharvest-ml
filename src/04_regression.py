import os
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    print("ETAPA 4 (Regressão): Modelagem de Produtividade Agrícola")
    print("=" * 60)

    # 1. Carregar dados e garantir que sejam buffers mutáveis (.copy())
    X_train = load_pkl("X_train.pkl").copy()
    X_test = load_pkl("X_test.pkl").copy()
    X_train_scaled = load_pkl("X_train_scaled.pkl").copy()
    X_test_scaled = load_pkl("X_test_scaled.pkl").copy()
    y_train = load_pkl("y_train_reg.pkl").copy()
    y_test = load_pkl("y_test_reg.pkl").copy()

    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

    # 2. Definição dos Modelos e Param Grids
    # Definimos um dicionário com os modelos, se necessitam de escala, grids de parâmetros e os estimadores base
    experiments = {
        "MLP Regressor": {
            "model": MLPRegressor(random_state=42, max_iter=2000),
            "use_scaled": True,
            "grid": {
                "hidden_layer_sizes": [(50, 50), (100,)],
                "alpha": [0.0001, 0.001],
                "learning_rate_init": [0.001, 0.01]
            }
        },
        "Random Forest Regressor": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "use_scaled": False,
            "grid": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5]
            }
        },
        "XGBoost Regressor": {
            "model": XGBRegressor(random_state=42, n_jobs=-1, eval_metric="rmse"),
            "use_scaled": False,
            "grid": {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2]
            }
        }
    }

    metrics_summary = {}

    for name, exp in experiments.items():
        print(f"\n--- Treinando {name} ---")
        
        # Selecionar dados de acordo com a sensibilidade de escala
        X_tr = X_train_scaled if exp["use_scaled"] else X_train
        X_te = X_test_scaled if exp["use_scaled"] else X_test
        
        # A. Avaliação Base (sem tuning)
        print("  Treinando modelo base...")
        base_model = exp["model"]
        base_model.fit(X_tr, y_train)
        y_pred_base = base_model.predict(X_te)
        
        mae_base = mean_absolute_error(y_test, y_pred_base)
        rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
        r2_base = r2_score(y_test, y_pred_base)
        
        print(f"  [Base] MAE: {mae_base:.2f} | RMSE: {rmse_base:.2f} | R²: {r2_base:.4f}")
        
        # B. GridSearchCV para ajuste de hiperparâmetros
        print("  Buscando melhores hiperparâmetros (GridSearchCV)...")
        grid_search = GridSearchCV(
            estimator=exp["model"],
            param_grid=exp["grid"],
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_tr, y_train)
        best_model = grid_search.best_estimator_
        
        print(f"  Melhores Parâmetros: {grid_search.best_params_}")
        
        # Avaliação do melhor modelo
        y_pred_best = best_model.predict(X_te)
        mae_best = mean_absolute_error(y_test, y_pred_best)
        rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
        r2_best = r2_score(y_test, y_pred_best)
        
        print(f"  [Tuned] MAE: {mae_best:.2f} | RMSE: {rmse_best:.2f} | R²: {r2_best:.4f}")
        
        # Salvar o modelo final
        model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
            
        # Salvar métricas para comparação
        metrics_summary[name] = {
            "base": {"mae": mae_base, "rmse": rmse_base, "r2": r2_base},
            "tuned": {"mae": mae_best, "rmse": rmse_best, "r2": r2_best},
            "best_params": grid_search.best_params_
        }

    # 3. Salvar métricas consolidadas em JSON
    metrics_path = RESULTS_DIR / "regression_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"\n✓ Processo de Regressão finalizado. Métricas salvas em: {metrics_path}")

if __name__ == "__main__":
    main()
