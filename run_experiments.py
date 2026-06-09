# /// script
# dependencies = [
#   "scikit-learn",
#   "pandas",
#   "numpy",
# ]
# ///

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def solve_question_2():
    print("==================================================")
    print("            RESOLVENDO A QUESTÃO 2")
    print("==================================================")
    
    # Dados da Tabela 1
    y_true = np.array(['A', 'A', 'B', 'B', 'B', 'A', 'B', 'A', 'A', 'B', 'A', 'B', 'B', 'A'])
    y_c1   = np.array(['A', 'B', 'A', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'A', 'B', 'B', 'A'])
    y_c2   = np.array(['B', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'B', 'B', 'A', 'B', 'B', 'B'])
    y_c3   = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'A', 'B'])
    
    # Calcular voto majoritário (Ensemble)
    # Para cada amostra, contar ocorrências de A e B
    y_ens = []
    for c1, c2, c3 in zip(y_c1, y_c2, y_c3):
        votes = [c1, c2, c3]
        majority = 'A' if votes.count('A') >= 2 else 'B'
        y_ens.append(majority)
    y_ens = np.array(y_ens)
    
    # Exibir predições do Ensemble
    print("Predições do Ensemble:")
    for idx, (real, c1, c2, c3, ens) in enumerate(zip(y_true, y_c1, y_c2, y_c3, y_ens), 1):
        status = "Correto" if real == ens else "Incorreto"
        print(f"ID {idx:02d}: Real={real} | C1={c1} | C2={c2} | C3={c3} | Ensemble={ens} ({status})")
        
    print("\n--------------------------------------------------")
    print("Resultados e Matrizes de Confusão (Questão 2):")
    print("--------------------------------------------------")
    
    classifiers = {
        "Classificador 1": y_c1,
        "Classificador 2": y_c2,
        "Classificador 3": y_c3,
        "Ensemble (Voto Majoritário)": y_ens
    }
    
    for name, preds in classifiers.items():
        acc = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds, labels=['A', 'B'])
        print(f"\n{name}:")
        print(f"Acurácia: {acc:.4f} ({acc*100:.1f}%)")
        print("Matriz de Confusão:")
        print("          Predito A | Predito B")
        print(f"Real A:     {cm[0,0]:>5}   |   {cm[0,1]:>5}")
        print(f"Real B:     {cm[1,0]:>5}   |   {cm[1,1]:>5}")
        
        # Formato LaTeX
        print("LaTeX Confusion Matrix code:")
        print(f"\\begin{{pmatrix}} {cm[0,0]} & {cm[0,1]} \\\\ {cm[1,0]} & {cm[1,1]} \\end{{pmatrix}}")
        print(f"Acurácia LaTeX: {acc*100:.2f}\\%")

def solve_question_3():
    print("\n==================================================")
    print("            RESOLVENDO A QUESTÃO 3")
    print("==================================================")
    
    # Carregar o Wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    class_names = wine.target_names
    
    print(f"Conjunto de dados Wine carregado:")
    print(f"Instâncias: {X.shape[0]}")
    print(f"Atributos: {X.shape[1]}")
    print(f"Classes: {len(class_names)} {list(class_names)}")
    
    # Definir validação cruzada
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Modelos base
    mlp_base = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=2000, random_state=42)
    dt_base = DecisionTreeClassifier(random_state=42)
    
    # Modelos de Bagging
    bag_dt = BaggingClassifier(estimator=dt_base, n_estimators=50, random_state=42)
    bag_mlp = BaggingClassifier(estimator=mlp_base, n_estimators=10, random_state=42)
    
    # Configurar experimentos
    modelos = {
        "MLP (Sem Normalização)": mlp_base,
        "MLP (Com Normalização)": Pipeline([("scaler", StandardScaler()), ("mlp", mlp_base)]),
        "Bagging DT (Sem Normalização)": bag_dt,
        "Bagging DT (Com Normalização)": Pipeline([("scaler", StandardScaler()), ("bag_dt", bag_dt)]),
        "Bagging MLP (Sem Normalização)": bag_mlp,
        "Bagging MLP (Com Normalização)": Pipeline([("scaler", StandardScaler()), ("bag_mlp", bag_mlp)]),
    }
    
    for name, model in modelos.items():
        print(f"\n--------------------------------------------------")
        print(f"Executando: {name}")
        print(f"--------------------------------------------------")
        
        # Realizar validação cruzada
        preds = cross_val_predict(model, X, y, cv=cv)
        
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)
        report = classification_report(y, preds, target_names=class_names, digits=4)
        
        print(f"Acurácia: {acc:.6f} ({acc*100:.2f}%)")
        print("Matriz de Confusão:")
        print(cm)
        print("Relatório de Classificação:")
        print(report)
        
        # Gerar tabelas LaTeX
        print("Matriz de Confusão em LaTeX:")
        print(f"\\begin{{pmatrix}} {cm[0,0]} & {cm[0,1]} & {cm[0,2]} \\\\ {cm[1,0]} & {cm[1,1]} & {cm[1,2]} \\\\ {cm[2,0]} & {cm[2,1]} & {cm[2,2]} \\end{{pmatrix}}")
        
        # Vamos estruturar os dados do classification report para facilitar LaTeX
        # Extrair precision, recall, f1-score para cada classe e macro/weighted averages
        cls_rep = classification_report(y, preds, target_names=class_names, output_dict=True)
        print("Tabela de Métricas em LaTeX:")
        for k in ['class_0', 'class_1', 'class_2', 'macro avg', 'weighted avg']:
            metrics = cls_rep[k]
            if k in class_names:
                name_str = f"Classe {k}"
            elif k == 'macro avg':
                name_str = "Média Macro"
            elif k == 'weighted avg':
                name_str = "Média Ponderada"
            else:
                name_str = k.replace('_', ' ').title()
            print(f"{name_str} & {metrics['precision']:.4f} & {metrics['recall']:.4f} & {metrics['f1-score']:.4f} & {metrics['support']:.0f} \\\\")

if __name__ == "__main__":
    solve_question_2()
    solve_question_3()
