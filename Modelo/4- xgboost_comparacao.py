# =====================================================================
# SCRIPT 5: XGBOOST vs RANDOM FOREST (Comparação Final)
# INTEGRANTES: Otacílio, Alisson, André, Gabriel e Mateus
# =====================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

print("\n[ A CARREGAR DADOS PARA O XGBOOST ]")
df = pd.read_csv('ai4i2020.csv')

# 1. Tradução e Seleção (SEM PARÊNTESES RETOS PARA O XGBOOST NÃO RECLAMAR!)
df_pt = df.rename(columns={
    'Air temperature [K]': 'Temperatura do Ar K',
    'Process temperature [K]': 'Temperatura do Processo K',
    'Rotational speed [rpm]': 'Velocidade de Rotacao rpm',
    'Torque [Nm]': 'Torque Nm',
    'Tool wear [min]': 'Desgaste da Ferramenta min',
    'Machine failure': 'Falha da Maquina'
})

# 2. Feature Engineering
df_pt['Delta_Temperatura K'] = df_pt['Temperatura do Processo K'] - df_pt['Temperatura do Ar K']
df_pt['Potencia_Mecanica W'] = df_pt['Torque Nm'] * df_pt['Velocidade de Rotacao rpm'] * (2 * np.pi / 60)

atributos = [
    'Temperatura do Ar K', 'Temperatura do Processo K', 
    'Velocidade de Rotacao rpm', 'Torque Nm', 'Desgaste da Ferramenta min',
    'Delta_Temperatura K', 'Potencia_Mecanica W'
]
X = df_pt[atributos]
y = df_pt['Falha da Maquina']

# 3. Divisão Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# MODELO 1: RANDOM FOREST (Para comparação)
# =====================================================================
print("-> A treinar Random Forest (Ajustado)...")
floresta = RandomForestClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42)
floresta.fit(X_train, y_train)

acc_treino_rf = accuracy_score(y_train, floresta.predict(X_train)) * 100
acc_teste_rf = accuracy_score(y_test, floresta.predict(X_test)) * 100

# =====================================================================
# MODELO 2: XGBOOST COM TUNING (Os parâmetros do teu relatório)
# =====================================================================
print("-> A treinar o XGBoost (Gradient Boosting)...")

# Os hiperparâmetros exatos que estão no relatório do grupo!
modelo_xgb = xgb.XGBClassifier(
    n_estimators=100,         # 100 árvores sequenciais
    learning_rate=0.1,        # Taxa de aprendizagem conservadora
    max_depth=5,              # Restrição a 5 níveis de profundidade
    subsample=0.8,            # Apenas 80% das amostras por árvore
    colsample_bytree=0.8,     # Apenas 80% das variáveis por árvore
    random_state=42,
    eval_metric='logloss'     # Métrica interna de avaliação recomendada
)

modelo_xgb.fit(X_train, y_train)

acc_treino_xgb = accuracy_score(y_train, modelo_xgb.predict(X_train)) * 100
acc_teste_xgb = accuracy_score(y_test, modelo_xgb.predict(X_test)) * 100

# =====================================================================
# IMPRESSÃO DA COMPARAÇÃO FINAL
# =====================================================================
print("\n" + "="*60)
print(" BATALHA FINAL: RANDOM FOREST vs XGBOOST")
print("="*60)
print(f"[ RANDOM FOREST ]")
print(f" - Acurácia de Treino: {acc_treino_rf:.2f}%")
print(f" - Acurácia de Teste:  {acc_teste_rf:.2f}%")
print("-" * 60)
print(f"[ XGBOOST (Tuning) ]")
print(f" - Acurácia de Treino: {acc_treino_xgb:.2f}%")
print(f" - Acurácia de Teste:  {acc_teste_xgb:.2f}%")
print("="*60)

if acc_teste_xgb > acc_teste_rf:
    print(f"VENCEDOR: XGBoost superou em {(acc_teste_xgb - acc_teste_rf):.2f}%!")
elif acc_teste_xgb == acc_teste_rf:
    print("EMPATE TÉCNICO! Ambos os algoritmos chegaram ao limite físico dos dados.")
else:
    print("VENCEDOR: Random Forest superou o XGBoost nesta calibração.")
print("="*60 + "\n")