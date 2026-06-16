# =====================================================================
# SCRIPT 5: XTREME GRADIENT BOOSTING (XGBoost) E COMPARAÇÃO FINAL
# INTEGRANTES: Otacílio, Alisson, André, Gabriel e Mateus
# =====================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

print("\n[ PREPARANDO DADOS PARA O XGBOOST ]")
df = pd.read_csv('ai4i2020.csv')

# 1. Tradução e Seleção
df_pt = df.rename(columns={
    'Air temperature [K]': 'Temperatura do Ar [K]',
    'Process temperature [K]': 'Temperatura do Processo [K]',
    'Rotational speed [rpm]': 'Velocidade de Rotação [rpm]',
    'Torque [Nm]': 'Torque [Nm]',
    'Tool wear [min]': 'Desgaste da Ferramenta [min]',
    'Machine failure': 'Falha da Maquina'
})

# 2. Feature Engineering
df_pt['Delta_Temperatura [K]'] = df_pt['Temperatura do Processo [K]'] - df_pt['Temperatura do Ar [K]']
df_pt['Potencia_Mecanica [W]'] = df_pt['Torque [Nm]'] * df_pt['Velocidade de Rotação [rpm]'] * (2 * np.pi / 60)

atributos = [
    'Temperatura do Ar [K]', 'Temperatura do Processo [K]', 
    'Velocidade de Rotação [rpm]', 'Torque [Nm]', 'Desgaste da Ferramenta [min]',
    'Delta_Temperatura [K]', 'Potencia_Mecanica [W]'
]
X = df_pt[atributos]
y = df_pt['Falha da Maquina']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# MODELO 3: XGBOOST COM TUNING DE HIPERPARÂMETROS
# =====================================================================
print("\n-> Treinando modelo XGBoost com Tuning...")

modelo_xgb = xgb.XGBClassifier(
    n_estimators=100,         # Número de árvores em sequência
    max_depth=5,              # Profundidade máxima de cada árvore
    learning_rate=0.1,        # Taxa de aprendizado (ajuste fino dos erros)
    subsample=0.8,            # Usa 80% dos dados por árvore para evitar overfitting
    colsample_bytree=0.8,     # Usa 80% das variáveis por árvore
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

modelo_xgb.fit(X_train, y_train)

acc_treino_xgb = accuracy_score(y_train, modelo_xgb.predict(X_train)) * 100
acc_teste_xgb = accuracy_score(y_test, modelo_xgb.predict(X_test)) * 100

# =====================================================================
# QUADRO DE AVALIAÇÃO FINAL DA MATÉRIA
# =====================================================================
print("\n" + "="*60)
print(" DESEMPENHO FINAL: XGBOOST vs OUTROS MODELOS")
print("="*60)
print(f"[ XGBOOST (Ajustado) ]")
print(f" - Acurácia de Treino: {acc_treino_xgb:.2f}%")
print(f" - Acurácia de Teste:  {acc_teste_xgb:.2f}%")
print("-" * 60)
print("-> O XGBoost otimiza os erros sequenciais, garantindo alta precisão.")
print("="*60 + "\n")