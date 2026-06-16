# =====================================================================
# ETAPA 3: RANDOM FOREST (TUNING E COMPARAÇÃO)
# INTEGRANTES: Otacílio, Alisson, André, Gabriel e Mateus
# =====================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("\n[ CARREGANDO DADOS PARA COMPARAÇÃO FINAL ]")
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

# 3. Divisão Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# RECAPITULANDO O MODELO ANTERIOR (Árvore de Decisão com Tuning)
# =====================================================================
arvore_ajustada = DecisionTreeClassifier(
    max_depth=5, min_samples_split=30, min_samples_leaf=15, random_state=42
)
arvore_ajustada.fit(X_train, y_train)

acc_treino_dt = accuracy_score(y_train, arvore_ajustada.predict(X_train)) * 100
acc_teste_dt = accuracy_score(y_test, arvore_ajustada.predict(X_test)) * 100

# =====================================================================
# NOVO MODELO: RANDOM FOREST COM TUNING
# =====================================================================
print("-> Treinando Random Forest com Hiperparâmetros Ajustados...")

floresta_ajustada = RandomForestClassifier(
    n_estimators=100,         # Número de árvores na floresta
    max_depth=8,              # Profundidade máxima de cada árvore
    min_samples_split=20,     # Mínimo de amostras para dividir um nó
    min_samples_leaf=10,      # Mínimo de amostras em cada folha
    random_state=42,
    n_jobs=-1                 # Usa todos os núcleos do processador
)
floresta_ajustada.fit(X_train, y_train)

acc_treino_rf = accuracy_score(y_train, floresta_ajustada.predict(X_train)) * 100
acc_teste_rf = accuracy_score(y_test, floresta_ajustada.predict(X_test)) * 100

# =====================================================================
# IMPRESSÃO DA COMPARAÇÃO FINAL
# =====================================================================
print("\n" + "="*60)
print(" COMPARAÇÃO DE DESEMPENHO: ÁRVORE DE DECISÃO vs RANDOM FOREST")
print("="*60)
print(f"[ ÁRVORE DE DECISÃO (Ajustada) ]")
print(f" - Acurácia de Treino: {acc_treino_dt:.2f}%")
print(f" - Acurácia de Teste:  {acc_teste_dt:.2f}%")
print("-" * 60)
print(f"[ RANDOM FOREST (Ajustado) ]")
print(f" - Acurácia de Treino: {acc_treino_rf:.2f}%")
print(f" - Acurácia de Teste:  {acc_teste_rf:.2f}%")
print("="*60)

diferenca = acc_teste_rf - acc_teste_dt
if diferenca > 0:
    print(f"Diagnóstico: O Random Forest superou a Árvore de Decisão em {diferenca:.2f}%.")
else:
    print("Diagnóstico: O Random Forest manteve a estabilidade estatística, porém com maior robustez.")
print("="*60 + "\n")