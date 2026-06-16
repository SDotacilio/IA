# =====================================================================
# ETAPA 1: MODELO INICIAL DE ÁRVORE DE DECISÃO
# INTEGRANTES: Otacílio, Alisson, André, Gabriel e Mateus
# =====================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("\n[ CARREGANDO DADOS PARA O MODELO INICIAL ]")
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

# 2. Feature Engineering (As variáveis que criamos)
df_pt['Delta_Temperatura [K]'] = df_pt['Temperatura do Processo [K]'] - df_pt['Temperatura do Ar [K]']
df_pt['Potencia_Mecanica [W]'] = df_pt['Torque [Nm]'] * df_pt['Velocidade de Rotação [rpm]'] * (2 * np.pi / 60)

atributos = [
    'Temperatura do Ar [K]', 'Temperatura do Processo [K]', 
    'Velocidade de Rotação [rpm]', 'Torque [Nm]', 'Desgaste da Ferramenta [min]',
    'Delta_Temperatura [K]', 'Potencia_Mecanica [W]'
]
X = df_pt[atributos]
y = df_pt['Falha da Maquina']

# 3. Divisão dos dados em 80% Treino e 20% Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# TREINAMENTO DO MODELO INICIAL (Sem limites)
# =====================================================================
print("\n-> Treinando a Árvore de Decisão Padrão...")

arvore_inicial = DecisionTreeClassifier(random_state=42)
arvore_inicial.fit(X_train, y_train)

# Calculando as notas (acurácia)
acc_treino = accuracy_score(y_train, arvore_inicial.predict(X_train)) * 100
acc_teste = accuracy_score(y_test, arvore_inicial.predict(X_test)) * 100

print("\n" + "="*50)
print(" RESULTADOS DO MODELO INICIAL")
print("="*50)
print(f"Acurácia nos dados de TREINO: {acc_treino:.2f}%")
print(f"Acurácia nos dados de TESTE:  {acc_teste:.2f}%")
print(f"Profundidade da Árvore:       {arvore_inicial.get_depth()} níveis")
print(f"Total de Folhas (Regras):     {arvore_inicial.get_n_leaves()} regras")
print("="*50 + "\n")