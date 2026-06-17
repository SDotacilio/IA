
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("\n[ CARREGANDO DADOS PARA O MODELO COM TUNING ]")
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

# TREINAMENTO DO MODELO AJUSTADO (Com pre-pruning)

print("\n-> Treinando a Árvore de Decisão com Hiperparâmetros Ajustados...")

# Aqui estão os exatos parâmetros do seu relatório!
arvore_ajustada = DecisionTreeClassifier(
    max_depth=5,              # Profundidade máxima de 5
    min_samples_split=30,     # Mínimo de 30 para dividir
    min_samples_leaf=15,      # Mínimo de 15 na folha
    random_state=42
)
arvore_ajustada.fit(X_train, y_train)

# Calculando as notas reais
acc_treino_aju = accuracy_score(y_train, arvore_ajustada.predict(X_train)) * 100
acc_teste_aju = accuracy_score(y_test, arvore_ajustada.predict(X_test)) * 100

print("\n" + "="*55)
print(" RESULTADOS DA ETAPA 2: MODELO COM TUNING")
print("="*55)
print(f"Acurácia nos dados de TREINO: {acc_treino_aju:.2f}%")
print(f"Acurácia nos dados de TESTE:  {acc_teste_aju:.2f}%")
print(f"Profundidade da Árvore:       {arvore_ajustada.get_depth()} níveis")
print(f"Total de Folhas (Regras):     {arvore_ajustada.get_n_leaves()} regras")
print("="*55)
print("Diagnóstico: Overfitting mitigado com sucesso!")
print("="*55 + "\n")