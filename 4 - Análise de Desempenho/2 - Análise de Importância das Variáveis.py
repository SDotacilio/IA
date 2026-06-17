
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("\n[ PREPARANDO DADOS PARA ANÁLISE DE IMPORTÂNCIA ]")
df = pd.read_csv('ai4i2020.csv') 

# 1. Tradução e Limpeza 
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TREINAMENTO DOS MODELOS AJUSTADOS

print("-> Treinando Modelos...")

# Instanciando os 3 modelos principais 
arvore = DecisionTreeClassifier(max_depth=5, min_samples_split=30, min_samples_leaf=15, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42)
xgboost = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)

# Treinando
arvore.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgboost.fit(X_train, y_train)

# EXTRAINDO AS IMPORTÂNCIAS
# Cria um DataFrame 
importancias = pd.DataFrame({
    'Atributo': atributos,
    'Árvore de Decisão': arvore.feature_importances_,
    'Random Forest': rf.feature_importances_,
    'XGBoost': xgboost.feature_importances_
})

# GERANDO OS GRÁFICOS  
print("-> Gerando Gráficos de Importância das Variáveis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparação da Importância das Variáveis por Modelo', fontsize=16, fontweight='bold', y=1.05)

modelos_nomes = ['Árvore de Decisão', 'Random Forest', 'XGBoost']
cores = ['#3498db', '#2ecc71', '#e74c3c']

for i, modelo_nome in enumerate(modelos_nomes):
    # Ordena do maior para o menor
    df_plot = importancias[['Atributo', modelo_nome]].sort_values(by=modelo_nome, ascending=False)
    
    sns.barplot(
        x=modelo_nome, 
        y='Atributo', 
        data=df_plot, 
        ax=axes[i], 
        color=cores[i],
        edgecolor='black'
    )
    
    axes[i].set_title(modelo_nome, fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Nível de Importância (0 a 1)')
    axes[i].set_ylabel('')
    axes[i].grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('4 - Análise de Desempenho/importancia_variaveis_comparacao.png', dpi=300, bbox_inches='tight')

print("-> Imagem 'importancia_variaveis_comparacao.png' salva com sucesso!")
print("\n[ PROCESSO CONCLUÍDO! ]\n")