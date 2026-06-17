
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("\n[ PREPARANDO DADOS PARA MEDIÇÃO DE TEMPO ]")
# Carrega a base
df = pd.read_csv('ai4i2020.csv') # Ajuste para 'Dados/ai4i2020.csv' se necessário

# Tradução (Sem colchetes)
df_pt = df.rename(columns={
    'Air temperature [K]': 'Temperatura do Ar K',
    'Process temperature [K]': 'Temperatura do Processo K',
    'Rotational speed [rpm]': 'Velocidade de Rotacao rpm',
    'Torque [Nm]': 'Torque Nm',
    'Tool wear [min]': 'Desgaste da Ferramenta min',
    'Machine failure': 'Falha da Maquina'
})

# Feature Engineering
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

# =====================================================================
# INICIALIZANDO OS MODELOS (Com os parâmetros finais)
# =====================================================================
modelos = {
    "Árvore de Decisão": DecisionTreeClassifier(max_depth=5, min_samples_split=30, min_samples_leaf=15, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
}

tempos_treino = {}
tempos_inferencia = {}

print("\n" + "="*70)
print(f"{'MODELO':<25} | {'TREINO (seg)':<15} | {'INFERÊNCIA (seg)':<15}")
print("="*70)

# =====================================================================
# CRONOMETRANDO
# =====================================================================
for nome, modelo in modelos.items():
    # 1. Medindo Tempo de TREINO (Aprender com os dados)
    inicio_treino = time.time()
    modelo.fit(X_train, y_train)
    fim_treino = time.time()
    tempos_treino[nome] = fim_treino - inicio_treino
    
    # 2. Medindo Tempo de INFERÊNCIA (Prever os dados do X_test inteiro - 2000 motores)
    inicio_inf = time.time()
    modelo.predict(X_test)
    fim_inf = time.time()
    tempos_inferencia[nome] = fim_inf - inicio_inf
    
    print(f"{nome:<25} | {tempos_treino[nome]:<15.5f} | {tempos_inferencia[nome]:<15.5f}")

print("="*70)

# =====================================================================
# GERANDO GRÁFICO COMPARATIVO
# =====================================================================
print("\n-> Gerando Gráfico de Tempos...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

modelos_nomes = list(modelos.keys())
cores = ['#3498db', '#2ecc71', '#e74c3c']

# Gráfico 1: Treino
sns.barplot(x=modelos_nomes, y=list(tempos_treino.values()), ax=axes[0], palette=cores)
axes[0].set_title('Tempo de Treinamento Computacional', fontweight='bold')
axes[0].set_ylabel('Tempo (Segundos)')

# Gráfico 2: Inferência
sns.barplot(x=modelos_nomes, y=list(tempos_inferencia.values()), ax=axes[1], palette=cores)
axes[1].set_title('Tempo de Inferência (Tomada de Decisão)', fontweight='bold')
axes[1].set_ylabel('Tempo (Segundos)')

plt.suptitle('Comparação de Desempenho Computacional', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('4 - Análise de Desempenho/tempos_execucao.png', dpi=300, bbox_inches='tight')
print("-> Imagem 'tempos_execucao.png' salva com sucesso!")