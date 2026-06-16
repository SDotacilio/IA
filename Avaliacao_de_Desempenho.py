# =====================================================================
# SCRIPT 8: AVALIAÇÃO DE DESEMPENHO GLOBAL E COMPARAÇÃO DE MODELOS
# INTEGRANTES: Otacílio, Alisson, André, Gabriel e Mateus
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

print("\n[ PREPARANDO DADOS PARA A AVALIAÇÃO FINAL ]")
df = pd.read_csv('ai4i2020.csv')

# 1. Tradução (Sem colchetes para o XGBoost funcionar)
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

# =====================================================================
# TREINAMENTO DE TODOS OS MODELOS
# =====================================================================
print("-> Treinando Modelos para Comparação...")

modelos = {
    "Árvore Inicial (Overfitting)": DecisionTreeClassifier(random_state=42),
    "Árvore Ajustada (Tuning)": DecisionTreeClassifier(max_depth=5, min_samples_split=30, min_samples_leaf=15, random_state=42),
    "Random Forest (Tuning)": RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42),
    "XGBoost (Tuning)": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
}

resultados = {}
previsoes_proba = {}

# Variáveis para a Curva ROC
plt.figure(figsize=(10, 8))

print("\n" + "="*80)
print(f"{'MODELO':<30} | {'ACURÁCIA':<10} | {'SENSIBILIDADE':<15} | {'ESPECIFICIDADE':<15}")
print("="*80)

# Preparando figura para as Matrizes de Confusão (Grid 2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (nome, modelo) in enumerate(modelos.items()):
    # Treino
    modelo.fit(X_train, y_train)
    
    # Previsões de classe (0 ou 1)
    y_pred = modelo.predict(X_test)
    
    # Previsões de probabilidade (para a curva ROC)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    previsoes_proba[nome] = y_prob
    
    # Cálculo das métricas usando a Matriz de Confusão
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    acuracia = accuracy_score(y_test, y_pred)
    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0 # Taxa de Acerto em Falhas reais
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0 # Taxa de Acerto em Motores Bons reais
    
    print(f"{nome:<30} | {acuracia*100:>8.2f}% | {sensibilidade*100:>13.2f}% | {especificidade*100:>13.2f}%")
    
    # Desenhando a Matriz de Confusão
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f'Matriz: {nome}', fontweight='bold')
    axes[i].set_xlabel('Previsão da IA')
    axes[i].set_ylabel('Realidade (Chão de Fábrica)')
    axes[i].set_xticklabels(['Normal (0)', 'Falha (1)'])
    axes[i].set_yticklabels(['Normal (0)', 'Falha (1)'])

    # Desenhando a Curva ROC para este modelo
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(1) # Volta para a figura da ROC
    plt.plot(fpr, tpr, lw=2, label=f'{nome} (AUC = {roc_auc:.3f})')

print("="*80)

# =====================================================================
# SALVANDO OS GRÁFICOS
# =====================================================================

# Finaliza Gráfico das Matrizes
fig.suptitle('Comparação de Matrizes de Confusão', fontsize=18, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('matrizes_confusao.png', dpi=300, bbox_inches='tight')
print("\n-> Imagem 'matrizes_confusao.png' gerada com sucesso!")

# Finaliza Gráfico da Curva ROC
plt.figure(1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Acaso (Sorteio)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (Alarme Falso)', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
plt.title('Curva ROC Comparativa dos Modelos', fontsize=15, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('curva_roc.png', dpi=300, bbox_inches='tight')
print("-> Imagem 'curva_roc.png' gerada com sucesso!")

print("\n[ AVALIAÇÃO CONCLUÍDA! VERIFIQUE AS IMAGENS E OS DADOS IMPRESSOS. ]\n")