import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Carrega o dataset original
df = pd.read_csv('ai4i2020.csv')

# Dicionário de tradução para blindar o relatório
traducao_colunas = {
    'Air temperature [K]': 'Temperatura do Ar [K]',
    'Process temperature [K]': 'Temperatura do Processo [K]',
    'Rotational speed [rpm]': 'Velocidade de Rotação [rpm]',
    'Torque [Nm]': 'Torque [Nm]',
    'Tool wear [min]': 'Desgaste da Ferramenta [min]',
    'Machine failure': 'Falha da Maquina'
}
df_pt = df.rename(columns=traducao_colunas)

# 2. SELEÇÃO DE VARIÁVEIS 
atributos_preditores = [
    'Temperatura do Ar [K]', 'Temperatura do Processo [K]', 
    'Velocidade de Rotação [rpm]', 'Torque [Nm]', 'Desgaste da Ferramenta [min]'
]

X = df_pt[atributos_preditores]
y = df_pt['Falha da Maquina']

# 3. Treinamento do Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_ai4i = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_ai4i.fit(X_train, y_train)

# 4. Extração e organização das importâncias
importancias = modelo_ai4i.feature_importances_
df_imp = pd.DataFrame({'Variável': X.columns, 'Importância (%)': importancias * 100})
df_imp = df_imp.sort_values(by='Importância (%)', ascending=False)

print("--- IMPORTÂNCIA DAS VARIÁVEIS (AI4I 2020) ---")
for index, row in df_imp.iterrows():
    print(f"{row['Variável']}: {row['Importância (%)']:.2f}%")

# 5. Plotagem do Gráfico Didático
plt.figure(figsize=(10, 5))
sns.barplot(data=df_imp, x='Importância (%)', y='Variável', palette='Blues_r')
plt.title('Seleção de Variáveis: Relevância dos Atributos (AI4I 2020)', fontweight='bold', pad=15)
plt.xlabel('Importância Relativa (%)')
plt.ylabel('Sensores da Fresadora')
plt.tight_layout()
plt.savefig('1 - Dados/importancia_variaveis_portugues.png', dpi=300)
plt.show()