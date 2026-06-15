# =====================================================================
# TRABALHO 2: Manutenção Preditiva com Dataset AI4I 2020
# INTEGRANTES: Otacílio, Alisson, André, Gabriel e Mateus
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("-> Iniciando o processamento do Dataset AI4I 2020...\n")

# 1. CARREGAR O ARQUIVO ORIGINAL (Em inglês)
df = pd.read_csv('ai4i2020.csv')

# 2. TRADUZIR AS COLUNAS IMPORTANTES
# Note que deixamos 'UDI' e 'Product ID' de fora do dicionário de propósito
traducao_colunas = {
    'Air temperature [K]': 'Temperatura do Ar [K]',
    'Process temperature [K]': 'Temperatura do Processo [K]',
    'Rotational speed [rpm]': 'Velocidade de Rotação [rpm]',
    'Torque [Nm]': 'Torque [Nm]',
    'Tool wear [min]': 'Desgaste da Ferramenta [min]',
    'Machine failure': 'Falha da Maquina'
}
df_pt = df.rename(columns=traducao_colunas)

# 3. CRIAÇÃO DE NOVAS VARIÁVEIS (FEATURE ENGINEERING)
# Nova Variável 1: Diferença de Temperatura (Delta T)
df_pt['Delta_Temperatura [K]'] = df_pt['Temperatura do Processo [K]'] - df_pt['Temperatura do Ar [K]']

# Nova Variável 2: Potência Mecânica Real (P = Torque * Velocidade Angular)
df_pt['Potencia_Mecanica [W]'] = df_pt['Torque [Nm]'] * df_pt['Velocidade de Rotação [rpm]'] * (2 * np.pi / 60)

print("-> Novas variáveis (Delta T e Potência) criadas com sucesso!")

# 4. SELEÇÃO DE VARIÁVEIS (Filtrando apenas o que vai para a IA)
# O 'UDI' e 'Product ID' são ignorados aqui porque não colocamos eles nessa lista
atributos_preditores = [
    'Temperatura do Ar [K]', 'Temperatura do Processo [K]', 
    'Velocidade de Rotação [rpm]', 'Torque [Nm]', 'Desgaste da Ferramenta [min]',
    'Delta_Temperatura [K]', 'Potencia_Mecanica [W]'
]

X = df_pt[atributos_preditores] # X nasce limpo, apenas com dados físicos relevantes
y = df_pt['Falha da Maquina']

# 5. TREINAMENTO DO MODELO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_ai4i = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_ai4i.fit(X_train, y_train)

print("-> Random Forest treinado com o novo dataset!")

# 6. CALCULAR E MOSTRAR A IMPORTÂNCIA ATUALIZADA
importancias = modelo_ai4i.feature_importances_
df_imp = pd.DataFrame({'Variável': X.columns, 'Importância (%)': importancias * 100})
df_imp = df_imp.sort_values(by='Importância (%)', ascending=False)

print("\n--- RELEVÂNCIA DAS VARIÁVEIS NO MODELO FINAL ---")
for index, row in df_imp.iterrows():
    print(f"{row['Variável']}: {row['Importância (%)']:.2f}%")

# 7. GERAR O GRÁFICO PARA O PDF
plt.figure(figsize=(11, 5))
sns.barplot(data=df_imp, x='Importância (%)', y='Variável', palette='Blues_r')
plt.title('Seleção de Variáveis: Relevância dos Atributos (AI4I 2020)', fontweight='bold', pad=15)
plt.xlabel('Importância Relativa (%)')
plt.ylabel('Atributos da Fresadora')
plt.tight_layout()
plt.savefig('importancia_variaveis_ai4i.png', dpi=300)
print("\n-> Gráfico 'importancia_variaveis_ai4i.png' salvo!")
plt.show()