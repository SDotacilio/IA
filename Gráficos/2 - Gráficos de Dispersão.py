# =====================================================================
# SCRIPT DE EXPLORAÇÃO: GRÁFICOS DE DISPERSÃO (SCATTER PLOTS)
# INTEGRANTES: Otacílio, Alisson, André, Gabriel e Mateus
# =====================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("[ PREPARANDO OS DADOS PARA OS GRÁFICOS ]")
# 1. Carrega a base de dados (Ajuste o caminho se estiver dentro da pasta 'Dados/')
# Se o csv estiver na mesma pasta, deixe só 'ai4i2020.csv'
df = pd.read_csv('ai4i2020.csv')

# 2. Tradução das colunas
df_pt = df.rename(columns={
    'Air temperature [K]': 'Temperatura do Ar [K]',
    'Process temperature [K]': 'Temperatura do Processo [K]',
    'Rotational speed [rpm]': 'Velocidade de Rotação [rpm]',
    'Torque [Nm]': 'Torque [Nm]',
    'Tool wear [min]': 'Desgaste da Ferramenta [min]',
    'Machine failure': 'Falha da Maquina'
})

# 3. Criando a variável Delta T (necessária para o Gráfico 3)
df_pt['Delta_Temperatura [K]'] = df_pt['Temperatura do Processo [K]'] - df_pt['Temperatura do Ar [K]']


# =====================================================================
# GRÁFICO 1: DESGASTE VS TORQUE (FALHAS DE SOBRECARGA)
# =====================================================================
print("\n-> Gerando Gráfico de Dispersão 1 (Desgaste vs Torque)...")

plt.figure(figsize=(10, 6))

# Cria o gráfico de dispersão
sns.scatterplot(
    data=df_pt, 
    x='Desgaste da Ferramenta [min]', 
    y='Torque [Nm]', 
    hue='Falha da Maquina', 
    palette={0: '#2ecc71', 1: '#9b59b6'}, # Verde para normal, Roxo para falha
    alpha=0.7,
    edgecolor=None
)

# Textos e Títulos
plt.title('Análise Multivariada: Torque vs. Desgaste da Ferramenta', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Tempo de Desgaste da Ferramenta [min]', fontsize=12)
plt.ylabel('Torque Efetivo [Nm]', fontsize=12)

# Ajusta a legenda
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['Operação Segura (0)', 'Falha por Sobrecarga (1)'], title='Status do Motor')

# Salva na pasta
plt.tight_layout()
plt.savefig('dispersao_torque_desgaste.png', dpi=300)
print("-> Gráfico 'dispersao_torque_desgaste.png' salvo com sucesso!")
plt.show()


# =====================================================================
# GRÁFICO 2: ROTAÇÃO VS DELTA T (FALHAS TÉRMICAS / HDF)
# =====================================================================
print("\n-> Gerando Gráfico de Dispersão 2 (Rotação vs Delta T)...")

plt.figure(figsize=(10, 6))

# Cria o gráfico de dispersão
sns.scatterplot(
    data=df_pt, 
    x='Velocidade de Rotação [rpm]', 
    y='Delta_Temperatura [K]', 
    hue='Falha da Maquina', 
    palette={0: '#95a5a6', 1: '#e67e22'}, # Cinza para normal, Laranja para falha térmica
    alpha=0.7,
    edgecolor=None
)

# Textos e Títulos
plt.title('Análise Multivariada: Velocidade de Rotação vs. Gradiente Térmico', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Velocidade de Rotação [rpm]', fontsize=12)
plt.ylabel('Diferença de Temperatura (Processo - Ar) [K]', fontsize=12)

# Adiciona linhas pontilhadas para mostrar exatamente onde a IA vai cortar
plt.axvline(x=1380, color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=8.6, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Ajusta a legenda
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['Operação Segura (0)', 'Falha por Aquecimento (1)'], title='Status do Motor')

# Salva na pasta
plt.tight_layout()
plt.savefig('dispersao_rotacao_temperatura.png', dpi=300)
print("-> Gráfico 'dispersao_rotacao_temperatura.png' salvo com sucesso!")
plt.show()


# ==============================================================================
# GRÁFICO 3: DISPERSÃO (Relação Física entre Rotação e Torque colorida por Falha)
# ==============================================================================
print("\n-> Gerando Gráfico de Dispersão 3 (Rotação vs Torque)...")

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_pt,  # <-- CORRIGIDO: Antes estava df_plot, agora é df_pt
    x='Velocidade de Rotação [rpm]', 
    y='Torque [Nm]', 
    hue='Falha da Maquina', 
    palette={0: 'lightgrey', 1: 'crimson'},
    alpha=0.7
)

plt.title('Gráfico de Dispersão: Velocidade de Rotação vs. Torque', fontweight='bold')
plt.xlabel('Velocidade de Rotação [rpm]')
plt.ylabel('Torque [Nm]')
plt.legend(title='Estado', labels=['Normal (0)', 'Falha (1)'])

plt.tight_layout()
plt.savefig('dispersao_rotacao_torque.png', dpi=300)
print("-> Gráfico 'dispersao_rotacao_torque.png' salvo com sucesso!")
plt.show()

print("\n[ TODOS OS GRÁFICOS FORAM GERADOS E SALVOS COM SUCESSO! ]")