import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# 1. Carrega o conjunto de dados
df = pd.read_csv('ai4i2020.csv')

# Renomeando colunas apenas para facilitar a plotagem e legenda em português
colunas_PT = {
    'Air temperature [K]': 'Temperatura do Ar [K]',
    'Process temperature [K]': 'Temperatura do Processo [K]',
    'Rotational speed [rpm]': 'Velocidade de Rotação [rpm]',
    'Torque [Nm]': 'Torque [Nm]',
    'Tool wear [min]': 'Desgaste da Ferramenta [min]',
    'Machine failure': 'Falha da Maquina'
}
df_plot = df.rename(columns=colunas_PT)

# ==============================================================================
# GRÁFICO 1: HISTOGRAMAS (Distribuição Univariada das Variáveis Físicas)
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribuição Estatística das Variáveis Operacionais', fontweight='bold')

features = [
    'Temperatura do Ar [K]', 'Temperatura do Processo [K]', 
    'Velocidade de Rotação [rpm]', 'Torque [Nm]', 'Desgaste da Ferramenta [min]'
]

for i, col in enumerate(features):
    ax = axes[i // 3, i % 3]
    sns.histplot(data=df_plot, x=col, kde=True, ax=ax, color='royalblue')
    ax.set_title(f'Distribuição de {col.split(" [")[0]}')
    ax.set_ylabel('Frequência')

# Remove o último eixo vazio (posição 2,3)
fig.delaxes(axes[1, 2])
plt.tight_layout()
plt.savefig('histogramas_distribuicao.png', dpi=300)
plt.close()

# ==============================================================================
# GRÁFICO 2: BOXPLOTS (Relação Bivariada com a Variável Resposta de Falha)
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Análise de Boxplot: Variáveis Operacionais vs. Falha da Máquina', fontweight='bold')

for i, col in enumerate(features):
    ax = axes[i // 3, i % 3]
    sns.boxplot(data=df_plot, x='Falha da Maquina', y=col, ax=ax, palette='Set2')
    ax.set_xticklabels(['Normal (0)', 'Falha (1)'])
    ax.set_title(f'{col.split(" [")[0]} por Estado')
    ax.set_xlabel('Estado da Máquina')

fig.delaxes(axes[1, 2])
plt.tight_layout()
plt.savefig('boxplots_falha.png', dpi=300)
plt.close()

# ==============================================================================
# GRÁFICO 3: DISPERSÃO (Relação Física entre Rotação e Torque colorida por Falha)
# ==============================================================================
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_plot, 
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
plt.close()

print("Gráficos gerados com sucesso e salvos na pasta do projeto!")