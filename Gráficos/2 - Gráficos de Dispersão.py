# =====================================================================
# GRÁFICO 2: DESGASTE VS TORQUE (FALHAS DE SOBRECARGA)
# =====================================================================
print("\n-> Gerando Gráfico de Dispersão 2 (Desgaste vs Torque)...")

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
# GRÁFICO 3: ROTAÇÃO VS DELTA T (FALHAS TÉRMICAS / HDF)
# =====================================================================
print("\n-> Gerando Gráfico de Dispersão 3 (Rotação vs Delta T)...")

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
# GRÁFICO 4: DISPERSÃO (Relação Física entre Rotação e Torque colorida por Falha)
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