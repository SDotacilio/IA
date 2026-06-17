
import pandas as pd

# Carrega o dataset (ajuste o nome do arquivo se necessário)
df = pd.read_csv('ai4i2020.csv')

# Verifica a contagem de valores nulos/missing por coluna
valores_faltantes = df.isnull().sum()
porcentagem_faltantes = (df.isnull().sum() / len(df)) * 100

# Cria uma tabela para o relatório
tabela_integridade = pd.DataFrame({
    'Valores Nulos': valores_faltantes,
    'Percentual (%)': porcentagem_faltantes
})

print("--- Análise Preliminar de Integridade ---")
print(tabela_integridade)