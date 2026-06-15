# =====================================================================
# PROJETO DE EXTENSÃO: Monitoramento de Falhas em Motores Elétricos
# ALGORITMO: Classificação (Random Forest - Floresta Aleatória)
# INTEGRANTES: Otacílio, Alisson, André, Gabriel e Mateus
# =====================================================================

import pandas as pd              
import numpy as np               
import matplotlib.pyplot as plt  
import seaborn as sns            
import time
import serial                    

# Importações para o modelo e avaliação
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier    
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 

print("Passo 1: Bibliotecas importadas com sucesso!")

# ---------------------------------------------------------------------
# METODOLOGIA DE EXECUÇÃO - PASSO 2: COLETA DE DADOS (COM RUÍDO REALISTA)
# ---------------------------------------------------------------------
np.random.seed(42)
n_amostras = 500

# 1. MOTORES SAUDÁVEIS 
vibracao_normal = np.random.normal(loc=0.3, scale=0.15, size=n_amostras // 2)
temp_normal = np.random.normal(loc=30.0, scale=3.0, size=n_amostras // 2)
falha_normal = np.zeros(n_amostras // 2) # <--- CORRIGIDO: Recriado aqui

# 2. MOTORES EM ESTRESSE (Zona de dúvida/sobreposição)
vibracao_falha = np.random.normal(loc=0.6, scale=0.25, size=n_amostras // 2) 
temp_falha = np.random.normal(loc=35.0, scale=4.0, size=n_amostras // 2)
falha_falha = np.ones(n_amostras // 2) # <--- CORRIGIDO: Recriado aqui

# Juntando no DataFrame
df = pd.DataFrame({
    'vibracao_rms': np.concatenate([vibracao_normal, vibracao_falha]),
    'temperatura_c': np.concatenate([temp_normal, temp_falha]),
    'falha_24h': np.concatenate([falha_normal, falha_falha])
})

# ---------------------------------------------------------------------
# METODOLOGIA DE EXECUÇÃO - PASSO 4: TREINAMENTO DO MODELO
# ---------------------------------------------------------------------
X = df[['vibracao_rms', 'temperatura_c']]
y = df['falha_24h']
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTreinando o cérebro do Random Forest...")
modelo_motor = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_motor.fit(X_treino, y_treino)
print("-> Random Forest treinado com sucesso!")
# =====================================================================
# EXTRAÇÃO DA MATRIZ DE CONFUSÃO (PROVA MATEMÁTICA PARA A BANCA)
# =====================================================================
print("\n" + "-"*50)
print(" GERANDO MÉTRICAS DE AVALIAÇÃO DO MODELO")
print("-" * 50)

# 1. Faz as previsões com a base de teste
y_pred = modelo_motor.predict(X_teste)
cm = confusion_matrix(y_teste, y_pred)

print("Matriz de Confusão Bruta (Valores no terminal):")
print(cm)

# MAIOR SACADA: Criando a matriz de textos personalizados que vai aparecer no gráfico!
rotulos_quadrados = np.array([
    [f"VN (Verdadeiro Negativo)\n\nAcertou Normal: {cm[0,0]}", f"FP (Falso Positivo)\n\nAlarme Falso: {cm[0,1]}"],
    [f"FN (Falso Negativo)\n\nFalha Oculta: {cm[1,0]}", f"VP (Verdadeiro Positivo)\n\nAcertou Falha: {cm[1,1]}"]
])

# 2. Configura o tamanho da imagem
plt.figure(figsize=(8, 6))

# 3. Desenha o mapa de calor (Heatmap) com os nossos textos personalizados
sns.heatmap(
    cm, 
    annot=rotulos_quadrados, 
    fmt="", 
    cmap="Blues", 
    cbar=True,
    xticklabels=['Normal (0)', 'Falha (1)'], 
    yticklabels=['Normal (0)', 'Falha (1)'],
    annot_kws={"size": 11, "weight": "bold"} # Deixa o texto interno destacado
)

# 4. Configura os títulos dos eixos de forma clara
plt.title("Matriz de Confusão Didática - Validação da Bancada", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Previsão da Inteligência Artificial (Predicted)", fontsize=12, labelpad=10)
plt.ylabel("Estado Real do Motor na Bancada (True)", fontsize=12, labelpad=10)

# 5. Salva a imagem na pasta do projeto
plt.tight_layout()
plt.savefig("matriz_confusao_motor.png", dpi=300)
print("\n-> Gráfico 'matriz_confusao_motor.png' salvo com sucesso!")
print("-" * 50 + "\n")

# Mostra a tela com o novo gráfico na tela
plt.show()

# =====================================================================
# INTERFÁCIA SERIAL DE EXTRAÇÃO NUMÉRICA (PARSER INTELIGENTE)
# =====================================================================
print("\n" + "="*65)
print("       CONFIGURANDO LEITURA SERIAL EM TEMPO REAL (ARDUINO MEGA)")
print("="*65)

PORTA_SERIAL = 'COM3' 
BAUD_RATE = 115200 

try:
    arduino = serial.Serial(PORTA_SERIAL, BAUD_RATE, timeout=1)
    time.sleep(2) 
    print("=== CONEXÃO ESTABELECIDA COM SUCESSO ===")
    print("Filtrando rótulos do Serial Plotter automaticamente...\n")
    print("VIBRAÇÃO | TEMPERATURA | PROBABILIDADE DE FALHA | STATUS")
    print("-" * 75)

    while True:
        linha_bruta = arduino.readline().decode('utf-8', errors='ignore').strip()
        
        if linha_bruta:
            if "Sistema Pronto" in linha_bruta:
                continue
                
            linha_limpa = linha_bruta
            
            tags_para_remover = [
                "Vibracao:", "Temp_Motor:", "Lim_V:", "Lim_T:", 
                "!!! ALERTA VIBRACAO !!!", "!!! ALERTA CALOR !!!"
            ]
            for tag in tags_para_remover:
                linha_limpa = linha_limpa.replace(tag, "")
            
            partes = linha_limpa.split()
            
            try:
                valores = []
                for x in partes:
                    x_limpo = x.strip().replace(',', '.')
                    if x_limpo.replace('.', '', 1).replace('-', '', 1).isdigit():
                        valores.append(float(x_limpo))
                
                if len(valores) >= 2:
                    vibracao_atual = valores[0]
                    temperatura_atual = valores[1]
                    
                    novo_dado = pd.DataFrame({
                        'vibracao_rms': [vibracao_atual],
                        'temperatura_c': [temperatura_atual]
                    })
                    
                    probabilidades = modelo_motor.predict_proba(novo_dado)[0]
                    chance_falha = probabilidades[1] * 100 
                    
                    if chance_falha < 30:
                        status = "SISTEMA SEGURO (OK)"
                    elif chance_falha < 70:
                        status = "ALERTA: OPERAÇÃO EM ESTRESSE"
                    else:
                        status = "PERIGO: FALHA IMINENTE!"
                        
                    print(f"{vibracao_atual:8.2f} | {temperatura_atual:10.2f}°C | {chance_falha:20.1f}% | {status}")
                    
            except Exception:
                continue

except serial.SerialException:
    print(f"\n[ERRO] Não foi possível acessar a porta {PORTA_SERIAL}.")