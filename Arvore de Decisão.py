# =====================================================================
# MODELO INICIAL: ÁRVORE DE DECISÃO (TESTE DE OVERFITTING)
# =====================================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("\n-> Treinando Modelo Inicial (Árvore de Decisão)...")

# Cria a árvore SEM nenhum limite de profundidade (para forçar o overfitting)
arvore_inicial = DecisionTreeClassifier(random_state=42)
arvore_inicial.fit(X_train, y_train)

# 1. Calcula a nota da IA na prova que ela já viu (Treino)
y_pred_treino = arvore_inicial.predict(X_train)
acuracia_treino = accuracy_score(y_train, y_pred_treino) * 100

# 2. Calcula a nota da IA na prova que ela nunca viu (Teste)
y_pred_teste = arvore_inicial.predict(X_test)
acuracia_teste = accuracy_score(y_test, y_pred_teste) * 100

# 3. Extrai a profundidade que a árvore atingiu
profundidade = arvore_inicial.get_depth()
folhas = arvore_inicial.get_n_leaves()

print("-" * 50)
print(f"Acurácia nos dados de TREINO: {acuracia_treino:.2f}% (A IA decorou as respostas)")
print(f"Acurácia nos dados de TESTE:  {acuracia_teste:.2f}% (A IA no mundo real)")
print(f"Profundidade da Árvore:       {profundidade} níveis")
print(f"Quantidade de Folhas:         {folhas} regras criadas")
print("-" * 50)