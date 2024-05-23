# Algoritmos de Aprendizado de Máquina

## Introdução

Este repositório contém a implementação de diversos algoritmos de aprendizado de máquina. Estes algoritmos são amplamente utilizados em problemas de classificação e regressão. Abaixo, fornecemos uma breve descrição de cada algoritmo implementado, bem como instruções sobre como executar os códigos.

## Modelo de Classificação
Técnicas de Classificação:
<ul>
  <li>Máquinas de Vetores de Suporte (SVM)</li>
  <li>Árvores de Decisão</li>
  <li>Regras de Classificação</li>
  <li>Naive Bayes</li>
  <li>K-Nearest Neighbors (KNN)</li>
</ul>

### Naive Bayes

~~~python
from sklearn.naive_bayes import GaussianNB

# Dados de exemplo
X = [[100, 20], [150, 30], [120, 25], [140, 28]]
y = ['Não Spam', 'Spam', 'Não Spam', 'Spam']

# Treinando o modelo
model = GaussianNB()
model.fit(X, y)


# Previsão para um novo e-mail
novo_email = [[130, 22]]
resultado = model.predict(novo_email)
print(f"Previsão para o novo e-mail: {resultado[0]}")
~~~
![NaiveBayes](https://github.com/motielk/Regressao-Classificacao-BI/assets/49123696/a14a11d4-9979-4c99-87cf-312c14278ca3)

~~~python
# Passo 1: Importar as bibliotecas necessárias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
emails = [
    "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
    "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
    "Confira as novas ofertas da loja. Não perca!",
    "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
    "Lembrete: pagamento da fatura do cartão de crédito vence amanhã."
]

labels = [1, 1, 1, 0, 0]  # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo
model = MultinomialNB()  # Criar o modelo Naive Bayes multinomial
model.fit(X_train, y_train)  # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(X_test)  # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions)  # Calcular a precisão do modelo
print("Accuracy:", accuracy)
~~~

![NaiveBayes2](https://github.com/motielk/Regressao-Classificacao-BI/assets/49123696/deb977d4-5c7a-489e-820b-e77ad63abecd)

### K-Nearest Neighbors (KNN)
~~~python
# Passo 1: Importar as bibliotecas necessárias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
# Suponha que temos um conjunto de dados com e-mails e seus rótulos (spam ou não spam)
# Aqui está um exemplo simples de um conjunto de dados fictício:
emails = [
    "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
    "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
    "Confira as novas ofertas da loja. Não perca!",
    "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
    "Lembrete: pagamento da fatura do cartão de crédito vence amanhã."
]

labels = [1, 1, 1, 0, 0]  # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo KNN
model = KNeighborsClassifier(n_neighbors=3)  # Criar o modelo KNN com 3 vizinhos
model.fit(X_train, y_train)  # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(X_test)  # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions)  # Calcular a precisão do modelo
print("Accuracy:", accuracy)
~~~
![KNN](https://github.com/motielk/Regressao-Classificacao-BI/assets/49123696/3f9a667e-106e-4419-a9e7-ee71153a5c7d)

### Árvores de Decisão

~~~python
# Passo 1: Importar as bibliotecas necessárias
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
# Suponha que temos um conjunto de dados com e-mails e seus rótulos (spam ou não spam)
# Aqui está um exemplo simples de um conjunto de dados fictício:

emails = {
    "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
    "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
    "Confira as novas ofertas da loja, Não perca!",
    "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
    "Lembrete: pagamento da fatura do cartão de crédito vence amanhã.",
}

labels = [1, 1, 1, 0, 0] # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dados em conjuto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test, = train_test_split(X, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo SVM
model = SVC(kernel='linear') # Criar o modelo SVM com kernel linear
model.fit(X_train, y_train) # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(X_test) # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions) # Calcular a precisão do modelo
print("Accuracy:", accuracy)
~~~
![Arvore](https://github.com/motielk/Regressao-Classificacao-BI/assets/49123696/f3a2407b-57a5-42f0-8d1f-b157ffa980e9)


## Modelo de Regressão
Tecnicas de Regressão
<ul>
  <li>Regressão Linear Simples</li>
  <li>Regressão Linear Múltipla</li>
  <li>Regressão Logística</li>
  <li>Regressão Polinomial</li>
  <li>Métodos de Regressão Não Linear</li>
</ul>

### Regressão Linear Simples

~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados de horas de estudo e notas no exame
horas_estudo = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).reshape(-1, 1)
notas_exame = np.array([65, 70, 75, 80, 85, 90, 95, 100, 105, 110])

# Criar um modelo de regressão linear
modelo = LinearRegression()

# Treinar o modelo
modelo.fit(horas_estudo, notas_exame)

# Coeficientes do modelo
coef_angular = modelo.coef_[0]
coef_linear = modelo.intercept_

# Plotar os dados e a reta de regressão
plt.scatter(horas_estudo, notas_exame, color='blue')
plt.plot(horas_estudo, modelo.predict(horas_estudo), color='red')
plt.title('Regressão Linear Simples')
plt.xlabel('Horas de Estudo')
plt.ylabel('Nota no Exame')
plt.show()

# Fazer previsões com o modelo
horas_estudo_novo = np.array([[8]])  # Horas de estudo do novo aluno
nota_prevista = modelo.predict(horas_estudo_novo)
print("Nota prevista para {} horas de estudo: {:.2f}".format(horas_estudo_novo[0][0], nota_prevista[0]))
~~~
![Regressão linear](https://github.com/motielk/Regressao-Classificacao-BI/assets/49123696/e6440091-5d84-43b3-a669-289af8638c5c)

### Regressão Linear Multipla

~~~python
# Regressão Linear Múltipla
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Dados de horas de estudo, tempo de sono e notas do exame
horas_estudo = np.array([2,3,4,5,6,7,8,9,10,11]).reshape(-1,1)
tempo_sono = np.array([7,6,5,6,7,8,9,8,7,6]).reshape(-1,1)
notas_exames = np.array([65,70,75,80,85,90,95,100,105,110])

# Criar um modelo de regressão linear
modelo = LinearRegression()

# Combinação de horas de estudos e tempo de sono com variáveis independentes
x = np.concatenate((horas_estudo,tempo_sono), axis=1)

# Treinar o modelo
modelo.fit(x, notas_exames)

# Coeficientes do modelo
coef_angular = modelo.coef_
coef_linear = modelo.intercept_

# Plotar os dados em 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(horas_estudo, tempo_sono, notas_exames, color='blue')

# Prever notas para o intervalo de horas de estudos e tempo de sono
x_test = np.array([[x,y] for x in range(2, 12) for y in range(5,10)])
nota_previstas = modelo.predict(x_test)

# Plotar o plano de regressão
x_surf, y_surf = np.meshgrid(range(2,12), range(5,10))
exog = np.column_stack((x_surf.flatten(), y_surf.flatten()))
nota_previstas = modelo.predict(exog)
ax.plot_surface(x_surf, y_surf, nota_previstas.reshape(x_surf.shape), color='red', alpha=0.5)

ax.set_xlabel('Horas de Estudo')
ax.set_ylabel('Tempo de Sono')
ax.set_zlabel('Notas do Exame')

plt.show()
~~~
![RegressãoLinearMultipla](https://github.com/motielk/Regressao-Classificacao-BI/assets/49123696/4c0e15ca-958d-416f-9761-8852bfced933)

### Regressão Logistica

~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Passo 1: Carregar o conjunto de dados iris
iris = load_iris()
x = iris.data[:, :2] # Apenas as duas primeiras características para visualização
y = iris.target

# Passo 2: Dividir o conjunto de dados em conjunto de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=42)

# Passo 3: Pré-processamento (padronização)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Passo 4: Criar e treinar modelo de regressãologística
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# Passo 5: Fazer previsões no conjunto de teste
y_pred = model.predict(x_test_scaled)

# Passo 6: Avaliar o desempenho do modelo
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test,y_pred))

# Passo 7: Visiuazlização dos resultados
plt.figure(figsize=(10,6))

# Plotar os pontos de dados de treinamento
plt.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=y_train, cmap='viridis', edgecolors='k', label='Treinamento')

# Plotar os pontos de dados de teste
plt.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=y_test, cmap='viridis', marker='x', s=100, label='Teste')

# Plotar as regiões de decisão
x_min, x_max = x_train_scaled[:, 0].min() -1, x_train_scaled[:,0].max() + 1
y_min, y_max = x_train_scaled[:, 0].min() -1, x_train_scaled[:,0].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.contourf(xx, yy, z, alpha=0.3, cmap='viridis')

plt.xlabel('Comprimento da Sépala Padronizado')
plt.ylabel('Largura da Sépala Padronizado')
plt.title('Regressão Logística para Classificaçãode Espécies Iris')
plt.legend()
plt.show()
~~~
![RegressãoLinearLogistica](https://github.com/motielk/Regressao-Classificacao-BI/assets/49123696/47c5687d-6d4c-4ad2-930c-9530bb54c3d0)

### Regressão Polinomial

~~~python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Gerar dados sintéticos
np.random.seed(0)
X = 2 * np.random.rand(100, 1) - 1  # Variáveis independentes entre -1 e 1
y = 3 * X**2 + 0.5 * X + 2 + np.random.randn(100, 1)  # Relação quadrática com ruído

# Plotar os dados
plt.scatter(X, y, color='blue', label='Dados')

# Ajustar uma regressão polinomial de grau 2 aos dados
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Plotar a curva ajustada
X_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot = lin_reg.predict(X_plot_poly)
plt.plot(X_plot, y_plot, color='red', label='Regressão Polinomial (grau 2)')

# Avaliar o modelo
y_pred = lin_reg.predict(X_poly)
mse = mean_squared_error(y, y_pred)
print("Erro médio quadrático:", mse)

plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.title('Regressão Polinomial de Grau 2')
plt.legend()
plt.show()
~~~
![Regressão polinomial](https://github.com/motielk/Regressao-Classificacao-BI/assets/49123696/864078a1-1d2a-405b-a0ac-389cd6b2148f)

### Regressão Não Linear

~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# Função exponencial para ajustar aos dados
def modelo_exponencial(x, a, b):
    return a * np.exp(b * x)

# Gerar dados sintéticos
np.random.seed(0)
X = np.linspace(0, 5, 100)  # Variável independente
y = 2.5 * np.exp(0.5 * X) + np.random.normal(0, 0.5, 100)  # Relação exponencial com ruído

# Ajustar o modelo aos dados usando curve_fit
params, _ = curve_fit(modelo_exponencial, X, y)

# Plotar os dados
plt.scatter(X, y, color='blue', label='Dados')

# Plotar a curva ajustada
plt.plot(X, modelo_exponencial(X, *params), color='red', label='Regressão Exponencial')

plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.title('Regressão Não Linear Exponencial')
plt.legend()
~~~
![RegressãoNãoLinear](https://github.com/motielk/Regressao-Classificacao-BI/assets/49123696/a36dcfc5-6e3d-4b43-8fa7-5d8fa9dbff09)




