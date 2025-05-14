# Importação de bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Carregando o dataset
base_vinhos = pd.read_csv('winequality-red.csv', sep=';')
print(base_vinhos.head())  # Visualizar as primeiras linhas

# Descrição geral dos dados
print(base_vinhos.describe())
print(base_vinhos.info())
print(base_vinhos.isnull().sum())  # Verificar valores ausentes

# Visualização de distribuições
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=base_vinhos)
plt.title('Distribuição das Notas de Qualidade do Vinho')
plt.xlabel('Qualidade')
plt.ylabel('Contagem')
plt.show()

# Histograma de algumas variáveis importantes
base_vinhos.hist(bins=15, figsize=(15, 10), layout=(4, 3))
plt.tight_layout()
plt.show()


# Separar variáveis independentes (X) e dependente (y)
x_vinhos = base_vinhos.drop('quality', axis=1).values  # Previsores
y_vinhos = base_vinhos['quality'].values  # Classe

# Escalonamento dos dados
scaler_vinhos = StandardScaler()
x_vinhos = scaler_vinhos.fit_transform(x_vinhos)
print(x_vinhos[0])

# Divisão entre treino e teste
x_vinhos_treino, x_vinhos_teste, y_vinhos_treino, y_vinhos_teste = train_test_split(
    x_vinhos, y_vinhos, test_size=0.2, random_state=42
)
print(x_vinhos_treino.shape, x_vinhos_teste.shape)

# Salvando os dados processados
with open('vinhos.pkl', mode='wb') as f:
    pickle.dump((x_vinhos_treino, x_vinhos_teste, y_vinhos_treino, y_vinhos_teste), f)


def carregar_dados_processados():
    with open('vinhos.pkl', 'rb') as f:
        return pickle.load(f)  