# Projeto: Análise e Pré-Processamento de Dados com o Dataset de Vinhos

Este projeto é parte da minha trilha de estudos em **Machine Learning e Inteligência Artificial com Python**, e foi desenvolvido com base no módulo de **Pré-Processamento de Dados**.

Utilizei o dataset [Wine Quality - Red Wine](https://archive.ics.uci.edu/ml/datasets/wine+quality) da UCI Machine Learning Repository,
que contém informações físico-químicas sobre diferentes amostras de vinhos tintos, junto à sua avaliação de qualidade sensorial.

## O que foi feito

- **Importação do dataset** e análise de tipos de dados e valores ausentes.
- **Análise exploratória**, incluindo:
  - Gráficos de distribuição (`countplot`, `histogram`)
  - Verificação de distribuição da variável-alvo: `quality`
- **Separação de atributos** (previsores e variável alvo).
- **Escalonamento dos dados** com `StandardScaler`, para garantir que todas as variáveis estejam na mesma escala.
- **Divisão em dados de treino e teste** utilizando `train_test_split`.
- **Serialização com Pickle** para salvar os dados prontos para uso em modelos futuros.

## Tecnologias usadas

- Python 3.13
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Plotly
- Scikit-learn
- Pickle
- Streamlit

## Visualização com Streamlit
Para tornar a análise mais acessível e interativa, foi criada uma interface com Streamlit, permitindo:

- Exibição dos dados brutos e processados
- Visualização dos principais gráficos e estatísticas
- Inspeção dos dados escalonados
- Download do dataset .pkl já pronto para ser usado em modelos de machine learning


