import streamlit as st
from data import carregar_dados_processados


st.title('Análise e Pré-Processamento de Dados - Vinho Tinto')


x_treino, x_teste, y_treino, y_teste = carregar_dados_processados()


st.subheader('Informações das Bases')
st.write(f'Base de treino: {x_treino.shape[0]} amostras')
st.write(f'Base de teste: {x_teste.shape[0]} amostras')


if st.button('Ver amostras de treino'):
    st.write(x_treino[:5])
