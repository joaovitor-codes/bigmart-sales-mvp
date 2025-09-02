import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importar as funções dos arquivos que você criou
from core.data.io import read_csv_smart
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_regressor
from core.models.predict import evaluate_regressor
from core.explain.coefficients import extract_linear_importances
from core.chatbot.rules import answer_from_metrics

# Configurar a página do Streamlit
st.set_page_config(page_title="Análise de Vendas BigMart", layout="wide")

# --------------------------------------------------------------------------------------
# Estado inicial
# --------------------------------------------------------------------------------------
if "chat_messages" not in st.session_state:
    # Histórico do chat para a aba Chat
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Olá! Envie seu CSV, treine o modelo e depois me pergunte sobre métricas ou variáveis importantes. �"}
    ]

# Para o chat usar contexto do último treino
for key in ["last_task", "last_metrics", "last_importances"]:
    st.session_state.setdefault(key, None)

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.title("🧪 Análise de Vendas BigMart")

with st.sidebar:
    st.header("Configurações")
    st.info("Esta aplicação é para Regressão, prevendo vendas (`Item_Outlet_Sales`).")
    test_size = st.slider("Tamanho do teste", 0.1, 0.4, 0.2, 0.05)
    uploaded = st.file_uploader("Envie o arquivo Train.csv", type=["csv"])

# --------------------------------------------------------------------------------------
# Abas: Treino & Métricas | Chat
# --------------------------------------------------------------------------------------
tab_train, tab_chat = st.tabs(["📊 Treino & Métricas", "💬 Chat"])

with tab_train:
    question_train = st.text_input("Pergunte algo (rápido) aqui durante o treino (opcional):", placeholder="Ex.: Quais variáveis mais importam?")

    if uploaded:
        df = read_csv_smart(uploaded)
        st.write("Prévia dos dados", df.head())

        # Tratar o caso de o usuário enviar o arquivo errado
        target = "Item_Outlet_Sales"
        if target not in df.columns:
            st.error(f"Coluna alvo '{target}' não encontrada no CSV. Por favor, envie o 'Train.csv' correto.")
            st.stop()

        # Remover colunas irrelevantes para o modelo
        drop_cols = [c for c in ["Item_Identifier", "Outlet_Identifier"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        # Tratar valores ausentes
        df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
        df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
            'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'
        })

        # Separar features (X) e a variável alvo (y)
        y = df[target]
        X = df.drop(columns=[target])

        # Pipeline de pré-processamento
        pre = make_preprocess_pipeline(X)
        model, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)

        # Métricas de regressão
        metrics = evaluate_regressor(model, X_test, y_test)
        st.subheader("📈 Métricas (Regressão)")
        st.json(metrics)

        # Importâncias
        importances = extract_linear_importances(model, X.columns, pre)
        st.subheader("🔎 Importâncias (Coeficientes)")
        st.dataframe(importances.head(20), use_container_width=True)

        # Salvar para o chat
        st.session_state.last_task = "Regressão"
        st.session_state.last_metrics = metrics
        st.session_state.last_importances = importances

        if question_train:
            ans = answer_from_metrics(question_train, "Regressão", metrics, importances)
            st.info(ans)
            st.session_state.chat_messages.append({"role": "user", "content": question_train})
            st.session_state.chat_messages.append({"role": "assistant", "content": ans})

    else:
        st.info("⬆️ Envie o arquivo Train.csv na barra lateral para começar.")

with tab_chat:
    st.caption("Converse com o assistente sobre as métricas e importâncias do último treino.")
    # Render histórico
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Entrada de chat ao final da página
    prompt = st.chat_input("Faça sua pergunta (ex.: Quais variáveis mais importam?)")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        task_ctx = st.session_state.get("last_task")
        metrics_ctx = st.session_state.get("last_metrics")
        importances_ctx = st.session_state.get("last_importances")

        if task_ctx and metrics_ctx is not None and importances_ctx is not None:
            ans = answer_from_metrics(prompt, task_ctx, metrics_ctx, importances_ctx)
        else:
            ans = "Ainda não há um modelo treinado nesta sessão. Vá em **📊 Treino & Métricas**, envie o CSV e treine o modelo primeiro."

        st.session_state.chat_messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)