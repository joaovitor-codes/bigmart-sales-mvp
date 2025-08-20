import streamlit as st
import pandas as pd
import numpy as np

# (Opcional) habilite o heatmap: pip install matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from core.data.io import read_csv_smart
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_classifier, train_regressor
from core.models.predict import evaluate_classifier, evaluate_regressor
from core.explain.coefficients import extract_logit_importances, extract_linear_importances
from core.chatbot.rules import answer_from_metrics

st.set_page_config(page_title="Chatbot Kaggle MVP", layout="wide")

# --------------------------------------------------------------------------------------
# Estado inicial
# --------------------------------------------------------------------------------------
if "chat_messages" not in st.session_state:
    # histórico do chat para a aba Chat
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Olá! Envie um CSV, treine o modelo e depois me pergunte sobre métricas ou variáveis importantes. 🙂"}
    ]

# Para o chat usar contexto do último treino
for key in ["last_task", "last_metrics", "last_importances"]:
    st.session_state.setdefault(key, None)

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.title("🧪 Kaggle Chatbot MVP — Tema Titanic")

with st.sidebar:
    st.header("Configurações")
    task = st.selectbox("Tarefa", ["Classificação (Survived)", "Regressão (Fare)"])
    test_size = st.slider("Tamanho do teste", 0.1, 0.4, 0.2, 0.05)
    uploaded = st.file_uploader("Envie o CSV do Titanic (train.csv)", type=["csv"])

# --------------------------------------------------------------------------------------
# Abas: Treino & Métricas | Chat
# --------------------------------------------------------------------------------------
tab_train, tab_chat = st.tabs(["📊 Treino & Métricas", "💬 Chat"])

with tab_train:
    question_train = st.text_input("Pergunte algo (rápido) aqui durante o treino (opcional):", placeholder="Ex.: Quais variáveis mais importam?")

    if uploaded:
        df = read_csv_smart(uploaded)
        st.write("Prévia dos dados", df.head())

        # Remover colunas textuais/IDs do Titanic se existirem
        drop_cols = [c for c in ["PassengerId","Name","Ticket","Cabin"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        if task.startswith("Classificação"):
            target = "Survived"
            if target not in df.columns:
                st.error(f"Coluna alvo '{target}' não encontrada no CSV.")
                st.stop()
            y = df[target]
            X = df.drop(columns=[target])

            pre = make_preprocess_pipeline(X)
            model, X_test, y_test = train_classifier(X, y, pre, test_size=test_size)

            # Métricas e matriz de confusão
            metrics, cm = evaluate_classifier(model, X_test, y_test)
            st.subheader("📈 Métricas (Classificação)")
            st.json(metrics)

            # Matriz de confusão: tabela rotulada
            st.subheader("🧮 Matriz de Confusão")
            cm_arr = np.array(cm)
            # Tente inferir rótulos 0/1; ajuste se precisar
            idx = ["Verdadeiro 0", "Verdadeiro 1"][:cm_arr.shape[0]]
            cols = ["Predito 0", "Predito 1"][:cm_arr.shape[1]]
            df_cm = pd.DataFrame(cm_arr, index=idx, columns=cols)
            st.dataframe(df_cm, use_container_width=True)

            # Heatmap (opcional)
            if HAS_MPL:
                fig, ax = plt.subplots()
                im = ax.imshow(df_cm.values, cmap="Blues")
                ax.set_xticks(range(df_cm.shape[1])); ax.set_xticklabels(cols)
                ax.set_yticks(range(df_cm.shape[0])); ax.set_yticklabels(idx)
                ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
                for i in range(df_cm.shape[0]):
                    for j in range(df_cm.shape[1]):
                        ax.text(j, i, df_cm.values[i, j], ha="center", va="center")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig, use_container_width=True)

            # Importâncias
            importances = extract_logit_importances(model, X.columns, pre)
            st.subheader("🔎 Importâncias (Logistic Coef / Odds Ratio)")
            st.dataframe(importances.head(20), use_container_width=True)

            # Salvar para o chat
            st.session_state.last_task = task
            st.session_state.last_metrics = metrics
            st.session_state.last_importances = importances

            # Resposta rápida nesta aba (se perguntado aqui)
            if question_train:
                ans = answer_from_metrics(question_train, task, metrics, importances)
                st.info(ans)
                # Também empurrar para histórico do chat
                st.session_state.chat_messages.append({"role": "user", "content": question_train})
                st.session_state.chat_messages.append({"role": "assistant", "content": ans})

        else:
            # Regressão
            target = "Fare"
            if target not in df.columns:
                st.error(f"Coluna alvo '{target}' não encontrada no CSV.")
                st.stop()
            y = df[target]
            X = df.drop(columns=[target])

            pre = make_preprocess_pipeline(X)
            model, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)

            # Métricas de regressão
            metrics = evaluate_regressor(model, X_test, y_test)
            st.subheader("📈 Métricas (Regressão)")
            st.json(metrics)

            # Importâncias
            importances = extract_linear_importances(model, X.columns, pre)
            st.subheader("🔎 Importâncias (Coeficientes normalizados)")
            st.dataframe(importances.head(20), use_container_width=True)

            # Salvar para o chat
            st.session_state.last_task = task
            st.session_state.last_metrics = metrics
            st.session_state.last_importances = importances

            if question_train:
                ans = answer_from_metrics(question_train, task, metrics, importances)
                st.info(ans)
                st.session_state.chat_messages.append({"role": "user", "content": question_train})
                st.session_state.chat_messages.append({"role": "assistant", "content": ans})

    else:
        st.info("⬆️ Envie um CSV do Titanic (train.csv) na barra lateral para começar.")

with tab_chat:
    st.caption("Converse com o assistente sobre as métricas e importâncias do último treino.")
    # Render histórico
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Entrada de chat ao final da página (UX melhor; rola automático)
    prompt = st.chat_input("Faça sua pergunta (ex.: Quais variáveis mais importam?)")
    if prompt:
        # Mostrar a pergunta
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # Responder usando contexto do último treino (se existir)
        task_ctx = st.session_state.get("last_task")
        metrics_ctx = st.session_state.get("last_metrics")
        importances_ctx = st.session_state.get("last_importances")

        if task_ctx and metrics_ctx is not None and importances_ctx is not None:
            ans = answer_from_metrics(prompt, task_ctx, metrics_ctx, importances_ctx)
        else:
            ans = "Ainda não há um modelo treinado nesta sessão. Vá em **📊 Treino & Métricas**, envie o CSV e treine o modelo primeiro."

        st.session_state.chat_messages.append({"role": "assistant", "content": ans})

        # Render imediata da última troca (opcional; o Streamlit costuma rerodar)
        with st.chat_message("assistant"):
            st.markdown(ans)
