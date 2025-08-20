import streamlit as st
import pandas as pd
from core.data.io import read_csv_smart
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_classifier, train_regressor
from core.models.predict import evaluate_classifier, evaluate_regressor
from core.explain.coefficients import extract_logit_importances, extract_linear_importances
from core.chatbot.rules import answer_from_metrics

st.set_page_config(page_title="Chatbot Kaggle MVP", layout="wide")
st.title("🧪 Kaggle Chatbot MVP — Tema Titanic")

st.sidebar.header("Configurações")
task = st.sidebar.selectbox("Tarefa", ["Classificação (Survived)", "Regressão (Fare)"])
test_size = st.sidebar.slider("Tamanho do teste", 0.1, 0.4, 0.2, 0.05)

uploaded = st.file_uploader("Envie o CSV do Titanic (train.csv)", type=["csv"])
question = st.text_input("Pergunte algo ao chatbot (ex.: 'Quais variáveis mais importam?')")

if uploaded:
    df = read_csv_smart(uploaded)
    st.write("Prévia dos dados", df.head())

    drop_cols = [c for c in ["PassengerId","Name","Ticket","Cabin"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    if task.startswith("Classificação"):
        target = "Survived"
        y = df[target]
        X = df.drop(columns=[target])

        pre = make_preprocess_pipeline(X)
        model, X_test, y_test = train_classifier(X, y, pre, test_size=test_size)

        metrics, cm = evaluate_classifier(model, X_test, y_test)
        st.subheader("📈 Métricas (Classificação)")
        st.write(metrics)
        st.write("Matriz de Confusão", cm)

        importances = extract_logit_importances(model, X.columns, pre)
        st.subheader("🔎 Importâncias (Logistic Coef / Odds Ratio)")
        st.dataframe(importances.head(20))

        if question:
            st.info(answer_from_metrics(question, task, metrics, importances))
    else:
        target = "Fare"
        y = df[target]
        X = df.drop(columns=[target])

        pre = make_preprocess_pipeline(X)
        model, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)

        metrics = evaluate_regressor(model, X_test, y_test)
        st.subheader("📈 Métricas (Regressão)")
        st.write(metrics)

        importances = extract_linear_importances(model, X.columns, pre)
        st.subheader("🔎 Importâncias (Coeficientes normalizados)")
        st.dataframe(importances.head(20))

        if question:
            st.info(answer_from_metrics(question, task, metrics, importances))
