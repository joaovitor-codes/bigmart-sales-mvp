import streamlit as st
import pandas as pd
import os
import pickle

# Importar as funções dos arquivos do projeto
from core.data.io import read_csv_smart
from core.data.database import (
    create_database_and_tables,
    insert_csv_to_sor,
    run_etl_sor_to_sot,
    run_etl_sot_to_spec,
    load_spec_data_for_training,
    drop_database
)
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_regressor
from core.models.predict import evaluate_regressor
from core.explain.coefficients import extract_linear_importances
from core.chatbot.rules import answer_from_metrics

# --- Configurações da Página ---
st.set_page_config(page_title="Análise de Vendas BigMart", layout="wide")

# --- Estado da Sessão ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Olá! Envie seu CSV, execute o pipeline e depois me pergunte sobre o modelo."}
    ]
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

for key in ["last_task", "last_metrics", "last_importances"]:
    st.session_state.setdefault(key, None)

# --- Diretórios ---
MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "regressor_model.pickle")

# --- Título e Sidebar ---
st.title("🧪 Análise de Vendas BigMart")

with st.sidebar:
    st.header("1. Configurações")
    st.info("Esta aplicação treina um modelo de Regressão para prever `Item_Outlet_Sales`.")
    test_size = st.slider("Tamanho do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
    
    st.header("2. Upload de Dados")
    uploaded_file = st.file_uploader("Envie o arquivo Train.csv", type=["csv"])

    st.header("3. Controle do Pipeline")
    if st.button("Executar Pipeline de Dados e Treino"):
        if uploaded_file is not None:
            with st.spinner("Executando pipeline completo..."):
                # Etapa 1: Ler o CSV
                df_raw = read_csv_smart(uploaded_file)
                
                # Etapa 2: Criar DB e tabelas
                create_database_and_tables()
                
                # Etapa 3: Inserir dados na SOR e rodar ETL
                insert_csv_to_sor(df_raw)
                run_etl_sor_to_sot()
                run_etl_sot_to_spec()
                
                # Etapa 4: Carregar dados da SPEC para treino
                df_train = load_spec_data_for_training()
                
                # Etapa 5: Treinar o modelo
                target = "Item_Outlet_Sales"
                if target in df_train.columns:
                    y = df_train[target]
                    X = df_train.drop(columns=[target])
                    
                    pre = make_preprocess_pipeline(X)
                    model, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)
                    
                    # Etapa 6: Salvar o modelo treinado
                    with open(MODEL_PATH, "wb") as f:
                        pickle.dump(model, f)
                    st.session_state.model_trained = True
                    
                    # Avaliar e salvar métricas
                    metrics = evaluate_regressor(model, X_test, y_test)
                    importances = extract_linear_importances(model, X.columns, pre)
                    
                    st.session_state.last_task = "Regressão"
                    st.session_state.last_metrics = metrics
                    st.session_state.last_importances = importances
                else:
                    st.error(f"Coluna alvo '{target}' não encontrada.")
            st.success("Pipeline executado com sucesso! Modelo treinado e salvo.")
        else:
            st.warning("Por favor, envie um arquivo CSV para começar.")

    if st.button("Limpar Banco de Dados e Resetar"):
        drop_database()
        st.session_state.model_trained = False
        st.session_state.last_metrics = None
        st.session_state.last_importances = None
        st.info("Banco de dados dropado e estado resetado.")

# --- Abas Principais ---
tab_results, tab_chat = st.tabs(["📊 Resultados do Treino", "💬 Chat com o Modelo"])

with tab_results:
    if not st.session_state.model_trained:
        st.info("⬆️ Envie o CSV e execute o pipeline na barra lateral para ver os resultados.")
    else:
        st.subheader("📈 Métricas (Regressão)")
        st.json(st.session_state.last_metrics)
        
        st.subheader("🔎 Importâncias (Coeficientes do Modelo)")
        st.dataframe(st.session_state.last_importances.head(20), use_container_width=True)
        
        if os.path.exists(MODEL_PATH):
            st.success(f"Modelo salvo com sucesso em: `{MODEL_PATH}`")

with tab_chat:
    st.caption("Converse sobre os resultados do último modelo treinado.")
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ex: Quais as variáveis mais importantes?")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        if not st.session_state.model_trained:
            ans = "O modelo ainda não foi treinado. Por favor, execute o pipeline primeiro na barra lateral."
        else:
            ans = answer_from_metrics(
                prompt,
                st.session_state.last_task,
                st.session_state.last_metrics,
                st.session_state.last_importances
            )
        
        st.session_state.chat_messages.append({"role": "assistant", "content": ans})
        st.rerun()