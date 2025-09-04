import streamlit as st
import pandas as pd
import os
import pickle

# --- Importações do Projeto ---
from core.data.io import read_csv_smart
from core.data.database import (
    create_database_and_tables,
    insert_csv_to_sor,
    run_etl_sor_to_sot,
    run_etl_sot_to_spec_train,
    run_etl_for_test_data,
    load_data,
    drop_database
)
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_regressor
from core.models.predict import evaluate_regressor
from core.explain.coefficients import extract_linear_importances
from core.chatbot.rules import answer_from_metrics

# --- Configurações da Página e Estado ---
st.set_page_config(page_title="Análise de Vendas", layout="wide")

# (O resto do estado da sessão permanece o mesmo)
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "predictions_made" not in st.session_state:
    st.session_state.predictions_made = False
if "prediction_df" not in st.session_state:
    st.session_state.prediction_df = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "Olá! Treine um modelo ou carregue um modelo salvo para começar."}]
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "importances" not in st.session_state:
    st.session_state.importances = None

# --- Diretórios ---
MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "regressor_model.pickle")

# --- Funções Auxiliares ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Título e Sidebar ---
st.title("🧪 Pipeline de Previsão de Vendas")

with st.sidebar:
    st.header("1. Upload dos Dados")
    uploaded_files = st.file_uploader(
        "Envie 'Train.csv' (para treino) e/ou 'Test.csv' (para previsão)",
        type=["csv"],
        accept_multiple_files=True
    )
    
    st.header("2. Ações do Pipeline")
    
    # --- AÇÃO 1: Treinar um novo modelo ---
    st.subheader("Treinar Novo Modelo")
    test_size = st.slider("Tamanho do conjunto de teste (validação)", 0.1, 0.4, 0.2, 0.05)
    if st.button("Executar Treinamento"):
        df_train = None
        for file in uploaded_files:
            if "train" in file.name.lower():
                df_train = read_csv_smart(file)
        
        if df_train is not None:
            with st.spinner("Treinando o modelo..."):
                # (Lógica de treino completa, como antes)
                create_database_and_tables()
                insert_csv_to_sor(df_train)
                run_etl_sor_to_sot()
                run_etl_sot_to_spec_train()
                df_spec_train = load_data("spec_bigmart_sales_train")
                target = "Item_Outlet_Sales"
                y = df_spec_train[target]
                X = df_spec_train.drop(columns=[target])
                pre = make_preprocess_pipeline(X)
                model, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)
                with open(MODEL_PATH, "wb") as f:
                    pickle.dump(model, f)
                st.session_state.metrics = evaluate_regressor(model, X_test, y_test)
                st.session_state.importances = extract_linear_importances(model, X.columns, pre)
                st.session_state.model_trained = True
                st.session_state.predictions_made = False
            st.success("Modelo treinado e salvo com sucesso!")
        else:
            st.warning("Arquivo 'Train.csv' não encontrado.")

    # --- AÇÃO 2: Usar o modelo salvo para prever ---
    st.subheader("Usar Modelo Existente")
    if st.button("Carregar Modelo e Fazer Previsões"):
        if not os.path.exists(MODEL_PATH):
            st.error("Nenhum modelo treinado foi encontrado! Por favor, execute o treinamento primeiro.")
        else:
            df_test = None
            for file in uploaded_files:
                if "test" in file.name.lower():
                    df_test = read_csv_smart(file)
            
            if df_test is not None:
                with st.spinner("Carregando modelo e fazendo previsões..."):
                    # Processa os dados de teste
                    run_etl_for_test_data(df_test)
                    df_spec_predict = load_data("spec_bigmart_sales_predict")
                    
                    # Carrega o modelo do arquivo .pickle
                    with open(MODEL_PATH, 'rb') as f:
                        model = pickle.load(f)

                    # Prepara os dados e faz a previsão
                    ids = df_spec_predict[['Item_Identifier', 'Outlet_Identifier']]
                    X_predict = df_spec_predict.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
                    predictions = model.predict(X_predict)
                    
                    # Exibe o resultado
                    result_df = ids.copy()
                    result_df['Item_Outlet_Sales'] = predictions
                    st.session_state.prediction_df = result_df
                    st.session_state.predictions_made = True
                st.success("Previsões geradas com sucesso usando o modelo salvo!")
            else:
                st.warning("Arquivo 'Test.csv' não encontrado.")
    
    # --- AÇÃO 3: Limpeza ---
    st.header("3. Manutenção")
    if st.button("Limpar Tudo"):
        drop_database()
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH) # Também remove o modelo salvo
        st.session_state.clear() # Limpa toda a sessão
        st.info("Banco de dados, modelo salvo e sessão resetados.")
        st.rerun()

# --- Abas Principais ---
# (O código das abas permanece exatamente o mesmo)
tab_train, tab_predict, tab_chat = st.tabs(["📊 Resultados do Treino", "🚀 Previsões", "💬 Chat com o Modelo"])

with tab_train:
    st.header("Métricas e Importância do Modelo")
    if not st.session_state.model_trained and st.session_state.metrics is None:
        st.info("Execute o treinamento na barra lateral para ver os resultados.")
    else:
        st.subheader("📈 Métricas (Regressão)")
        st.json(st.session_state.metrics)
        st.subheader("🔎 Importâncias (Coeficientes)")
        st.dataframe(st.session_state.importances.head(20), use_container_width=True)

with tab_predict:
    st.header("Previsões para os Dados de Teste")
    if not st.session_state.predictions_made:
        st.info("Faça uma previsão na barra lateral para ver os resultados.")
    else:
        st.dataframe(st.session_state.prediction_df)
        csv_data = convert_df_to_csv(st.session_state.prediction_df)
        st.download_button(
           label="Download das Previsões em CSV",
           data=csv_data,
           file_name='submission.csv',
           mime='text/csv',
        )

with tab_chat:
    st.header("Converse com o Assistente do Modelo")
    if not st.session_state.model_trained and st.session_state.metrics is None:
        st.info("Treine um modelo primeiro para poder conversar sobre seus resultados.")
    else:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Quais as variáveis mais importantes?"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            response = answer_from_metrics(
                question=prompt,
                task="Regressão",
                metrics_df_or_dict=st.session_state.metrics,
                importances_df=st.session_state.importances
            )
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()