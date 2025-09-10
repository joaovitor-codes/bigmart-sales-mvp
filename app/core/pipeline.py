import os
import pickle
import time
from core.data.database import (
    create_database_and_tables,
    insert_csv_to_sor,
    run_etl_sor_to_sot,
    run_etl_sot_to_spec_train,
    run_etl_for_test_data,
    load_data
)
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_regressor
from core.models.predict import evaluate_regressor
from core.explain.coefficients import extract_linear_importances

def run_training_pipeline(df_train, test_size, progress_bar, model_path):
    """
    Orquestra o pipeline completo de treinamento do modelo.
    """
    # Etapa 1: Criar DB (15%)
    create_database_and_tables()
    progress_bar.progress(15, text="Banco de dados criado.")
    time.sleep(0.5)

    # Etapa 2: Inserir dados brutos (30%)
    insert_csv_to_sor(df_train)
    progress_bar.progress(30, text="Dados brutos inseridos (SOR).")
    time.sleep(0.5)

    # Etapa 3: Limpar dados (45%)
    run_etl_sor_to_sot()
    run_etl_sot_to_spec_train()
    progress_bar.progress(45, text="Dados limpos e preparados (SOT/SPEC).")
    time.sleep(0.5)
    
    # Etapa 4: Treinar o modelo (75%)
    df_spec_train = load_data("spec_bigmart_sales_train")
    target = "Item_Outlet_Sales"
    y = df_spec_train[target]
    X = df_spec_train.drop(columns=[target])
    pre = make_preprocess_pipeline(X)
    model, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)
    progress_bar.progress(75, text="Modelo treinado com sucesso!")
    time.sleep(0.5)

    # Etapa 5: Salvar e avaliar (90%)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    metrics = evaluate_regressor(model, X_test, y_test)
    importances = extract_linear_importances(model, X.columns, pre)
    progress_bar.progress(90, text="Modelo salvo e métricas calculadas.")
    time.sleep(0.5)
    
    # Retorna as métricas e importâncias para serem salvas na sessão
    return metrics, importances

def run_prediction_pipeline(df_test, model_path):
    """
    Orquestra o pipeline de previsão usando um modelo salvo.
    """
    # Processa os dados de teste
    run_etl_for_test_data(df_test)
    df_spec_predict = load_data("spec_bigmart_sales_predict")
    
    # Carrega o modelo do arquivo .pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Prepara os dados e faz a previsão
    ids = df_spec_predict[['Item_Identifier', 'Outlet_Identifier']]
    X_predict = df_spec_predict.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
    predictions = model.predict(X_predict)
    
    # Monta o resultado
    result_df = ids.copy()
    result_df['Item_Outlet_Sales'] = predictions
    
    return result_df