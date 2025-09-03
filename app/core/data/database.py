import sqlite3
import pandas as pd
import os

# Pega o caminho absoluto do diretório onde este arquivo (database.py) está.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SQL_DIR = os.path.join(CURRENT_DIR, "sql")
APP_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DB_NAME = os.path.join(APP_DIR, "bigmart.db")

def connect_db():
    """Cria uma conexão com o banco de dados SQLite."""
    return sqlite3.connect(DB_NAME)

def execute_sql_from_file(filepath):
    """Lê um arquivo .sql e executa os comandos."""
    conn = connect_db()
    cursor = conn.cursor()
    with open(filepath, 'r') as f:
        sql_script = f.read()
    cursor.executescript(sql_script)
    conn.commit()
    conn.close()

def create_database_and_tables():
    """Cria o banco de dados e todas as tabelas a partir dos arquivos .sql."""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    
    sql_files = [
        os.path.join(SQL_DIR, "sor_bigmart_sales.sql"),
        os.path.join(SQL_DIR, "sot_bigmart_sales.sql"),
        os.path.join(SQL_DIR, "spec_bigmart_sales_train.sql"),
        os.path.join(SQL_DIR, "spec_bigmart_sales_predict.sql") # <-- ADICIONADO
    ]
    for filepath in sql_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo SQL não encontrado: {filepath}")
        execute_sql_from_file(filepath)
    print("Banco de dados e tabelas criados com sucesso.")

def insert_csv_to_sor(df):
    """Insere os dados de um DataFrame na tabela SOR."""
    conn = connect_db()
    # A tabela SOR é genérica, então apenas inserimos os dados de treino nela
    df_train = df[df['Item_Outlet_Sales'].notna()]
    df_train.to_sql("sor_bigmart_sales", conn, if_exists="replace", index=False)
    conn.close()
    print("Dados de treino inseridos na tabela SOR.")

def run_etl_sor_to_sot():
    """Executa a transformação de SOR para SOT para os dados de treino."""
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM sor_bigmart_sales", conn)

    # Lógica de Transformação
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
    
    # Inserir na SOT (sem identifiers)
    df_sot = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'], errors='ignore')
    df_sot.to_sql("sot_bigmart_sales", conn, if_exists="replace", index=False)
    conn.close()
    print("ETL de SOR para SOT (treino) concluído.")

def run_etl_sot_to_spec_train():
    """Copia dados da SOT para a SPEC de treino."""
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM sot_bigmart_sales", conn)
    df.to_sql("spec_bigmart_sales_train", conn, if_exists="replace", index=False)
    conn.close()
    print("ETL de SOT para SPEC (treino) concluído.")

def run_etl_for_test_data(df_test):
    """Executa o ETL para os dados de teste e salva na SPEC de previsão."""
    conn = connect_db()
    
    # Aplica as mesmas transformações dos dados de treino
    df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean(), inplace=True)
    df_test['Outlet_Size'].fillna(df_test['Outlet_Size'].mode()[0], inplace=True)
    df_test['Item_Fat_Content'] = df_test['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
    
    # Mantém os identificadores para o resultado final
    df_spec = df_test[['Item_Identifier', 'Outlet_Identifier'] + [col for col in df_test.columns if col not in ['Item_Identifier', 'Outlet_Identifier']]]
    
    df_spec.to_sql("spec_bigmart_sales_predict", conn, if_exists="replace", index=False)
    conn.close()
    print("ETL para dados de teste concluído e salvo na SPEC (previsão).")

def load_data(table_name: str):
    """Carrega dados de qualquer tabela especificada."""
    conn = connect_db()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def drop_database():
    """Remove o arquivo do banco de dados."""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"Banco de dados '{DB_NAME}' removido.")