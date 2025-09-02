import sqlite3
import pandas as pd
import os

# --- INÍCIO DA PARTE IMPORTANTE ---
# Esta parte do código descobre o caminho completo para os seus arquivos,
# não importa de onde você execute o projeto.

# Pega o caminho absoluto do diretório onde este arquivo (database.py) está.
# Ex: C:\Users\Jhuliane\Desktop\ChatBot\bigmart_streamlit\app\core\data
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# A partir daí, ele sabe que a pasta 'sql' está logo ao lado.
# Ex: C:\Users\Jhuliane\Desktop\ChatBot\bigmart_streamlit\app\core\data\sql
SQL_DIR = os.path.join(CURRENT_DIR, "sql")

# E ele sabe onde salvar o banco de dados na pasta principal 'app'
APP_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DB_NAME = os.path.join(APP_DIR, "bigmart.db")
# --- FIM DA PARTE IMPORTANTE ---

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
    
    # A lista de arquivos agora usa o caminho completo que descobrimos acima
    sql_files = [
        os.path.join(SQL_DIR, "sor_bigmart_sales.sql"),
        os.path.join(SQL_DIR, "sot_bigmart_sales.sql"),
        os.path.join(SQL_DIR, "spec_bigmart_sales_train.sql")
    ]
    # Verifica se os arquivos realmente existem antes de tentar abri-los
    for filepath in sql_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo SQL não encontrado no caminho esperado: {filepath}")
        execute_sql_from_file(filepath)
    print("Banco de dados e tabelas criados com sucesso.")

def insert_csv_to_sor(df):
    """Insere os dados de um DataFrame na tabela SOR."""
    conn = connect_db()
    df.to_sql("sor_bigmart_sales", conn, if_exists="replace", index=False)
    conn.close()
    print("Dados inseridos na tabela SOR.")

def run_etl_sor_to_sot():
    """Executa a transformação de SOR para SOT e insere os dados."""
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM sor_bigmart_sales", conn)

    drop_cols = [c for c in ["Item_Identifier"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
    
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'
    })

    df.to_sql("sot_bigmart_sales", conn, if_exists="replace", index=False)
    conn.close()
    print("ETL de SOR para SOT concluído.")

def run_etl_sot_to_spec():
    """Copia dados da SOT para a SPEC (tabela de treino)."""
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM sot_bigmart_sales", conn)
    df.to_sql("spec_bigmart_sales_train", conn, if_exists="replace", index=False)
    conn.close()
    print("ETL de SOT para SPEC concluído.")

def load_spec_data_for_training():
    """Carrega os dados da tabela SPEC para treinar o modelo."""
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM spec_bigmart_sales_train", conn)
    conn.close()
    return df

def drop_database():
    """Remove o arquivo do banco de dados para limpeza."""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"Banco de dados '{DB_NAME}' removido com sucesso.")