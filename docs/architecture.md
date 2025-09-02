# Arquitetura da Aplicação de Análise de Vendas

Este documento descreve a arquitetura e o fluxo de dados da aplicação Streamlit para análise e previsão de vendas.

## Fluxo de Dados e Processamento

O fluxo foi desenhado para ser modular e robusto, separando a ingestão, o tratamento e a modelagem dos dados.

1.  **Upload do Usuário (UI):**
    - O usuário acessa a aplicação Streamlit e faz o upload de um arquivo `Train.csv` através da interface.

2.  **Criação do Banco de Dados (Backend):**
    - Ao clicar em "Executar Pipeline", a aplicação cria um banco de dados `SQLite` (`bigmart.db`) do zero.
    - As tabelas `sor_bigmart_sales`, `sot_bigmart_sales` e `spec_bigmart_sales_train` são criadas executando os scripts localizados em `core/data/sql/`.

3.  **Pipeline ETL (Extract, Transform, Load):**
    - **E (Extract):** O conteúdo do `Train.csv` é lido em um DataFrame pandas.
    - **T (Transform) & L (Load):**
        - **SOR:** O DataFrame bruto é inserido diretamente na tabela `sor_bigmart_sales`.
        - **SOT:** Os dados da SOR são lidos, e um processo de limpeza e transformação é aplicado (tratamento de nulos, padronização de categorias). O resultado limpo é salvo na `sot_bigmart_sales`.
        - **SPEC:** Os dados da SOT são carregados na `spec_bigmart_sales_train`, que serve como a fonte final para o treinamento do modelo.

4.  **Treinamento do Modelo (Machine Learning):**
    - Os dados são lidos da tabela `spec_bigmart_sales_train`.
    - O conjunto de dados é dividido em features (`X`) e alvo (`y`).
    - Um pipeline do Scikit-learn (`make_preprocess_pipeline`) é aplicado para imputação, one-hot encoding e scaling.
    - O modelo de Regressão Linear é treinado (`train_regressor`).

5.  **Armazenamento do Modelo (Serialização):**
    - Após o treinamento, o objeto do modelo treinado é serializado usando `pickle`.
    - O modelo é salvo como `regressor_model.pickle` dentro da pasta `model/`.

6.  **Apresentação de Resultados (UI):**
    - As métricas de avaliação (ex: RMSE) e a importância das features (coeficientes) são calculadas.
    - Os resultados são exibidos na interface do Streamlit, na aba "Resultados do Treino".

7.  **Limpeza (Cleanup):**
    - O usuário pode clicar no botão "Limpar Banco de Dados" para remover o arquivo `bigmart.db`, resetando o estado da aplicação para uma nova execução.