# Data Model: BigMart Sales

Este documento descreve a modelagem de dados em três camadas: System of Record (SOR), System of Truth (SOT), e Specification (SPEC).

## 1. System of Record (SOR)
**Tabela:** `sor_bigmart_sales`

Representa os dados brutos, exatamente como chegam do arquivo `Train.csv`. É a primeira camada de armazenamento, garantindo que tenhamos uma cópia fiel dos dados originais.

- **Propósito:** Ingestão e arquivamento dos dados brutos.
- **Estrutura:** As colunas e tipos de dados são uma correspondência direta do CSV. Nenhuma limpeza ou transformação é aplicada aqui.

| Coluna | Tipo de Dado (SQL) | Descrição |
|---|---|---|
| Item_Identifier | TEXT | ID único do produto. |
| Item_Weight | REAL | Peso do produto. |
| Item_Fat_Content | TEXT | Teor de gordura do produto. |
| Item_Visibility | REAL | % da área de exibição total dedicada ao produto. |
| Item_Type | TEXT | Categoria do produto. |
| Item_MRP | REAL | Preço máximo de varejo do produto. |
| Outlet_Identifier | TEXT | ID único da loja. |
| Outlet_Establishment_Year | INTEGER | Ano de fundação da loja. |
| Outlet_Size | TEXT | Tamanho da loja (Pequeno, Médio, Grande). |
| Outlet_Location_Type | TEXT | Tipo de cidade onde a loja está localizada (Tier 1/2/3). |
| Outlet_Type | TEXT | Tipo de loja (ex: Supermercado, Mercearia). |
| Item_Outlet_Sales | REAL | Vendas do produto na loja específica (variável alvo). |

---

## 2. System of Truth (SOT)
**Tabela:** `sot_bigmart_sales`

Esta camada representa a "versão única da verdade". Os dados da SOR são limpos, padronizados e enriquecidos. É a fonte confiável para análises e para a criação de tabelas de features.

- **Propósito:** Fornecer dados limpos e consistentes para a organização.
- **Transformações Aplicadas:**
  - Tratamento de valores nulos (ex: `Item_Weight` preenchido com a média).
  - Padronização de valores categóricos (ex: 'low fat' e 'LF' unificados para 'Low Fat').
  - Remoção de colunas não utilizadas para modelagem (ex: `Item_Identifier`).

| Coluna | Tipo de Dado (SQL) | Descrição |
|---|---|---|
| Item_Weight | REAL | Peso do produto, com nulos preenchidos. |
| Item_Fat_Content | TEXT | Teor de gordura, com valores padronizados. |
| Item_Visibility | REAL | Visibilidade do item. |
| Item_Type | TEXT | Categoria do produto. |
| Item_MRP | REAL | Preço do produto. |
| Outlet_Identifier | TEXT | ID da loja. |
| Outlet_Establishment_Year | INTEGER | Ano de fundação da loja. |
| Outlet_Size | TEXT | Tamanho da loja, com nulos preenchidos. |
| Outlet_Location_Type | TEXT | Tipo de localização da loja. |
| Outlet_Type | TEXT | Tipo de loja. |
| Item_Outlet_Sales | REAL | Variável alvo (vendas). |

---

## 3. Specification (SPEC)
**Tabela:** `spec_bigmart_sales_train`

Esta é a tabela final, pronta para ser consumida pelo modelo de machine learning. Contém as features (variáveis independentes) e a variável alvo.

- **Propósito:** Fornecer um conjunto de dados de treino limpo e pronto para a modelagem.
- **Estrutura:** É essencialmente uma cópia ou uma visão da SOT, garantindo que o modelo treine com dados da mais alta qualidade. A separação entre SOT e SPEC permite que, no futuro, possamos criar outras tabelas SPEC para diferentes modelos (ex: `spec_churn_features`) a partir da mesma SOT.

| Coluna | Tipo de Dado (SQL) | Descrição |
|---|---|---|
| ... | ... | Todas as colunas da `sot_bigmart_sales`. |
| Item_Outlet_Sales | REAL | Variável alvo. |