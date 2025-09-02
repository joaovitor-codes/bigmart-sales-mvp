create database bigmart_test;

use bigmart_test;

-- Tabela de Dimensão: Item
-- Armazena as características únicas de cada item.
CREATE TABLE dim_item (
    Item_Identifier VARCHAR(16) PRIMARY KEY NOT NULL,
    Item_Weight NUMERIC(10, 2),
    Item_Fat_Content VARCHAR(16) NOT NULL,
    Item_Visibility NUMERIC(10, 5) NOT NULL,
    Item_Type VARCHAR(32) NOT NULL,
    Item_MRP NUMERIC(10, 2) NOT NULL
);

-- Tabela de Dimensão: Outlet
-- Armazena as características únicas de cada loja.
CREATE TABLE dim_outlet (
    Outlet_Identifier VARCHAR(16) PRIMARY KEY NOT NULL,
    Outlet_Establishment_Year INTEGER NOT NULL,
    Outlet_Size VARCHAR(16),
    Outlet_Location_Type VARCHAR(16) NOT NULL,
    Outlet_Type VARCHAR(32) NOT NULL
);

-- Tabela de Fato: Vendas (apenas para teste)
-- Serve para ligar itens e lojas para a previsão.
CREATE TABLE ft_vendas_teste (
    Item_Identifier VARCHAR(16) NOT NULL,
    Outlet_Identifier VARCHAR(16) NOT NULL,
    PRIMARY KEY (Item_Identifier, Outlet_Identifier),
    FOREIGN KEY (Item_Identifier) REFERENCES dim_item(Item_Identifier),
    FOREIGN KEY (Outlet_Identifier) REFERENCES dim_outlet(Outlet_Identifier)
);