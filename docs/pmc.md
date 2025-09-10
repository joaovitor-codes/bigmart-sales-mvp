Project Model Canvas — Chatbot de Previsão de Vendas (BigMart)
Contexto
A previsão de vendas no varejo é um desafio crucial para a otimização de estoque, logística e estratégias de marketing. O dataset da BigMart contém informações detalhadas sobre produtos e as lojas em que são vendidos (tamanho da loja, localização, tipo de produto, preço, etc.).
O objetivo é usar este conjunto de dados para treinar um modelo de regressão e construir um chatbot interativo que explique os resultados.

Problema a ser Respondido
Como as características de um produto e da loja onde ele é vendido influenciam seu volume de vendas?
Podemos prever com um bom nível de acerto as vendas futuras de um item em uma loja específica?

Pergunta Norteadora
Quais características mais impactam nas vendas (Item_Outlet_Sales)? (Ex: Preço do item, tipo de loja, visibilidade do produto).

É possível treinar um modelo de regressão linear que forneça previsões úteis para a tomada de decisão de negócio?

Solução Proposta
Desenvolver um chatbot interativo em Streamlit que:

Permita o upload dos arquivos Train.csv e Test.csv da BigMart.

Treine um modelo de Regressão Linear para prever o Item_Outlet_Sales.

Mostre métricas de avaliação claras (RMSE, Erro Percentual e Pontuação de Acerto).

Explique a importância das variáveis por meio dos coeficientes do modelo.

Responda a perguntas do usuário sobre os resultados do treino via chatbot.

Permita carregar um modelo já treinado para fazer novas previsões rapidamente.

Desenho de Arquitetura
O sistema será estruturado em camadas para garantir organização e manutenibilidade:

Interface (app/): Streamlit como front-end para upload, visualização de resultados, execução do pipeline e interação com o chatbot.

Core (core/): Módulos para pipeline de dados, pré-processamento, treino de modelos, avaliação, explicabilidade e lógica do chatbot.

Dados (data/): Pastas SQL para a criação das tabelas (SOR, SOT, SPEC).

Modelo (model/): Pasta para armazenar o modelo treinado (.pickle).

Resultados Esperados
Modelo de regressão com uma Pontuação de Acerto superior a 50%.

Geração de um arquivo de submissão (submission.csv) com as previsões para os dados de teste.

Relatório interativo de métricas e das variáveis mais importantes.

Deploy da aplicação na Streamlit Cloud com a documentação do projeto no GitHub.

Observação Didática
O PMC é o mapa inicial do projeto, ligando contexto, problema e solução a uma implementação prática.
Ele permite que o grupo alinhe objetivos antes de programar e serve como documento pedagógico para conectar gestão de projetos com ciência de dados.