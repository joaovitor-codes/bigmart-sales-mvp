# Project Model Canvas — Kaggle Chatbot MVP (Exemplo Titanic)

## Contexto
O naufrágio do Titanic em 1912 resultou em uma das bases de dados mais conhecidas em ciência de dados.  
O dataset contém informações sobre passageiros (idade, sexo, classe, tarifa, etc.).  
O objetivo educacional é usar esse conjunto como exemplo para treinar modelos simples e construir um chatbot interativo.

---

## Problema a ser Respondido
Como variáveis socioeconômicas e demográficas influenciaram as chances de sobrevivência dos passageiros?  
Podemos prever a sobrevivência com base nesses dados?

---

## Pergunta Norteadora
- Quais características mais impactaram na sobrevivência (idade, sexo, classe, tarifa)?  
- É possível treinar um modelo de aprendizado de máquina simples que faça boas previsões?  

---

## Solução Proposta
Desenvolver um **chatbot educacional em Streamlit** que:  
1. Permita upload do arquivo `train.csv` do Titanic.  
2. Treine modelos de:
   - Regressão logística (classificação da sobrevivência).  
   - Regressão linear (predição da tarifa).  
3. Mostre métricas de avaliação (acurácia, f1-score, RMSE).  
4. Explique a importância das variáveis por meio de coeficientes e odds ratios.  
5. Responda perguntas do usuário via chatbot regrado.  

---

## Desenho de Arquitetura
O sistema será estruturado em camadas:  

- **Interface (app/):** Streamlit como front-end para upload, treino e perguntas.  
- **Core (core/):** módulos para dados, features, modelos, explicabilidade e chatbot.  
- **Dados (data/):** pastas para armazenar arquivos brutos, tratados e modelos treinados.  
- **Documentação (docs/):** PMC, arquitetura, governança e testes.  

---

## Resultados Esperados
- Modelo de classificação com acurácia próxima de **75–80%**.  
- Relatório de métricas e importâncias de variáveis.  
- Deploy em **Streamlit Cloud** com documentação completa no GitHub.  

---

## Observação Didática
O **PMC é o mapa inicial do projeto**, ligando contexto, problema e solução a uma implementação prática.  
Ele permite que o grupo alinhe objetivos antes de programar e serve como documento pedagógico para conectar gestão de projetos com ciência de dados.
