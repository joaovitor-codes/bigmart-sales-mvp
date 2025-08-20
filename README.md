# Kaggle Chatbot MVP

MVP educacional para responder perguntas sobre datasets do Kaggle via Streamlit, com treino de modelos simples (regressão logística ou linear) e documentação organizada.

## Documentação

A documentação completa está na pasta [`docs/`](./docs):

- [PMC](./docs/pmc.md)
- [Arquitetura](./docs/architecture.md)
- [Modelagem de Dados](./docs/data_model.md)
- [Governança LGPD/DAMA](./docs/governance_lgpd.md)
- [Testes](./docs/testing.md)
- [Deploy](./docs/deployment.md)

## Como rodar o projeto

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main_app.py
```