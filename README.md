
# Guia de Aula — Kaggle Chatbot MVP (Python + Streamlit)

## 1) Objetivo da aula

Montar um **MVP** de app com **Python + Streamlit** capaz de:

- **Conversar** sobre o tema escolhido (chat “guiado” pelas features);
- **Treinar** um modelo simples (regressão **logística** ou **linear**);
- **Responder perguntas** do usuário com base em **métricas** e **importâncias de variáveis**;
- Estar pronto para **deploy** (Streamlit Cloud/Render) com **docs organizadas** (PM Canvas, arquitetura, dados, LGPD, governança DAMA, testes).

---

## 2) Estrutura de repositório (padrão para todos os temas)

> Um **único repositório** com pastas “de responsabilidade” (parecendo **camadas**), mantendo a simplicidade do MVP.

```
kaggle-chatbot-mvp/
├─ app/                         # "Front" Streamlit e rotas de UI
│  ├─ pages/                    # Páginas extras do Streamlit (opcional)
│  └─ main_app.py               # App principal (Streamlit)
│
├─ core/                        # "Back" de regras do domínio (sem UI)
│  ├─ data/                     # Funções de carga/validação de dados
│  │  ├─ io.py                  # ler_csv, salvar_csv, baixar_zip (se preciso)
│  │  └─ schema.py              # checagens simples (tipos, colunas obrigatórias)
│  ├─ features/                 # Engenharia de atributos
│  │  └─ preprocess.py          # pipelines: imputar, one-hot, escala
│  ├─ models/                   # Treino e predição
│  │  ├─ train.py               # treinar_regressao, treinar_classificacao
│  │  └─ predict.py             # carregar_modelo, prever, explicar_resultado
│  ├─ explain/                  # Explicabilidade simples
│  │  └─ coefficients.py        # coeficientes, odds ratio, importância relativa
│  └─ chatbot/                  # Regras do "chat" por tema (sem LLM)
│     └─ rules.py               # respostas guiadas (FAQ + métricas/coeficientes)
│
├─ configs/
│  ├─ settings.example.toml     # variáveis de config (tema, alvo, etc)
│  └─ logging.conf              # logging básico
│
├─ data/
│  ├─ raw/                      # CSVs brutos (não comitar dados sensíveis)
│  ├─ processed/                # CSVs tratados
│  └─ models/                   # .pkl dos modelos (ok no MVP)
│
├─ notebooks/                   # Exploração inicial (EDA)
│  └─ 01_eda_titanic.ipynb
│
├─ tests/
│  ├─ test_data_io.py
│  ├─ test_preprocess.py
│  └─ test_models.py
│
├─ docs/                        # "Docusaurus-like" em Markdown
│  ├─ README.md                 # documentação portal (índice)
│  ├─ pmc.md                    # Project Model Canvas (foto + campos)
│  ├─ architecture.md           # Arquitetura de Software (com .drawio)
│  ├─ data_model.md             # DER/MER e dicionário de dados
│  ├─ governance_lgpd.md        # DAMA + LGPD aplicados
│  ├─ testing.md                # Estratégia de testes
│  ├─ deployment.md             # Deploy (Streamlit Cloud/Render)
│  └─ theme_guides/
│     ├─ titanic.md
│     ├─ ecommerce.md
│     └─ ...
│
├─ .gitignore
├─ requirements.txt
├─ runtime.txt                  # (opcional) travar versão do Python p/ deploy
└─ Makefile                     # (opcional) atalhos: make run, make test
```

**Por que assim?**

- `app/` é o “front” Streamlit; `core/` é a “lógica de negócio”; `docs/` centraliza documentação (estilo Docusaurus em Markdown), incluindo **PMC, arquitetura, dados, LGPD, governança**.
- `data/` guarda datasets e modelos (no MVP; futuramente usar storage externo).
- `tests/` dá o hábito de **testes unitários e de integração**.

---

## 3) Template de código (tema Titanic como exemplo)

### 3.1 `app/main_app.py` (UI do Streamlit)

```python
import streamlit as st
import pandas as pd
from core.data.io import read_csv_smart
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_classifier, train_regressor
from core.models.predict import evaluate_classifier, evaluate_regressor
from core.explain.coefficients import extract_logit_importances, extract_linear_importances
from core.chatbot.rules import answer_from_metrics

st.set_page_config(page_title="Chatbot Kaggle MVP", layout="wide")

st.title("🧪 Kaggle Chatbot MVP — Tema Titanic")

st.sidebar.header("Configurações")
task = st.sidebar.selectbox("Tarefa", ["Classificação (Survived)", "Regressão (Fare)"])
test_size = st.sidebar.slider("Tamanho do teste", 0.1, 0.4, 0.2, 0.05)

uploaded = st.file_uploader("Envie o CSV do Titanic (train.csv)", type=["csv"])
question = st.text_input("Pergunte algo ao chatbot (ex.: 'Quais variáveis mais importam?')")

if uploaded:
    df = read_csv_smart(uploaded)
    st.write("Prévia dos dados", df.head())

    # Definições simples de colunas
    drop_cols = [c for c in ["PassengerId","Name","Ticket","Cabin"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    if task.startswith("Classificação"):
        target = "Survived"
        y = df[target]
        X = df.drop(columns=[target])

        pre = make_preprocess_pipeline(X)
        model, X_test, y_test = train_classifier(X, y, pre, test_size=test_size)

        metrics, cm = evaluate_classifier(model, X_test, y_test)
        st.subheader("📈 Métricas (Classificação)")
        st.write(metrics)
        st.write("Matriz de Confusão", cm)

        importances = extract_logit_importances(model, X.columns, pre)
        st.subheader("🔎 Importâncias (Logistic Coef / Odds Ratio)")
        st.dataframe(importances.head(20))

        if question:
            st.info(answer_from_metrics(question, task, metrics, importances))

    else:
        target = "Fare"
        y = df[target]
        X = df.drop(columns=[target])

        pre = make_preprocess_pipeline(X)
        model, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)

        metrics = evaluate_regressor(model, X_test, y_test)
        st.subheader("📈 Métricas (Regressão)")
        st.write(metrics)

        importances = extract_linear_importances(model, X.columns, pre)
        st.subheader("🔎 Importâncias (Coeficientes normalizados)")
        st.dataframe(importances.head(20))

        if question:
            st.info(answer_from_metrics(question, task, metrics, importances))
```

---

### 3.2 `core/data/io.py`

```python
import pandas as pd

def read_csv_smart(file_or_path):
    # tenta detectar separador automaticamente
    return pd.read_csv(file_or_path, sep=None, engine="python")
```

---

### 3.3 `core/features/preprocess.py`

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def infer_cols(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def make_preprocess_pipeline(X_df):
    num_cols, cat_cols = infer_cols(X_df)

    num_pipe = SimpleImputer(strategy="median")
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    # scaler após o ColumnTransformer (para regressão/coefs estáveis)
    return Pipeline([("pre", pre), ("scaler", StandardScaler(with_mean=False))])
```

> Obs.: note o `from sklearn.pipeline import Pipeline` no topo.

---

### 3.4 `core/models/train.py`

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline

def split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)

def train_classifier(X, y, pre, test_size=0.2):
    X_train, X_test, y_train, y_test = split(X, y, test_size=test_size)
    clf = LogisticRegression(max_iter=1000)
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)
    return model, X_test, y_test

def train_regressor(X, y, pre, test_size=0.2):
    X_train, X_test, y_train, y_test = split(X, y, test_size=test_size)
    reg = LinearRegression()
    model = Pipeline([("pre", pre), ("reg", reg)])
    model.fit(X_train, y_train)
    return model, X_test, y_test
```

---

### 3.5 `core/models/predict.py`

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
import numpy as np

def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    return metrics, cm

def evaluate_regressor(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {"rmse": rmse}
```

---

### 3.6 `core/explain/coefficients.py`

```python
import numpy as np
import pandas as pd

def _feature_names_from_preprocess(pre, original_cols):
    # tenta extrair nomes após OneHot
    pre_step = pre.named_steps["pre"]
    num_cols = pre_step.transformers_[0][2]
    cat_cols = pre_step.transformers_[1][2]
    # num: nomes originais
    out = list(num_cols)
    # cat: pega categories_ do OneHot
    ohe = pre_step.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    return out + cat_feature_names

def extract_logit_importances(model_pipe, original_cols, pre):
    # modelo final: Pipeline(pre → clf)
    clf = model_pipe.named_steps["clf"]
    feature_names = _feature_names_from_preprocess(model_pipe.named_steps["pre"], original_cols)
    coefs = clf.coef_.ravel()
    odds = np.exp(coefs)
    df = pd.DataFrame({"feature": feature_names, "coef": coefs, "odds_ratio": odds})
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")

def extract_linear_importances(model_pipe, original_cols, pre):
    reg = model_pipe.named_steps["reg"]
    feature_names = _feature_names_from_preprocess(model_pipe.named_steps["pre"], original_cols)
    coefs = reg.coef_.ravel()
    df = pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)})
    return df.sort_values("abs_coef", ascending=False)
```

---

### 3.7 `core/chatbot/rules.py` (chat “regrado” sem LLM)

```python
def answer_from_metrics(question: str, task: str, metrics_df_or_dict, importances_df):
    q = (question or "").lower()

    if "importan" in q or "importân" in q or "variáve" in q or "features" in q:
        top = importances_df.head(5)[["feature"]].to_dict("records")
        top_str = ", ".join([t["feature"] for t in top])
        return f"As variáveis mais influentes são: {top_str}. (Baseado em coeficientes/odds ratio)"

    if "métric" in q or "score" in q or "acur" in q or "rmse" in q:
        return f"Métricas da tarefa {task}: {metrics_df_or_dict}"

    if "como foi treinado" in q or "pipeline" in q:
        return "O pipeline aplica imputação, one-hot e padronização; depois treina Logistic Regression (class.) ou Linear Regression (regr.)."

    if "privacid" in q or "lgpd" in q:
        return "No MVP, evitamos dados sensíveis, anonimização por padrão e não persistimos dados pessoais. Para produção: consentimento expresso, minimização e auditoria."

    return "Posso falar sobre variáveis importantes, métricas do modelo e como o pipeline funciona. Pergunte algo como 'Quais variáveis mais importam?'."
```

---

## 4) Documentação estilo “Docusaurus” (em Markdown)

### 4.1 `docs/README.md` (portal)

```markdown
# Kaggle Chatbot MVP

MVP educacional para responder perguntas sobre um tema do Kaggle via Streamlit, com treino de modelo (regressão logística ou linear) e documentação integrada (PMC, arquitetura, dados, LGPD/DAMA, testes e deploy).

## Como rodar
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main_app.py
```

## Estrutura
- `app/` UI (Streamlit)
- `core/` regras de negócio (dados, features, modelos, chatbot, explicabilidade)
- `docs/` documentação do projeto (PMC, arquitetura, dados, governança, testes, deploy)
- `data/` dados e modelos (MVP)
- `tests/` testes unit./integração

## Deploy rápido
Veja `docs/deployment.md`.
```
```

### 4.2 `docs/pmc.md` (Project Model Canvas)

- Adicione a **imagem do PMC preenchido** pelo grupo (exportada do Miro/Whimsical/Excalidraw).
- Inclua os blocos adaptados a software: **Objetivo, Justificativa, Produto, Requisitos (inclui LGPD+DAMA), Público-alvo, Equipe, Riscos, Cronograma, Custos (esforço), Recursos, Stakeholders**.

### 4.3 `docs/architecture.md` (Arquitetura de Software)

- Inclua um **.drawio** com:
  - Módulos: `app (Streamlit)` → `core (data, features, models, chatbot)` → `data/ (arquivos)`.
  - Fluxos: **Upload CSV → Pré-processar → Treinar → Métricas → Resposta do Chat**.
  - Setas e anotações de **camadas** (UI, aplicação, dados).
- Exporte uma imagem PNG e embuta:
  ```markdown
  ![Arquitetura](./images/architecture.png)
  ```

### 4.4 `docs/data_model.md` (Arquitetura/Modelagem de Dados)

- **DER simples** (ex.: Tabela “passengers” no Titanic) com campos e tipos.
- **Dicionário de dados** (nome, tipo, descrição, domínio).

### 4.5 `docs/governance_lgpd.md` (Governança DAMA + LGPD)

- **DAMA**: qualidade (validação de entrada), metadados (README do dataset), segurança (não versionar dados sensíveis), ciclo de vida (limpeza da pasta `data/`).
- **LGPD**: minimização (não coletar além do necessário), consentimento (documentar como seria), anonimização, auditoria (logs básicos).

### 4.6 `docs/testing.md`

- Testes unitários: `read_csv_smart`, `make_preprocess_pipeline`, `train_classifier/regressor`.
- Testes de integração: fluxo **“CSV → métricas”**.

### 4.7 `docs/deployment.md` (Deploy simples)

- **Streamlit Cloud**:
  1. Repo público no GitHub
  2. App: `app/main_app.py`
  3. Python 3.11 (por ex.), `requirements.txt`
  4. `data/` vazia (ou exemplo pequeno)
- **Render.com** (opcional): usar `gunicorn` + `streamlit` com *start command*.

---

## 5) `requirements.txt` sugerido

```txt
pandas
numpy
scikit-learn
streamlit
```

> Se usar gráficos, adicione `plotly`.

---

## 6) Testes (exemplos mínimos)

### `tests/test_data_io.py`

```python
from io import StringIO
from core.data.io import read_csv_smart

def test_read_csv_smart():
    csv = "A,B\n1,2\n3,4\n"
    df = read_csv_smart(StringIO(csv))
    assert df.shape == (2,2)
```

### `tests/test_preprocess.py`

```python
import pandas as pd
from core.features.preprocess import make_preprocess_pipeline

def test_make_preprocess():
    X = pd.DataFrame({"num":[1,2,3], "cat":["a","b","a"]})
    pre = make_preprocess_pipeline(X)
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == 3
```
