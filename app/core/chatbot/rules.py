def answer_from_metrics(question: str, task: str, metrics_df_or_dict, importances_df):
    """
    Responde a perguntas do usuário com base nas métricas e importâncias do modelo.
    """
    q = (question or "").lower()

    if "importan" in q or "importân" in q or "variáve" in q or "features" in q:
        if importances_df is not None and not importances_df.empty:
            top = importances_df.head(5)[["feature"]].to_dict("records")
            top_str = ", ".join([t["feature"] for t in top])
            return f"As variáveis mais influentes para o modelo de {task} são: {top_str}. (Baseado nos coeficientes do modelo)"
        else:
            return "Ainda não tenho dados de importância das variáveis para mostrar."

    if "métric" in q or "score" in q or "acur" in q or "rmse" in q:
        return f"As métricas do modelo de {task} são: {metrics_df_or_dict}"

    if "como foi treinado" in q or "pipeline" in q:
        return "O pipeline aplica imputação de dados faltantes, one-hot encoding para variáveis categóricas e padronização (scaling). Depois, treina um modelo de Regressão Linear."

    if "privacid" in q or "lgpd" in q:
        return "Para este projeto, evitamos dados sensíveis. Em um ambiente de produção, garantiríamos consentimento expresso, minimização de dados e auditoria."

    return "Desculpe, não entendi. Você pode perguntar sobre 'variáveis importantes' ou 'métricas'."