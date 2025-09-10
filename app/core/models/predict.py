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
    
    # --- INÍCIO DA MODIFICAÇÃO ---
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mean_sales = float(y_test.mean())
    
    # Calcula o erro percentual (NRMSE)
    if mean_sales > 0:
        error_percent = (rmse / mean_sales) * 100
    else:
        error_percent = 0.0

    # Calcula a pontuação de acerto como o inverso do erro
    accuracy_score = 100 - error_percent

    # Retorna um dicionário com nomes mais descritivos
    return {
        "1. Erro Absoluto (RMSE)": f"R$ {rmse:,.2f}",
        "2. Venda Média (no teste)": f"R$ {mean_sales:,.2f}",
        "3. Erro Percentual (NRMSE)": f"{error_percent:.2f}%",
        "4. Pontuação de Acerto": f"{accuracy_score:.2f}%"
    }
    # --- FIM DA MODIFICAÇÃO ---