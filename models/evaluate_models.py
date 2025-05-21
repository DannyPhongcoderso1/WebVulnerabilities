from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import json
import pickle
import pandas as pd
from config_module.config import JSON_FILE
def print_metrics(train_type, y_true, y_pred, y_proba, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    test_error = 1 - accuracy
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    logloss = log_loss(y_true, y_proba)
    
    print(f"\nValidation on {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Test Error (1-Accuracy): {test_error:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")  
    
    
    return {
        'Train Type': train_type,
        'Dataset': dataset_name,
        'Accuracy': accuracy,
        'Test Error': test_error,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Log Loss': logloss        
    }
    
    
def test(model: RandomForestClassifier, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    test_error = 1 - accuracy
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
   
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Test Error (1-Accuracy): {test_error:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")  
    
def save_model_and_metrics(model, y_val, y_val_pred, y_val_proba,
                           y_test, y_test_pred, y_test_proba,
                           training_time, prediction_time,
                           config_path=JSON_FILE):

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_name = config.get("model_name", "UnknownModel")
    model_path = config.get("model_path", "model.pkl")
    results_file = config.get("result_csv", "model_results.csv")

 
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f" Mô hình đã được lưu tại: {model_path}")


    val_metrics = print_metrics('ML', y_val, y_val_pred, y_val_proba, "tập val")
    test_metrics = print_metrics('ML', y_test, y_test_pred, y_test_proba, "tập test")
    test_metrics['Training Time'] = training_time
    test_metrics['Prediction Time'] = prediction_time


    results_df = pd.DataFrame([val_metrics, test_metrics])
    results_df['Model'] = model_name

    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        results_df = pd.concat([existing_results, results_df], ignore_index=True)

    results_df.to_csv(results_file, index=False)
    print(f" Kết quả đã được lưu vào: {results_file}")
    
if __name__ == "__main__":
    print(JSON_FILE)