import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('models/train_model.py'), '..')))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
import time
from sklearn.tree import DecisionTreeClassifier


from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from models.evaluate_models import save_model_and_metrics
from xgboost import XGBClassifier

from config_module.config import RANDOM_STATE, N_JOBS, VERBOSE, N_ESTIMATORS, GRID_SEARCH_N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT, CV, XGBOOST_SCALE_POSITIVE_WEIGHT, XGBOOST_COLSAMPLE_BYTREE, OBJECTIVE,KNN_GRID,DECISION_TREE_GRID,LINEAR_SVC_GRID,XGBOOST_PARAM_GRID, RANDOM_FOREST_GRID_SEARCH,NAIVE_BAYES_GRID
from config_module.config import JSON_FILE
    
def random_forest(X_train, y_train, X_test, X_val, y_val, y_test):
    rf_model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    
    #Training
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    #Prediction
    start_time = time.time()
    y_val_pred = rf_model.predict(X_val)
    y_val_proba = rf_model.predict_proba(X_val)[:, 1]
    y_test_pred = rf_model.predict(X_test)
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time(s): {prediction_time:.2f}")
    
    save_model_and_metrics(
        rf_model,
        y_val,
        y_val_pred,
        y_val_proba,
        y_test,
        y_test_pred,
        y_test_proba,
        training_time,
        prediction_time,
        model_name_match = "Random Forest",
        config_path=JSON_FILE,
          # Chọn theo tên
    )
    
    
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba
    
    
def random_forest_grid_search(X_train, y_train, X_test, X_val,y_val, y_test):

    # Initial Grid Search
    rf_model = RandomForestClassifier(random_state=N_ESTIMATORS, n_jobs=N_JOBS)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=RANDOM_FOREST_GRID_SEARCH,
        cv=CV, 
        scoring='recall', 
        n_jobs=N_JOBS,  
        verbose=VERBOSE
    )

    # Training
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Tranining time (s): {training_time:.2f}")


    # Get best model
    best_rf_model = grid_search.best_estimator_
    print("\nBest hyperparameter :", grid_search.best_params_)
    print("Best recall on cross-validation:", grid_search.best_score_)

    # Prediction
    start_time = time.time()
    y_val_pred = best_rf_model.predict(X_val)
    y_val_proba = best_rf_model.predict_proba(X_val)[:, 1]
    y_test_pred = best_rf_model.predict(X_test)
    y_test_proba = best_rf_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s): {prediction_time:.2f}")
    save_model_and_metrics(
        best_rf_model,
        y_val,
        y_val_pred,
        y_val_proba,
        y_test,
        y_test_pred,
        y_test_proba,
        training_time,
        prediction_time,
        model_name_match = "Random Forest Grids",
        config_path=JSON_FILE,
          # Chọn theo tên
    )
    
    
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba, best_rf_model
    

 
def naive_bayes_opt_gs(X_train, y_train, X_test, X_val, y_val, y_test):
    nb_model = GaussianNB()
    
    grid_search = GridSearchCV(
    estimator=GaussianNB(),
    param_grid=NAIVE_BAYES_GRID,
    scoring='f1',     # Hoặc 'accuracy', 'roc_auc'
    cv=5,
    n_jobs=-1,
    verbose=1
    )
    start_time = time.time()
    grid_search.fit(X_train, y_train)  
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")
    
    best_nb_model = grid_search.best_estimator_
    print("\nBest hyperparameter :", grid_search.best_params_)
    print("Best recall on cross-validation:", grid_search.best_score_)

    # Prediction
    start_time = time.time()
    y_val_pred = best_nb_model.predict(X_val)
    y_val_proba = best_nb_model.predict_proba(X_val)[:, 1]
    y_test_pred = best_nb_model.predict(X_test)
    y_test_proba = best_nb_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    
    save_model_and_metrics(
        best_nb_model,
        y_val,
        y_val_pred,
        y_val_proba,
        y_test,
        y_test_pred,
        y_test_proba,
        training_time,
        prediction_time,
        model_name_match = "Naive Bayes",
        config_path=JSON_FILE,
        
    )
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,best_nb_model
    
def decision_tree(X_train, y_train, X_test, X_val, y_val, y_test):
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(estimator=dt_model, param_grid=DECISION_TREE_GRID, 
                           cv=5, scoring='f1', n_jobs=-1, verbose=1)

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")
    
    best_dt_model = grid_search.best_estimator_
    print("\nBest hyperparameter :", grid_search.best_params_)
    print("Best recall on cross-validation:", grid_search.best_score_)

    # Prediction
    start_time = time.time()
    y_val_pred = best_dt_model.predict(X_val)
    y_val_proba = best_dt_model.predict_proba(X_val)[:, 1]
    y_test_pred = best_dt_model.predict(X_test)
    y_test_proba = best_dt_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    
    save_model_and_metrics(
        best_dt_model,
        y_val,
        y_val_pred,
        y_val_proba,
        y_test,
        y_test_pred,
        y_test_proba,
        training_time,
        prediction_time,
        model_name_match = "Decision Tree",
        config_path=JSON_FILE,
          # Chọn theo tên
    )
   
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,best_dt_model

def knn(X_train, y_train, X_test, X_val, y_val, y_test):
    knn_model = KNeighborsClassifier()
    
    grid_search = GridSearchCV(estimator= knn_model, param_grid=KNN_GRID, 
                           cv=5, scoring='f1', n_jobs=-1, verbose=1)
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")
    
    best_knn_model = grid_search.best_estimator_
    print("\nBest hyperparameter :", grid_search.best_params_)
    print("Best recall on cross-validation:", grid_search.best_score_)

    # Prediction
    start_time = time.time()
    y_val_pred = best_knn_model.predict(X_val)
    y_val_proba = best_knn_model.predict_proba(X_val)[:, 1]
    y_test_pred = best_knn_model.predict(X_test)
    y_test_proba = best_knn_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    
    save_model_and_metrics(
        best_knn_model,
        y_val,
        y_val_pred,
        y_val_proba,
        y_test,
        y_test_pred,
        y_test_proba,
        training_time,
        prediction_time,
        model_name_match = "Knn",
        config_path=JSON_FILE,
          # Chọn theo tên
    )
    
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,best_knn_model

def linear_svc(X_train, y_train, X_test, X_val, y_val, y_test):
     # Tạo base model
    base_model = LinearSVC(random_state=RANDOM_STATE, dual=False)

    # Tạo mô hình hiệu chỉnh xác suất
    svm_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=5)

    # Grid search
    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=LINEAR_SVC_GRID,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # Training
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")
    
    best_svm_model = grid_search.best_estimator_
    print("\nBest hyperparameter :", grid_search.best_params_)
    print("Best recall on cross-validation:", grid_search.best_score_)

    # Prediction
    start_time = time.time()
    y_val_pred = best_svm_model.predict(X_val)
    y_val_proba = best_svm_model.predict_proba(X_val)[:, 1]
    y_test_pred = best_svm_model.predict(X_test)
    y_test_proba = best_svm_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    
    save_model_and_metrics(
        best_svm_model,
        y_val,
        y_val_pred,
        y_val_proba,
        y_test,
        y_test_pred,
        y_test_proba,
        training_time,
        prediction_time,
        model_name_match = "Linear SVC",
        config_path= JSON_FILE,
          # Chọn theo tên
    )
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,best_svm_model

def xgboost(X_train, y_train, X_test, X_val, y_val, y_test):
    xgb_model = XGBClassifier(
    scale_pos_weight=XGBOOST_SCALE_POSITIVE_WEIGHT,
    colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
    objective=OBJECTIVE,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)
    
    grid_search = GridSearchCV(estimator=xgb_model,
                           param_grid=XGBOOST_PARAM_GRID,
                           scoring='f1',   
                           cv=5,
                           verbose=1,
                           n_jobs=-1,error_score='raise')

    # Training
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")
    
    best_xgboost_model = grid_search.best_estimator_
    print("\nBest hyperparameter :", grid_search.best_params_)
    print("Best recall on cross-validation:", grid_search.best_score_)

    # Prediction
    start_time = time.time()
    y_val_pred = best_xgboost_model.predict(X_val)
    y_val_proba = best_xgboost_model.predict_proba(X_val)[:, 1]
    y_test_pred = best_xgboost_model.predict(X_test)
    y_test_proba = best_xgboost_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    
    save_model_and_metrics(
        best_xgboost_model,
        y_val,
        y_val_pred,
        y_val_proba,
        y_test,
        y_test_pred,
        y_test_proba,
        training_time,
        prediction_time,
        model_name_match = "XGBoost",
        config_path= JSON_FILE,
          # Chọn theo tên
    )
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,best_xgboost_model

