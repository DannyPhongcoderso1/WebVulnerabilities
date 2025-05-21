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

from config_module.config import RANDOM_STATE, MAX_ITER, N_JOBS, VERBOSE, N_ESTIMATORS, GRID_SEARCH_N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT, CV, KNN_N_NEIGHBORS, KNN_METRIC, XGBOOST_N_ESTIMATORS, XGBOOST_MAX_DEPTH, XGBOOST_LEARNING_RATE, XGBOOST_SUBSAMPLE,XGBOOST_SCALE_POSITIVE_WEIGHT, XGBOOST_COLSAMPLE_BYTREE, OBJECTIVE, LINEAR_SVC_C, LINEAR_SVC_CV, LINEAR_SVC_TOL, LINEAR_SVC_MAX_ITER
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
    # Define hyperparameter 
    param_grid = {
        'n_estimators': GRID_SEARCH_N_ESTIMATORS,  
        'max_depth': MAX_DEPTH,   
        'min_samples_split': MIN_SAMPLES_SPLIT      
    }

    # Initial Grid Search
    rf_model = RandomForestClassifier(random_state=N_ESTIMATORS, n_jobs=N_JOBS)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
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
    start_time = time.time()
    nb_model.fit(X_train, y_train)  
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")

    # Prediction
    start_time = time.time()
    y_val_pred = nb_model.predict(X_val)
    y_val_proba = nb_model.predict_proba(X_val)[:, 1]
    y_test_pred = nb_model.predict(X_test)
    y_test_proba = nb_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    
    save_model_and_metrics(
        nb_model,
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
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,nb_model
    
def decision_tree(X_train, y_train, X_test, X_val, y_val, y_test):
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)

    start_time = time.time()
    dt_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")

    # Prediction
    start_time = time.time()
    y_val_pred = dt_model.predict(X_val)
    y_val_proba = dt_model.predict_proba(X_val)[:, 1]
    y_test_pred = dt_model.predict(X_test)
    y_test_proba = dt_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    
    save_model_and_metrics(
        dt_model,
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
   
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,dt_model

def knn(X_train, y_train, X_test, X_val, y_val, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS, n_jobs=N_JOBS)
    start_time = time.time()
    knn_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")

    # Prediction
    start_time = time.time()
    y_val_pred = knn_model.predict(X_val)
    y_val_proba = knn_model.predict_proba(X_val)[:, 1]
    y_test_pred = knn_model.predict(X_test)
    y_test_proba = knn_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    
    save_model_and_metrics(
        knn_model,
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
    
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,knn_model

def linear_svc(X_train, y_train, X_test, X_val, y_val, y_test):
    base_model = LinearSVC(C=LINEAR_SVC_C, random_state=RANDOM_STATE, tol=LINEAR_SVC_TOL, max_iter=MAX_ITER)
    svm_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=LINEAR_SVC_CV)

    # Training
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")

    # Prediction
    start_time = time.time()
    y_val_pred = svm_model.predict(X_val)
    y_val_proba = svm_model.predict_proba(X_val)[:, 1]
    y_test_pred = svm_model.predict(X_test)
    y_test_proba = svm_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    
    save_model_and_metrics(
        svm_model,
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
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,svm_model

def xgboost(X_train, y_train, X_test, X_val, y_val, y_test):
    xgb_model = XGBClassifier(
    scale_pos_weight=XGBOOST_SCALE_POSITIVE_WEIGHT,
    n_estimators=XGBOOST_N_ESTIMATORS,
    max_depth=XGBOOST_MAX_DEPTH,
    learning_rate=XGBOOST_LEARNING_RATE,
    subsample=XGBOOST_SUBSAMPLE,
    colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
    objective=OBJECTIVE,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

    # Training
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")

    # Prediction
    start_time = time.time()
    y_val_pred = xgb_model.predict(X_val)
    y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
    y_test_pred = xgb_model.predict(X_test)
    y_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    
    save_model_and_metrics(
        xgb_model,
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
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,xgb_model

