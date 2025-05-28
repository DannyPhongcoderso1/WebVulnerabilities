# File paths

# Vectorizer
MAX_FEATURE = 1000


# Training params
TEST_SIZE_1 = 0.3
TEST_SIZE_2 = 0.67
RANDOM_STATE = 42

MAX_ITER = 10000
VERBOSE = 1
N_JOBS = 2

#RANDOM FOREST
N_ESTIMATORS = 100


#RANDOM FOREST GRID SEARCH
GRID_SEARCH_N_ESTIMATORS = [50, 100]  # Số lượng cây trong rừng
MAX_DEPTH = [10, None]  # Độ sâu tối đa của cây
MIN_SAMPLES_SPLIT = [2, 5]  # Số mẫu tối thiểu để chia một node
CV = 3  # Số lần gập trong cross-validation

RANDOM_FOREST_GRID_SEARCH = {
    'n_estimators': [50, 100],
    'max_depth': [10, None],
    'min_samples_split': [2,5]
}

#NB
NAIVE_BAYES_GRID = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # Độ mượt để tránh chia cho 0
}

#DECISION TREE 
DECISION_TREE_GRID= {
    'criterion': ['gini', 'entropy'],           # giảm từ 3 xuống 2
    'max_depth': [None, 10, 20],                 # giảm từ 5 xuống 3
    'min_samples_split': [2, 5],                 # giảm từ 3 xuống 2
    'min_samples_leaf': [1, 2],                   # giảm từ 3 xuống 2
    'max_features': [None, 'sqrt'],              # giảm từ 3 xuống 2
    'ccp_alpha': [0.0, 0.01]                      # giảm từ 3 xuống 2
}


#KNN
KNN_N_NEIGHBORS = 5
KNN_METRIC = 'euclidean'

KNN_GRID = {
        'n_neighbors': [3, 7],               
        'weights': ['uniform', 'distance'],  
        'algorithm': ['auto', 'kd_tree'],    
        'leaf_size': [20, 30],               
        'p': [1]                            
    }


#XGBoost
XGBOOST_SCALE_POSITIVE_WEIGHT = 1.5
XGBOOST_N_ESTIMATORS = 50
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_SUBSAMPLE = 0.8
XGBOOST_COLSAMPLE_BYTREE = 0.8
OBJECTIVE = 'binary:logistic'

XGBOOST_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [3, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
}


# LINEAR SVC
LINEAR_SVC_C = 1.0
LINEAR_SVC_TOL = 0.01
LINEAR_SVC_MAX_ITER = 1000
LINEAR_SVC_CV = 5

LINEAR_SVC_GRID = {
        'estimator__C': [0.1, 1, 10, 100],
        'estimator__tol': [1e-3],
        'estimator__max_iter': [1000]
    }



# Preprocessing
PCA_COMPONENT = 300

# Data paths
import os

import os

# Lấy thư mục config (chứa config.py)
CONFIG_DIR = os.path.dirname(__file__)

# Lấy thư mục cha của config (thư mục gốc dự án)
BASE_DIR = os.path.dirname(CONFIG_DIR)

# Đường dẫn tới thư mục data ở cùng cấp với config
JSON_DIR = os.path.join(BASE_DIR, "config_module")
DATA_DIR = os.path.join(BASE_DIR, "data")
XTRAIN = os.path.join(DATA_DIR, "X_train.npy")
YTRAIN = os.path.join(DATA_DIR, "y_train.npy")
XVAL = os.path.join(DATA_DIR, "X_val.npy")
YVAL = os.path.join(DATA_DIR, "y_val.npy")
XTEST = os.path.join(DATA_DIR, "X_test.npy")
YTEST = os.path.join(DATA_DIR, "y_test.npy")

CSIC_FILE = os.path.join(DATA_DIR, "csic_database.csv")

PARSE_REQUEST_TEST = os.path.join(DATA_DIR, "parsed_requests_test.csv")
JSON_FILE = os.path.join(JSON_DIR, "config.json")
PARSE_REQUEST_TRAIN = os.path.join(DATA_DIR, "parsed_requests_train.csv")

