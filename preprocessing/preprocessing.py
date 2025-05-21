import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from urllib.parse import urlparse
import re
import pandas as pd
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("config_module/config"), '..')))

from config_module.config import PCA_COMPONENT, MAX_FEATURE

def extract_features(url):
    parsed = urlparse(url)
    features = {}
    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_at"] = url.count("@")
    features["uses_https"] = int(parsed.scheme == "https")
    features["has_ip"] = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', parsed.netloc)))
    features["num_subdomains"] = max(0, len(parsed.netloc.split(".")) - 2)
    features["path_length"] = len(parsed.path)
    return features

def CSIC_preprocess(df: pd.DataFrame):
    # df = pd.read_csv(url, delimiter=',', on_bad_lines='skip')
    # df['Accept'] = df['Accept'].fillna(df['Accept'].mode()[0])
    
    df['lenght'] = df['lenght'].fillna('0')
    df['content-type'] = df['content-type'].fillna('None')
    df['content'] = df['content'].fillna('None')
    df['is_post'] = df['Method'].apply(lambda x: 1 if x == 'POST' else 0)

    df.drop(columns=['lable'],  errors='ignore', inplace=True)
    
    malicious_keywords = [
        'SELECT', 'UNION', 'DROP', 'DELETE', 'FROM', 'WHERE', 'OR', 'LIKE', 'AND', '1=1', '--', '\'',
        'SCRIPT', 'javascript', 'alert', 'iframe', 'src=', 'onerror', 'prompt', 'confirm', 'eval', 'onload',
        'mouseover', 'onunload', 'document.', 'window.', 'xmlhttprequest', 'xhr', 'cookie',
        'tamper', 'vaciar', 'carrito', 'incorrect', 'pwd', 'login', 'password', 'id',
        '%0D', '%0A', '.php', '.js', 'admin', 'administrator'
    ]

    # Feature extraction from URL
    df['url_length'] = df['URL'].apply(len)
    df['url_special_chars'] = df['URL'].apply(lambda x: len(re.findall(r'[%;=<>\/&\'"()\[\]#\-\+]', x)))
    df['url_malicious_keywords'] = df['URL'].apply(lambda x: sum(1 for kw in malicious_keywords if kw.lower() in x.lower()))
    df['url_params_count'] = df['URL'].apply(lambda x: x.count('&') + 1 if '?' in x else 0)

    # Feature extraction from content
    df['content_length'] = df['content'].fillna('').apply(len)
    df['content_special_chars'] = df['content'].fillna('').apply(lambda x: len(re.findall(r'[%;=<>\/&\'"()\[\]#\-\+]', x)))
    df['content_malicious_keywords'] = df['content'].fillna('').apply(lambda x: sum(1 for kw in malicious_keywords if kw.lower() in x.lower()))

    # Check Feature Distribution
    print("PFeature Distribution by Class:")
    print(df.groupby('classification')[['url_length', 'url_special_chars', 'url_malicious_keywords', 'url_params_count',
                                                'content_length', 'content_special_chars', 'content_malicious_keywords']].mean())

    # Save Feature contribution
    df.to_csv('dataset_with_features.csv', index=False)
    print("Feature Distribution was saved!")
    
    df['content'] = df['content'].fillna(' ')
    url_tfidf = TfidfVectorizer(max_features=MAX_FEATURE, lowercase=True, token_pattern=r'(?u)\b\w+\b')
    url_features = url_tfidf.fit_transform(df['URL']).toarray()
    # url_feature_names = url_tfidf.get_feature_names_out()
    
    content_tfidf = TfidfVectorizer(max_features=MAX_FEATURE, lowercase=True, token_pattern=r'(?u)\b\w+\b')
    content_features = content_tfidf.fit_transform(df['content']).toarray()
    # content_feature_names = content_tfidf.get_feature_names_out()
    
    df = pd.get_dummies(
        df,                          # DataFrame gốc
        columns=['Method', 'content-type'],    # Cột cần mã hóa
        prefix=['Method', 'content-type'],     # Tiền tố cho tên cột mới
        drop_first=True                        # Bỏ cột đầu tiên (giảm đa cộng tuyến)
    )
    
    manual_features = df[['url_length', 'url_special_chars', 'url_malicious_keywords', 'url_params_count',
                               'content_length', 'content_special_chars', 'content_malicious_keywords',
                               'is_post']]  

    one_hot_features = df.filter(like='Method_|content-type_')
    feature_matrix = np.hstack([
        manual_features.values, 
        one_hot_features.values, 
        url_features, 
        content_features
    ])
    

    pca = PCA(n_components=PCA_COMPONENT)  # hoặc chọn giữ 95% phương sai
    feature_matrix = pca.fit_transform(feature_matrix)
    
    print("Feature matrix shape:", feature_matrix.shape[1])
    
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(feature_matrix, df['classification'].values)
    
    return X_resampled, y_resampled


def parsed_request_test_preprocess(df: pd.DataFrame):
    
    df['content-type'] = df['Content-Type'].fillna('None')
    df['content'] = df['Content'].fillna('None')
    df['is_post'] = df['Method'].apply(lambda x: 1 if x == 'POST' else 0)

    df.drop(columns=['lable'],  errors='ignore', inplace=True)
    df["classification"] = df["classification"].apply(lambda x: 0 if x == 'Valid' else 1)

    
    malicious_keywords = [
        'SELECT', 'UNION', 'DROP', 'DELETE', 'FROM', 'WHERE', 'OR', 'LIKE', 'AND', '1=1', '--', '\'',
        'SCRIPT', 'javascript', 'alert', 'iframe', 'src=', 'onerror', 'prompt', 'confirm', 'eval', 'onload',
        'mouseover', 'onunload', 'document.', 'window.', 'xmlhttprequest', 'xhr', 'cookie',
        'tamper', 'vaciar', 'carrito', 'incorrect', 'pwd', 'login', 'password', 'id',
        '%0D', '%0A', '.php', '.js', 'admin', 'administrator'
    ]

    # Feature extraction from URL
    df['url_length'] = df['URL'].fillna('').apply(len)
    df['url_special_chars'] = df['URL'].fillna('').apply(lambda x: len(re.findall(r'[%;=<>\/&\'"()\[\]#\-\+]', x)))
    df['url_malicious_keywords'] = df['URL'].fillna('').apply(lambda x: sum(1 for kw in malicious_keywords if kw.lower() in x.lower()))
    df['url_params_count'] = df['URL'].fillna('').apply(lambda x: x.count('&') + 1 if '?' in x else 0)

    # Feature extraction from content
    df['content_length'] = df['content'].fillna('').apply(len)
    df['content_special_chars'] = df['content'].fillna('').apply(lambda x: len(re.findall(r'[%;=<>\/&\'"()\[\]#\-\+]', x)))
    df['content_malicious_keywords'] = df['content'].fillna('').apply(lambda x: sum(1 for kw in malicious_keywords if kw.lower() in x.lower()))

    # Check Feature Distribution
    print("PFeature Distribution by Class:")
    print(df.groupby('classification')[['url_length', 'url_special_chars', 'url_malicious_keywords', 'url_params_count',
                                                'content_length', 'content_special_chars', 'content_malicious_keywords']].mean())

    # Save Feature contribution
    df.to_csv('dataset_with_features.csv', index=False)
    print("Feature Distribution was saved!")
    
    df['content'] = df['content'].fillna(' ')
    url_tfidf = TfidfVectorizer(max_features=MAX_FEATURE, lowercase=True, token_pattern=r'(?u)\b\w+\b')
    url_features = url_tfidf.fit_transform(df['URL'].fillna('')).toarray()
    url_feature_names = url_tfidf.get_feature_names_out()
    
    content_tfidf = TfidfVectorizer(max_features=MAX_FEATURE, lowercase=True, token_pattern=r'(?u)\b\w+\b')
    content_features = content_tfidf.fit_transform(df['content']).toarray()
    content_feature_names = content_tfidf.get_feature_names_out()
    
    df = pd.get_dummies(
        df,                          # DataFrame gốc
        columns=['Method', 'content-type'],    # Cột cần mã hóa
        prefix=['Method', 'content-type'],     # Tiền tố cho tên cột mới
        drop_first=True                        # Bỏ cột đầu tiên (giảm đa cộng tuyến)
    )
    
    manual_features = df[['url_length', 'url_special_chars', 'url_malicious_keywords', 'url_params_count',
                               'content_length', 'content_special_chars', 'content_malicious_keywords',
                               'is_post']]  

    one_hot_features = df.filter(like='Method_|content-type_')
    feature_matrix = np.hstack([
        manual_features.values, 
        one_hot_features.values, 
        url_features, 
        content_features
    ])
    

    pca = PCA(n_components=PCA_COMPONENT)  # hoặc chọn giữ 95% phương sai
    feature_matrix = pca.fit_transform(feature_matrix)
    
    print("Feature matrix shape:", feature_matrix.shape[1])
    
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    # Giả sử bạn đã load dữ liệu thành df, với cột 'URL' và 'classification'

    # Chuẩn bị feature (bỏ qua resample nếu chỉ test)
    X_test = feature_matrix  # sau khi xử lý dữ liệu
    y_test = df['classification'].values




    
    return X_test, y_test

def parsed_request_train_preprocess(df: pd.DataFrame):
    
    df['content-type'] = df['Content-Type'].fillna('None')
    df['content'] = df['Content'].fillna('None')
    df['is_post'] = df['Method'].apply(lambda x: 1 if x == 'POST' else 0)

    df.drop(columns=['lable'],  errors='ignore', inplace=True)
    df["classification"] = df["classification"].apply(lambda x: 0 if x == 'Valid' else 1)

    
    malicious_keywords = [
        'SELECT', 'UNION', 'DROP', 'DELETE', 'FROM', 'WHERE', 'OR', 'LIKE', 'AND', '1=1', '--', '\'',
        'SCRIPT', 'javascript', 'alert', 'iframe', 'src=', 'onerror', 'prompt', 'confirm', 'eval', 'onload',
        'mouseover', 'onunload', 'document.', 'window.', 'xmlhttprequest', 'xhr', 'cookie',
        'tamper', 'vaciar', 'carrito', 'incorrect', 'pwd', 'login', 'password', 'id',
        '%0D', '%0A', '.php', '.js', 'admin', 'administrator'
    ]

    # Feature extraction from URL
    df['url_length'] = df['URL'].fillna('').apply(len)
    df['url_special_chars'] = df['URL'].fillna('').apply(lambda x: len(re.findall(r'[%;=<>\/&\'"()\[\]#\-\+]', x)))
    df['url_malicious_keywords'] = df['URL'].fillna('').apply(lambda x: sum(1 for kw in malicious_keywords if kw.lower() in x.lower()))
    df['url_params_count'] = df['URL'].fillna('').apply(lambda x: x.count('&') + 1 if '?' in x else 0)

    # Feature extraction from content
    df['content_length'] = df['content'].fillna('').apply(len)
    df['content_special_chars'] = df['content'].fillna('').apply(lambda x: len(re.findall(r'[%;=<>\/&\'"()\[\]#\-\+]', x)))
    df['content_malicious_keywords'] = df['content'].fillna('').apply(lambda x: sum(1 for kw in malicious_keywords if kw.lower() in x.lower()))

    # Check Feature Distribution
    print("PFeature Distribution by Class:")
    print(df.groupby('classification')[['url_length', 'url_special_chars', 'url_malicious_keywords', 'url_params_count',
                                                'content_length', 'content_special_chars', 'content_malicious_keywords']].mean())

    # Save Feature contribution
    df.to_csv('dataset_with_features.csv', index=False)
    print("Feature Distribution was saved!")
    
    df['content'] = df['content'].fillna(' ')
    url_tfidf = TfidfVectorizer(max_features=MAX_FEATURE, lowercase=True, token_pattern=r'(?u)\b\w+\b')
    url_features = url_tfidf.fit_transform(df['URL'].fillna('')).toarray()
    url_feature_names = url_tfidf.get_feature_names_out()
    
    content_tfidf = TfidfVectorizer(max_features=MAX_FEATURE, lowercase=True, token_pattern=r'(?u)\b\w+\b')
    content_features = content_tfidf.fit_transform(df['content']).toarray()
    content_feature_names = content_tfidf.get_feature_names_out()
    
    df = pd.get_dummies(
        df,                          # DataFrame gốc
        columns=['Method', 'content-type'],    # Cột cần mã hóa
        prefix=['Method', 'content-type'],     # Tiền tố cho tên cột mới
        drop_first=True                        # Bỏ cột đầu tiên (giảm đa cộng tuyến)
    )
    
    manual_features = df[['url_length', 'url_special_chars', 'url_malicious_keywords', 'url_params_count',
                               'content_length', 'content_special_chars', 'content_malicious_keywords',
                               'is_post']]  

    one_hot_features = df.filter(like='Method_|content-type_')
    feature_matrix = np.hstack([
        manual_features.values, 
        one_hot_features.values, 
        url_features, 
        content_features
    ])
    

    pca = PCA(n_components=PCA_COMPONENT)  # hoặc chọn giữ 95% phương sai
    feature_matrix = pca.fit_transform(feature_matrix)
    
    print("Feature matrix shape:", feature_matrix.shape[1])
    
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(feature_matrix, df['classification'].values)
    
    return X_resampled, y_resampled


    


