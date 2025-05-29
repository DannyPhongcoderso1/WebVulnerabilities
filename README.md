# Phát hiện các cuộc tấn công HTTP sử dụng Machine Learning

## Mô tả đề tài

Đề tài sử dụng các mô hình học máy như Random Forest, Naive Bayes, Decision Tree, KNN, XGBoost và Linear SVC để phát hiện các cuộc tấn công trong các yêu cầu HTTP dựa trên dữ liệu từ tập CSIC 2010. Mục tiêu là huấn luyện mô hình có độ chính xác cao và ứng dụng được vào thực tiễn trong việc phát hiện xâm nhập hệ thống web.

## Cấu trúc thư mục

WEBVULNERABILITIES/
├── config_module/
│   ├── config.json
│   └── config.py
├── data/
│   ├── csic_database.csv
│   └── raw_data.py
├── features/
├── inference/
├── models/
│   ├── evaluate_models.py
│   ├── models.pkl
│   └── train_model.py
├── preprocessing/
│   ├── ppc.ipynb
│   ├── preprocessing.py
│   └── xml_preprocessing.py
├── utils/
│   ├── utils.py
│   └── model_results.csv
└── final.ipynb

## Hướng dẫn chạy

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt

2. Tiền xử lí dữ liệu
python scripts/preprocess.py

3. Huấn luyện mô hình
python scripts/train_models.py

4. Đánh giá mô hình
python scripts/evaluate_models.py

Thông tin thêm 
* Các mô hình và tham số được tinh chỉnh bằng Grid Search
* Kết quả đánh giá được lưu trong file results.csv
* Mô hình tốt nhất được lưu ra các file pkl
