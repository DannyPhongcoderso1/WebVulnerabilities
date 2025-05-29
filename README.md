# Phát hiện các cuộc tấn công HTTP sử dụng Machine Learning

## Mô tả đề tài

Đề tài sử dụng các mô hình học máy như Random Forest, Naive Bayes, Decision Tree, KNN, XGBoost và Linear SVC để phát hiện các cuộc tấn công trong các yêu cầu HTTP dựa trên dữ liệu từ tập CSIC 2010. Mục tiêu là huấn luyện mô hình có độ chính xác cao và ứng dụng được vào thực tiễn trong việc phát hiện xâm nhập hệ thống web.

## Cấu trúc thư mục

```text
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
```

## Môi trường sử dụng:
Python: 3.12.5
Hệ điều hành: Windows 11
IDE khuyên dùng: VSCode
* Thư viện chính:
    * scikit-learn
    * xgboost
    * pandas
    * matplotlib
    * seaborn

## Hướng dẫn chạy

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirement.txt
```
2.Mở file notebook
Chạy lệnh sau để khởi động Jupyter Notebook
```bash
jupyter notebook
```
Mở file final.ipynb (nằm ở thư mục gốc)

3. Thực hiện từng bước trong Notebook:
File được thiết kết để chạy tuần tự từng phần:
    1. Tiền xử lí dữ liệu (Load dữ liệu từ data/ vào preprocessing.py, phân chia dữ liệu thành các file .npy)
    2. Huấn luyện mô hình (Đọc cấu hình từ Config.py vào train_model.py, đọc các dữ liệu đã lưu vào train_model.py, các mô hình tốt nhất được lưu ra thành các file .pkl)
    3. Đánh giá (kết quả đánh giá được lưu vào file model_results.csv)

Thông tin thêm 
* Các mô hình và tham số được tinh chỉnh bằng Grid Search
* Kết quả đánh giá được lưu trong file results.csv
* Mô hình tốt nhất được lưu ra các file pkl
