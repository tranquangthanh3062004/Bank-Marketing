# 🏦 Data Lakehouse Chuẩn AI Engine Cho Dự Án Bank Marketing

Hệ thống **Data Lakehouse chuẩn AI Engine** xây dựng trên kiến trúc Medallion (Bronze – Silver – Gold) tích hợp **DuckDB + Parquet**, **Feature Store (Offline & Online)**, **MLOps Engine (LightGBM & Baseline Logistic Regression)**, **SHAP Explainability** và **Telemarketing Lead Scoring Serving API (FastAPI)**.

---

## 🏗️ 1. Kiến Trúc Tổng Thể (Architecture Overview)

```text
                                [ NGUỒN DỮ LIỆU THÔ ]
                     (CRM, File Batch CSV, Campaign Call Logs)
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           1. DATA LAKEHOUSE STORAGE                             │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ BRONZE LAYER (data/lakehouse/bronze/)                                   │   │
│   │ - Append-only raw Parquet + Audit metadata:                             │   │
│   │   (_batch_id, _ingested_at, _source_file, _record_hash)                 │   │
│   └────────────────────────────────────┬────────────────────────────────────┘   │
│                                        │                                        │
│                 [ Data Quality Contracts & Quarantine Split ]                   │
│                                        │                                        │
│   ┌────────────────────────────────────▼────────────────────────────────────┐   │
│   │ SILVER LAYER (data/lakehouse/silver/)                                   │   │
│   │ - dim_customer.parquet (SCD Type 1 - Cleaned Demographics & Portfolio) │   │
│   │ - fact_campaign_interaction.parquet (Conformed Telemarketing Facts)     │   │
│   │ - quarantine/ (Cách ly các bản ghi sai độ tuổi, thời lượng âm, format)  │   │
│   └────────────────────────────────────┬────────────────────────────────────┘   │
│                                        │                                        │
│                                        ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ GOLD LAYER & FEATURE STORE (data/lakehouse/gold/)                       │   │
│   │ - mart_campaign_performance.parquet (Báo cáo ROI & Kênh liên lạc)       │   │
│   │ - mart_customer_segment_conversion.parquet (Tỷ lệ chuyển đổi phân khúc) │   │
│   │ - offline_feature_store.parquet (Bộ đặc trưng point-in-time không leak) │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             2. AI & MLOPS ENGINE                                │
│                                                                                 │
│   - Feature Pipeline: Chuẩn hóa số thực & One-Hot Encoding danh mục             │
│   - Model Trainer: Huấn luyện LightGBM / Logistic Regression cân bằng lớp       │
│   - Threshold Optimizer: Tối ưu ngưỡng quyết định theo F1 / Lợi nhuận kỳ vọng   │
│   - SHAP Explainability: Trích xuất Top nhân tố thúc đẩy / rào cản chuyển đổi   │
│   - Model Registry (models/): Lưu trữ versioning, weights, metadata             │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     3. TELEMARKETING ACTIVATION & SERVING                       │
│                                                                                 │
│   - Batch Scorer: Chấm điểm danh sách khách hàng, phân hạng ưu tiên (Tier 1-4)  │
│   - Real-time REST API (FastAPI): Endpoint dự đoán xác suất + giải thích SHAP   │
│   - Online Feature Store: Tra cứu hồ sơ khách hàng 360 độ theo customer_id      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 2. Cấu Trúc Thư Mục `data_engineer/`

```text
data_engineer/
├── README.md                          # Tài liệu kỹ thuật chi tiết
├── requirements.txt                   # Thư viện phụ thuộc chuẩn
├── config/                            # Cấu hình tập trung
│   ├── lakehouse_config.yaml          # Đường dẫn Bronze/Silver/Gold/DuckDB
│   └── model_config.yaml              # Cấu hình tính năng, tham số ML, threshold
├── data/                              # Dữ liệu lưu trữ Lakehouse
│   ├── raw/                           # Landing zone nhận file thô
│   └── lakehouse/
│       ├── bronze/                    # Parquet thô kèm metadata audit
│       ├── silver/                    # Star Schema: dim_customer & fact_interaction
│       ├── gold/                      # Business marts & offline feature store
│       ├── quarantine/                # Bản ghi vi phạm hợp đồng chất lượng
│       └── bank_lakehouse.duckdb      # DuckDB Database Catalog
├── sample_data/                       # Dữ liệu mẫu chuẩn UCI Bank Marketing
│   ├── bank_raw_sample.csv            # 600 bản ghi mẫu đầy đủ 17 thuộc tính
│   └── leads_to_score.csv             # 100 khách hàng tiềm năng phục vụ scoring
├── models/                            # Model Registry & Artifacts
│   ├── best_model.joblib              # Mô hình tốt nhất (LightGBM)
│   ├── best_preprocessor.joblib       # Pipeline tiền xử lý đặc trưng
│   └── model_metadata.json            # Thông số huấn luyện, metrics, optimal threshold
├── src/                               # MÃ NGUỒN HỆ THỐNG
│   ├── __init__.py
│   ├── config.py                      # YAML config loader & path resolver
│   ├── cli.py                         # Unified Command Line Interface
│   ├── lakehouse/                     # MODULE DATA LAKEHOUSE
│   │   ├── __init__.py
│   │   ├── db.py                      # DuckDB Manager kết nối và query Parquet
│   │   ├── contracts.py               # Hợp đồng schema Pandera & Pydantic
│   │   ├── ingestion.py               # Nạp dữ liệu thô vào Bronze Parquet
│   │   ├── transformation.py          # Bronze -> Silver conformed tables
│   │   ├── marts.py                   # Silver -> Gold analytical marts
│   │   └── feature_store.py           # Offline & Online Feature Store
│   ├── ai_engine/                     # MODULE AI & MLOPS
│   │   ├── __init__.py
│   │   ├── features.py                # Preprocessor không rò rỉ dữ liệu (No leakage)
│   │   ├── trainer.py                 # Huấn luyện mô hình (LightGBM/LogisticRegression)
│   │   ├── evaluator.py               # Đánh giá ROC-AUC, PR-AUC & Threshold Tuner
│   │   ├── explainability.py          # SHAP explainer toàn cục và từng khách hàng
│   │   └── registry.py                # Quản lý phiên bản model & metadata
│   └── serving/                       # MODULE SERVING & TRIỂN KHAI
│       ├── __init__.py
│       ├── batch_scorer.py            # Chấm điểm hàng loạt danh sách cuộc gọi
│       └── app.py                     # REST API FastAPI real-time scoring
└── tests/                             # BỘ TEST TỰ ĐỘNG TOÀN DIỆN (100% PASS)
    ├── __init__.py
    ├── test_config.py                 # Kiểm tra load cấu hình
    ├── test_contracts.py              # Kiểm tra Data Quality & Quarantine
    ├── test_lakehouse.py              # Kiểm tra Bronze -> Silver -> Gold
    ├── test_feature_store.py          # Kiểm tra Offline & Online Feature Store
    ├── test_ai_engine.py              # Kiểm tra Training, Threshold Tuning, SHAP
    ├── test_serving.py                # Kiểm tra Batch Scorer & REST API
    └── test_e2e_pipeline.py           # Kiểm tra toàn bộ luồng End-to-End
```

---

## ⚙️ 3. Hướng Dẫn Cài Đặt & Sử Dụng

### 3.1. Cài đặt môi trường
Từ thư mục gốc dự án:
```bash
pip install -r data_engineer/requirements.txt
```

### 3.2. Chạy Kiểm Thử Tự Động (Automated Testing)
Hệ thống đi kèm **20 bài kiểm thử tự động** bao phủ từ hợp đồng dữ liệu, ETL, Feature Store, Model Trainer đến REST API:
```bash
python -m pytest data_engineer/tests -v
```

### 3.3. Chạy Toàn Bộ Pipeline Tự Động (One-Click Pipeline)
Thực thi toàn bộ luồng từ Ingestion -> Silver -> Gold -> Feature Store -> AI Training -> Batch Scoring chỉ bằng một lệnh:
```bash
python -m data_engineer.src.cli run-all
```

---

## 🛠️ 4. Sử Dụng Từng Thành Phần Qua CLI

### 4.1. Nạp dữ liệu vào Bronze Layer (Ingestion)
```bash
python -m data_engineer.src.cli ingest --source data_engineer/sample_data/bank_raw_sample.csv
```

### 4.2. Làm sạch & Chuẩn hóa sang Silver Layer (Transformation)
```bash
python -m data_engineer.src.cli transform-silver
```

### 4.3. Xây dựng Gold Data Marts
```bash
python -m data_engineer.src.cli transform-gold
```

### 4.4. Khởi tạo AI Feature Store
```bash
python -m data_engineer.src.cli build-features
```

### 4.5. Huấn luyện Mô hình & Đăng ký Model Registry
Huấn luyện mô hình LightGBM hoặc Logistic Regression, tự động tìm ngưỡng quyết định tối ưu và tính toán SHAP:
```bash
python -m data_engineer.src.cli train --model lightgbm
```

### 4.6. Chấm điểm hàng loạt tệp Leads (Batch Scoring)
```bash
python -m data_engineer.src.cli batch-score --input data_engineer/sample_data/leads_to_score.csv --output data_engineer/sample_data/scored_leads_output.csv
```

Kết quả xuất ra sẽ bao gồm:
- `predicted_conversion_prob`: Xác suất khách hàng sẽ gửi tiền.
- `priority_tier`: Phân loại khách hàng (`Tier 1 - Hot`, `Tier 2 - Warm`, `Tier 3 - Neutral`, `Tier 4 - Cold`).
- `telesales_action`: Đề xuất kịch bản chăm sóc tương ứng cho tổng đài viên.

### 4.7. Khởi chạy REST API Server (FastAPI)
```bash
python -m data_engineer.src.cli serve --host 127.0.0.1 --port 8000
```
Truy cập Swagger UI trực quan tại: **http://127.0.0.1:8000/docs**

---

## 📡 5. Các Endpoint API Chính

| Phương thức | Đường dẫn | Chức năng |
| :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra tình trạng sức khỏe hệ thống và model đang nạp |
| `GET` | `/api/v1/model/info` | Tra cứu metadata phiên bản model, metrics và optimal threshold |
| `POST` | `/api/v1/predict` | Chấm điểm thời gian thực 1 khách hàng + Giải thích **SHAP** (Top 3 lý do thúc đẩy/rào cản) |
| `POST` | `/api/v1/batch-predict` | Chấm điểm thời gian thực danh sách nhiều khách hàng |
| `GET` | `/api/v1/features/{customer_id}` | Tra cứu hồ sơ đặc trưng 360 độ từ Online Feature Store |

### Ví dụ Request Chấm Điểm Real-Time (`POST /api/v1/predict`):
```json
{
  "customer_id": "CUST_9901",
  "age": 45,
  "job": "management",
  "marital": "married",
  "education": "tertiary",
  "default": "no",
  "balance": 8500.0,
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "aug",
  "duration": 420,
  "campaign": 1,
  "pdays": 120,
  "previous": 2,
  "poutcome": "success"
}
```

### Ví dụ Response Trả Về Kèm Giải Thích SHAP:
```json
{
  "customer_id": "CUST_9901",
  "conversion_probability": 0.8924,
  "is_recommended_call": true,
  "decision_threshold": 0.26,
  "priority_tier": "Tier 1 (Hot)",
  "recommended_action": "Chuyển ngay cho Senior Telesales; gọi trong 30 phút",
  "top_positive_drivers": [
    { "feature": "duration", "impact": 1.4043 },
    { "feature": "balance", "impact": 0.4723 },
    { "feature": "poutcome_success", "impact": 0.3512 }
  ],
  "top_negative_barriers": []
}
```
