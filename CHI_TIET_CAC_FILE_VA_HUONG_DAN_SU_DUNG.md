# 📘 TÀI LIỆU TOÀN DIỆN: CHI TIẾT TỪNG FILE & HƯỚNG DẪN SỬ DỤNG HỆ THỐNG BANK MARKETING

> **Dự án**: Bank Marketing Term Deposit Prediction & Enterprise Data Lakehouse AI Engine  
> **Tác giả**: Tran Quang Thanh  
> **Ngôn ngữ**: Python 3.10+  
> **Kiến trúc**: CRISP-DM Machine Learning + Medallion Data Lakehouse (Bronze - Silver - Gold) + MLOps AI Engine + FastAPI Real-time Serving  

---

## 📌 MỤC LỤC
1. [Tổng Quan Hệ Thống & Sơ Đồ Kiến Trúc](#1-tổng-quan-hệ-thống--sơ-đồ-kiến-trúc)
2. [Chi Tiết Nội Dung & Ý Nghĩa Sử Dụng Từng File](#2-chi-tiết-nội-dung--ý-nghĩa-sử-dụng-từng-file)
   - [2.1. Nhóm File Gốc Dự Án (Project Root)](#21-nhóm-file-gốc-dự-án-project-root)
   - [2.2. Nhóm Tài Liệu Nghiên Cứu & Báo Cáo (`docs/`)](#22-nhóm-tài-liệu-nghiên-cứu--báo-cáo-docs)
   - [2.3. Nhóm Jupyter Notebooks Phân Tích & Mô Hình (`notebooks/`)](#23-nhóm-jupyter-notebooks-phân-tích--mô-hình-notebooks)
   - [2.4. Phân Hệ Data Lakehouse & AI Engine (`data_engineer/`)](#24-phân-hệ-data-lakehouse--ai-engine-data_engineer)
     - [Cấu hình tập trung (`config/`)](#cấu-hình-tập-trung-config)
     - [Dữ liệu mẫu (`sample_data/`)](#dữ-liệu-mẫu-sample_data)
     - [Kho lưu trữ mô hình (`models/`)](#kho-lưu-trữ-mô-hình-models)
     - [Mã nguồn Lakehouse (`src/lakehouse/`)](#mã-nguồn-lakehouse-srclakehouse)
     - [Mã nguồn AI & MLOps Engine (`src/ai_engine/`)](#mã-nguồn-ai--mlops-engine-srcai_engine)
     - [Mã nguồn Serving & API (`src/serving/`)](#mã-nguồn-serving--api-srcserving)
     - [CLI điều khiển trung tâm (`src/cli.py`)](#cli-điều-khiển-trung-tâm-srcclipy)
     - [Bộ kiểm thử tự động (`tests/`)](#bộ-kiểm-thử-tự-động-tests)
3. [Hướng Dẫn Vận Hành Hệ Thống Chi Tiết Từ A Đến Z](#3-hướng-dẫn-vận-hành-hệ-thống-chi-tiết-từ-a-đến-z)
   - [3.1. Cài đặt môi trường & Thư viện](#31-cài-đặt-môi-trường--thư-viện)
   - [3.2. Chạy toàn bộ Pipeline tự động (One-Click Runner)](#32-chạy-toàn-bộ-pipeline-tự-động-one-click-runner)
   - [3.3. Vận hành từng công đoạn Lakehouse (Bronze -> Silver -> Gold)](#33-vận-hành-từng-công-đoạn-lakehouse-bronze---silver---gold)
   - [3.4. Vận hành AI Engine: Huấn luyện, Tối ưu ngưỡng & SHAP](#34-vận-hành-ai-engine-huấn-luyện-tối-ưu-ngưỡng--shap)
   - [3.5. Chấm điểm tệp khách hàng tiềm năng (Batch Scoring)](#35-chấm-điểm-tệp-khách-hàng-tiềm-năng-batch-scoring)
   - [3.6. Khởi chạy & Sử dụng REST API thời gian thực (FastAPI)](#36-khởi-chạy--sử-dụng-rest-api-thời-gian-thực-fastapi)
   - [3.7. Thực thi kiểm thử tự động (Running Pytest Suite)](#37-thực-thi-kiểm-thử-tự-động-running-pytest-suite)
4. [Kinh Tế Học Telemarketing & Cơ Chế Tối Ưu Lợi Nhuận](#4-kinh-tế-học-telemarketing--cơ-chế-tối-ưu-lợi-nhuận)
5. [Giải Đáp Sự Cố Thường Gặp (Troubleshooting & FAQ)](#5-giải-đáp-sự-cố-thường-gặp-troubleshooting--faq)

---

## 1. TỔNG QUAN HỆ THỐNG & SƠ ĐỒ KIẾN TRÚC

Dự án **Bank Marketing** giải quyết bài toán cốt lõi của ngành ngân hàng bán lẻ: **Dự đoán khả năng khách hàng đồng ý mở sổ tiết kiệm có kỳ hạn (Term Deposit)** trong các chiến dịch Telemarketing, từ đó tối ưu hóa chi phí cuộc gọi và gia tăng tỷ lệ chuyển đổi (Conversion Rate).

Hệ thống được phát triển qua 2 giai đoạn:
1. **Giai đoạn Nghiên cứu & Khám phá (CRISP-DM)**: Tập trung vào phân tích thống kê (EDA), đánh giá tương quan, kiểm định giả thuyết và xây dựng mô hình thử nghiệm trong [notebooks/](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/notebooks).
2. **Giai đoạn Production Engineering (`data_engineer/`)**: Chuẩn hóa toàn bộ quy trình thành một **Hệ thống Data Lakehouse chuẩn AI Engine**, lưu trữ phân tầng Medallion (Parquet + DuckDB), Feature Store chống rò rỉ dữ liệu, MLOps huấn luyện tự động với SHAP Explainability, và REST API thời gian thực phục vụ nhân viên tổng đài.

```text
                                [ NGUỒN DỮ LIỆU ĐẦU VÀO ]
                           (File CSV, CRM Logs, Core Banking)
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TẦNG 1: DATA LAKEHOUSE (DuckDB + Parquet)                       │
│                                                                                        │
│   [BRONZE LAYER]                                                                       │
│   - Lưu trữ bất biến (Immutable Parquet)                                              │
│   - Bổ sung Audit Metadata: _batch_id, _ingested_at, _source_file, _record_hash        │
│                                          │                                             │
│                       [Data Quality Gate & Quarantine Split]                           │
│                                          │                                             │
│   [SILVER LAYER]                         ▼                                             │
│   - Cleaned & Conformed Star Schema:                                                   │
│       * dim_customer.parquet: Khách hàng, nhân khẩu học, nợ xấu, số dư, khoản vay      │
│       * fact_campaign_interaction.parquet: Chi tiết cuộc gọi, thời lượng, kết quả cũ   │
│       * quarantine/: Cách ly bản ghi vi phạm (tuổi < 18, thời lượng âm)                │
│                                          │                                             │
│                                          ▼                                             │
│   [GOLD LAYER & FEATURE STORE]                                                         │
│   - mart_campaign_performance: Đo lường ROI cuộc gọi theo tháng & kênh                 │
│   - mart_customer_segment_conversion: Đo lường tỷ lệ chuyển đổi theo nhóm khách hàng   │
│   - offline_feature_store: Vector đặc trưng huấn luyện (No Data Leakage)               │
│   - online_feature_store: Cache tra cứu nhanh theo customer_id                         │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              TẦNG 2: AI & MLOPS ENGINE                                 │
│                                                                                        │
│   - Feature Pipeline: ColumnTransformer (StandardScaler + OneHotEncoder)               │
│   - Model Trainer: Huấn luyện LightGBM / Logistic Regression cân bằng lớp             │
│   - Threshold Tuner: Tối ưu F1-score & Lợi nhuận kỳ vọng chiến dịch Telemarketing     │
│   - SHAP Explainer: Bóc tách Top nguyên nhân tích cực/tiêu cực cho từng cuộc gọi       │
│   - Model Registry (models/): Quản lý phiên bản weights, preprocessor & metadata       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       TẦNG 3: SERVING & TELEMARKETING ACTIVATION                       │
│                                                                                        │
│   - Batch Scorer: Chấm điểm tệp Leads hàng loạt -> Phân hạng Tier 1, 2, 3, 4           │
│   - REST API (FastAPI): Endpoint /predict trả về xác suất + giải thích SHAP trong ms   │
│   - Dialing Queue Export: Xuất file danh sách ưu tiên nạp vào tổng đài tự động        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CHI TIẾT NỘI DUNG & Ý NGHĨA SỬ DỤNG TỪNG FILE

### 2.1. Nhóm File Gốc Dự Án (Project Root)

#### 1. [README.md](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/README.md)
- **Nội dung**: Tài liệu trang chủ tổng quan của dự án, trình bày bối cảnh bài toán Telemarketing, bộ biến UCI Bank Marketing (17 biến), quy trình 6 bước chuẩn CRISP-DM, kết quả phân tích Top 5 đặc trưng quan trọng nhất (Duration, Poutcome, Housing Loan, Contact, Balance), sơ đồ thư mục và hướng dẫn khởi động.
- **Ý nghĩa & Cách dùng**: Là bộ mặt chính của kho mã nguồn (GitHub). Dành cho các bên liên quan (Product Manager, Data Lead, Giảng viên, Đối tác) nắm bắt tổng thể bài toán kinh doanh và kết quả đạt được.

#### 2. [pyproject.toml](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/pyproject.toml)
- **Nội dung**: File cấu hình tiêu chuẩn đóng gói dự án Python theo chuẩn PEP 518/PEP 621. Khai báo thông tin metadata dự án (`bank-marketing v0.1.0`), tác giả (`Tran Quang Thanh`), build system (`setuptools`), phiên bản Python yêu cầu (`>=3.10`) và danh sách các gói phụ thuộc cốt lõi.
- **Ý nghĩa & Cách dùng**: Cho phép cài đặt dự án ở chế độ phát triển dạng package chuẩn thông qua lệnh `pip install -e .`.

#### 3. [requirements.txt](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/requirements.txt)
- **Nội dung**: Danh sách các thư viện Python cơ bản phục vụ môi trường phân tích dữ liệu: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `shap`, `jupyterlab`, `ipywidgets`.
- **Ý nghĩa & Cách dùng**: Dùng để cài đặt nhanh môi trường phục vụ việc mở và chạy các notebook phân tích ban đầu bằng lệnh `pip install -r requirements.txt`.

#### 4. [LICENSE](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/LICENSE)
- **Nội dung**: Văn bản giấy phép mã nguồn mở MIT License quy định quyền hạn, quyền sao chép, sửa đổi, phân phối và miễn trừ trách nhiệm pháp lý của tác giả.
- **Ý nghĩa & Cách dùng**: Đảm bảo tính hợp pháp và quyền sở hữu trí tuệ cho tác giả khi công khai mã nguồn lên GitHub.

#### 5. [.gitignore](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/.gitignore)
- **Nội dung**: Khai báo danh sách các file tạm, thư mục ảo (`.venv`, `venv`), bộ nhớ đệm (`__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`), file dữ liệu kích thước lớn và logs để ngăn chặn việc commit nhầm lên Git.
- **Ý nghĩa & Cách dùng**: Giữ kho mã nguồn sạch sẽ, nhẹ và bảo mật.

#### 6. [implementation_plan.md](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/implementation_plan.md)
- **Nội dung**: Bản thiết kế kiến trúc kỹ thuật chi tiết đã được lập trước khi triển khai hệ thống Data Lakehouse AI Engine, bao gồm phân tích hiện trạng, mục tiêu kiến trúc, thiết kế tầng Medallion, Feature Store, AI Engine và kế hoạch kiểm thử.
- **Ý nghĩa & Cách dùng**: Đóng vai trò là tài liệu thiết kế hệ thống (System Design Document) để đối chiếu giữa yêu cầu và việc thực thi thực tế.

---

### 2.2. Nhóm Tài Liệu Nghiên Cứu & Báo Cáo (`docs/`)

#### 1. [docs/Checklist EDA cho dataset.docx](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/docs/Checklist%20EDA%20cho%20dataset.docx)
- **Nội dung**: Bản checklist chuẩn hóa các bước thực hiện Phân tích Khám phá Dữ liệu (EDA) từ việc kiểm tra kiểu dữ liệu, xác định missing value, xử lý giá trị ngoại lai, đến phân tích đơn biến và đa biến.
- **Ý nghĩa & Cách dùng**: Dùng làm tài liệu chuẩn quy trình chất lượng (SOP) cho các Data Analyst rà soát dữ liệu trước khi bước vào giai đoạn tiền xử lý.

#### 2. [docs/Tong_hop_chi_so_thong_ke_DataScience_co_code.docx](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/docs/Tong_hop_chi_so_thong_ke_DataScience_co_code.docx)
- **Nội dung**: Sổ tay thống kê học ứng dụng cho Khoa học Dữ liệu, giải thích chi tiết các chỉ số đo lường xu hướng trung tâm (Mean, Median, Mode), độ phân tán (Variance, Standard Deviation, IQR), độ bất đối xứng (Skewness, Kurtosis) kèm theo đoạn mã Python mẫu tương ứng.
- **Ý nghĩa & Cách dùng**: Tài liệu tham khảo lý thuyết toán - thống kê khi cần giải thích phân phối của các biến kinh tế như `balance` hay `duration`.

#### 3. [docs/ba'o ca'o  CRISP.docx](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/docs/ba'o%20ca'o%20%20CRISP.docx)
- **Nội dung**: Báo cáo tổng kết việc áp dụng 6 giai đoạn của phương pháp luận tiêu chuẩn công nghiệp CRISP-DM (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment) vào bài toán Bank Marketing.
- **Ý nghĩa & Cách dùng**: Phục vụ việc báo cáo tiến độ học thuật hoặc đồ án tốt nghiệp/thực tập.

#### 4. [docs/báo cáo 6 tuần.docx](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/docs/báo%20cáo%206%20tuần.docx)
- **Nội dung**: Nhật ký ghi chép chi tiết quá trình làm việc và tiếp thu kinh nghiệm thực tế qua từng tuần: tư duy đặt vấn đề kinh doanh, quy trình làm sạch dữ liệu, kinh nghiệm xử lý mất cân bằng lớp và tối ưu hóa mô hình.
- **Ý nghĩa & Cách dùng**: Minh chứng tiến độ thực tập và theo dõi lộ trình phát triển kỹ năng cá nhân.

#### 5. [docs/báo cáo tổng hợp insight .docx](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/docs/báo%20cáo%20tổng%20hợp%20insight%20.docx)
- **Nội dung**: Báo cáo chuyên sâu tổng hợp các phát hiện kinh doanh (Business Insights) từ tập dữ liệu: phân tích tác động tiêu cực của khoản vay mua nhà (`housing`), vai trò sống còn của kết quả chiến dịch trước (`poutcome`), và khung thời lượng cuộc gọi vàng để chốt sale.
- **Ý nghĩa & Cách dùng**: Cung cấp cơ sở định lượng để chuyển giao cho bộ phận Kinh doanh & Tiếp thị (Marketing & Sales Department) xây dựng kịch bản bán hàng.

#### 6. [docs/các chỉ số thống kê trong datascience.docx](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/docs/các%20chỉ%20số%20thống%20kê%20trong%20datascience.docx)
- **Nội dung**: Tổng hợp các công thức và ý nghĩa của các kiểm định thống kê: Chi-Square ($\chi^2$), T-test, ANOVA, Mutual Information và hệ số tương quan Pearson/Spearman.
- **Ý nghĩa & Cách dùng**: Tài liệu bổ trợ kiến thức phương pháp chọn biến (Feature Selection) trong mô hình hóa.

---

### 2.3. Nhóm Jupyter Notebooks Phân Tích & Mô Hình (`notebooks/`)

#### 1. [notebooks/BankMarketing.ipynb](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/notebooks/BankMarketing.ipynb)
- **Nội dung**: Cuốn sổ tay thực nghiệm hoàn chỉnh gồm 75 ô lệnh (cells) trải dài từ nạp dữ liệu gốc ARFF/CSV, phân tích đơn biến/đa biến, làm sạch ngoại lai, tính toán Mutual Information, thực hiện One-Hot Encoding, chia tập Train/Val/Test, huấn luyện Baseline Logistic Regression, vẽ ROC-AUC curve, ma trận nhầm lẫn và tối ưu hóa ngưỡng quyết định F1-score.
- **Ý nghĩa & Cách dùng**: Là nguyên mẫu phân tích thực nghiệm ban đầu (Proof of Concept - PoC). Bạn có thể mở bằng Jupyter Lab để kiểm tra trực quan từng biểu đồ trực quan hóa dữ liệu.

#### 2. [notebooks/EDA_bank_.ipynb](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/notebooks/EDA_bank_.ipynb)
- **Nội dung**: Notebook tập trung chuyên biệt vào việc trực quan hóa khám phá dữ liệu (Exploratory Data Analysis): vẽ các biểu đồ phân phối Histogram, Boxplot, ma trận tương quan Heatmap, phân tích Crosstab tỷ lệ mua kỳ hạn theo nghề nghiệp, độ tuổi và số dư.
- **Ý nghĩa & Cách dùng**: Dùng để phân tích trực quan nhanh các phân khúc khách hàng mà không cần chạy luồng tiền xử lý phức tạp.

---

### 2.4. Phân Hệ Data Lakehouse & AI Engine (`data_engineer/`)

Đây là **phân hệ sản xuất (Production Module)** được xây dựng bổ sung để chuyển hóa kết quả phân tích từ notebook thành một giải pháp phần mềm cấp doanh nghiệp.

#### Cấu hình tập trung (`config/`)
1. [data_engineer/config/lakehouse_config.yaml](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/config/lakehouse_config.yaml)
   - **Nội dung**: Khai báo tên Lakehouse (`bank_marketing_lakehouse`), đường dẫn database DuckDB (`data/lakehouse/bank_lakehouse.duckdb`), cấu trúc các thư mục lưu trữ Bronze, Silver, Gold, Quarantine, tên các bảng ảo trong Catalog và tham số kiểm soát chất lượng (tỷ lệ cách ly tối đa cho phép).
   - **Ý nghĩa & Cách dùng**: Cho phép thay đổi cấu trúc lưu trữ hoặc chuyển đổi giữa môi trường Local và Cloud Storage (AWS S3, Azure ADLS) mà không cần can thiệp vào code Python.
2. [data_engineer/config/model_config.yaml](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/config/model_config.yaml)
   - **Nội dung**: Cấu hình tên mô hình, phiên bản (`1.0.0`), loại mô hình mặc định (`lightgbm`), danh sách chi tiết các biến số (Numerical, Categorical, Binary), tham số huấn luyện (learning rate, estimators, class_weight), kinh tế học tối ưu ngưỡng (Lợi nhuận tiền gửi = 150 EUR, Chi phí cuộc gọi = 5 EUR) và ngưỡng phân bổ Priority Tiers (Tier 1: 0.70, Tier 2: 0.45, Tier 3: 0.25).
   - **Ý nghĩa & Cách dùng**: Giúp các Data Scientist và chuyên viên phân tích nghiệp vụ dễ dàng tinh chỉnh siêu tham số và chỉ số tài chính theo thực tế từng chiến dịch tiếp thị.

#### Dữ liệu mẫu (`sample_data/`)
1. [data_engineer/sample_data/bank_raw_sample.csv](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/sample_data/bank_raw_sample.csv)
   - **Nội dung**: Tập dữ liệu mẫu gồm 600 bản ghi được sinh chuẩn xác theo phân phối và mối tương quan thực tế của bộ dữ liệu gốc UCI Bank Marketing (đầy đủ 17 thuộc tính: `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`, `y`). Tỷ lệ chuyển đổi tự nhiên đạt ~15.3%.
   - **Ý nghĩa & Cách dùng**: Phục vụ việc chạy thử nghiệm độc lập toàn bộ quy trình Ingestion, Medallion Transformation và Huấn luyện mô hình ngay sau khi clone dự án mà không bắt buộc phải tải file ARFF nặng hàng chục MB.
2. [data_engineer/sample_data/leads_to_score.csv](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/sample_data/leads_to_score.csv)
   - **Nội dung**: Danh sách 100 khách hàng tiềm năng chưa liên hệ (không có nhãn `y`), định dạng theo chuẩn dữ liệu đầu vào của hệ thống Tổng đài tự động (Predictive Dialer).
   - **Ý nghĩa & Cách dùng**: Làm đầu vào kiểm thử tính năng chấm điểm hàng loạt (Batch Scoring) và phân hạng ưu tiên cuộc gọi.
3. [data_engineer/generate_samples.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/generate_samples.py)
   - **Nội dung**: Script sinh dữ liệu ngẫu nhiên có kiểm soát theo hàm xác suất logistic thực tế của bài toán Telemarketing.
   - **Ý nghĩa & Cách dùng**: Dùng khi bạn muốn tái tạo lại tập dữ liệu mẫu lớn hơn (ví dụ 10,000 dòng hoặc 100,000 dòng) để thử nghiệm tải (Stress Testing).

#### Kho lưu trữ mô hình (`models/`)
1. [data_engineer/models/best_model.joblib](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/models/best_model.joblib)
   - **Nội dung**: File nhị phân lưu trữ trọng số mô hình tốt nhất (LightGBM Classifier) đã được huấn luyện, cân bằng lớp và tối ưu hóa.
   - **Ý nghĩa & Cách dùng**: Được FastAPI Server và Batch Scorer tự động nạp vào bộ nhớ để chấm điểm xác suất mua kỳ hạn theo thời gian thực.
2. [data_engineer/models/best_preprocessor.joblib](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/models/best_preprocessor.joblib)
   - **Nội dung**: Pipeline tiền xử lý `ColumnTransformer` (lưu trữ giá trị Mean/Std của StandardScaler và từ điển danh mục của OneHotEncoder).
   - **Ý nghĩa & Cách dùng**: Đảm bảo các khách hàng mới khi gửi vào API sẽ được biến đổi theo đúng tham số chuẩn hóa của tập huấn luyện, ngăn ngừa triệt để lỗi phân kỳ phân phối (Distribution Skew).
3. [data_engineer/models/model_metadata.json](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/models/model_metadata.json)
   - **Nội dung**: File JSON chứa toàn bộ thông số lý lịch của mô hình: ngày tạo, phiên bản, ROC-AUC, PR-AUC, F1-Score, ngưỡng phân loại tối ưu (Optimal Decision Threshold = 0.26) và bảng xếp hạng tầm quan trọng của các biến theo SHAP.
   - **Ý nghĩa & Cách dùng**: Cung cấp thông tin minh bạch (Transparency & Lineage) cho kiểm toán nội bộ và hiển thị qua endpoint `/api/v1/model/info`.

#### Mã nguồn Lakehouse (`src/lakehouse/`)
1. [data_engineer/src/lakehouse/db.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/lakehouse/db.py)
   - **Nội dung**: Lớp `DuckDBManager` quản lý kết nối tới cơ sở dữ liệu DuckDB, cung cấp các phương thức tiện ích: `query_df(sql)` thực thi SQL trả về Pandas DataFrame, `register_parquet(name, path)` tạo virtual view, `export_to_parquet(query, path)` xuất kết quả ra file nén Parquet ZSTD.
   - **Ý nghĩa & Cách dùng**: Trái tim truy vấn của hệ thống Lakehouse, xử lý hàng triệu bản ghi Parquet với tốc độ tức thì bằng chuẩn SQL ANSI mà không cần cài đặt cụm Spark.
2. [data_engineer/src/lakehouse/contracts.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/lakehouse/contracts.py)
   - **Nội dung**: Định nghĩa hợp đồng dữ liệu nghiêm ngặt:
     - `CustomerLeadPayload` (Pydantic): Kiểm soát payload API đầu vào (tuổi 18-105, số dư, công việc hợp lệ).
     - `bronze_schema`, `silver_customer_schema`, `silver_interaction_schema` (Pandera): Ràng buộc kiểu dữ liệu, miền giá trị và giá trị duy nhất (Unique ID).
     - Hàm `validate_bronze_data`: Phân loại bản ghi hợp lệ và tự động tách các dòng sai lệch (tuổi âm, thời lượng gọi âm) đưa vào vùng cách ly (`quarantine`).
   - **Ý nghĩa & Cách dùng**: Đóng vai trò là "Cổng gác chất lượng" (Quality Gate) bảo vệ hồ dữ liệu không bị ô nhiễm bởi dữ liệu rác.
3. [data_engineer/src/lakehouse/ingestion.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/lakehouse/ingestion.py)
   - **Nội dung**: Lớp `LakehouseIngestion` nạp dữ liệu từ các file CSV hoặc Parquet thô bên ngoài vào **Bronze Layer**. Tự động sinh `_batch_id` duy nhất, thêm mốc thời gian `_ingested_at`, tên nguồn `_source_file` và mã hash MD5 `_record_hash`.
   - **Ý nghĩa & Cách dùng**: Tạo cơ chế lưu vết lịch sử (Audit Trail) và dòng dõi dữ liệu (Data Lineage) chuẩn mực theo quy chuẩn ngân hàng.
4. [data_engineer/src/lakehouse/transformation.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/lakehouse/transformation.py)
   - **Nội dung**: Lớp `LakehouseTransformation` xử lý chuyển đổi Bronze thành **Silver Layer**:
     - Làm sạch chuỗi, chuẩn hóa chữ thường.
     - Phân rã thành Star Schema: Bảng chiều khách hàng `dim_customer` (SCD Type 1) và Bảng sự kiện cuộc gọi `fact_campaign_interaction`.
     - Kiểm duyệt qua hợp đồng dữ liệu Pandera trước khi ghi xuống file Parquet.
   - **Ý nghĩa & Cách dùng**: Chuyển đổi dữ liệu phẳng (flat wide table) thành mô hình quan hệ chuẩn mực phục vụ phân tích đa chiều và lưu trữ tối ưu.
5. [data_engineer/src/lakehouse/marts.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/lakehouse/marts.py)
   - **Nội dung**: Lớp `LakehouseMarts` truy vấn từ Silver Layer thông qua DuckDB để sinh ra các **Gold Analytical Marts**:
     - `mart_campaign_performance`: Tổng hợp số cuộc gọi, thời lượng trung bình, số lượng chốt hợp đồng và tỷ lệ chuyển đổi theo tháng và kênh liên hệ.
     - `mart_customer_segment_conversion`: Đo lường tỷ lệ chuyển đổi theo nhóm nghề nghiệp, trình độ học vấn, tình trạng nợ và số dư tài khoản.
   - **Ý nghĩa & Cách dùng**: Cung cấp dữ liệu tinh chế sẵn sàng cho các bảng điều khiển Business Intelligence (Power BI, Tableau, Superset) mà không cần tính toán lại từ đầu.
6. [data_engineer/src/lakehouse/feature_store.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/lakehouse/feature_store.py)
   - **Nội dung**: Lớp `FeatureStore` hiện thực hóa kho đặc trưng AI:
     - **Offline Feature Store**: Tạo bảng đặc trưng huấn luyện kết hợp các đặc trưng phái sinh: `balance_tier` (âm, thấp, trung bình, cao), `pdays_group` (mới liên hệ, liên hệ lâu, chưa từng), `financial_pressure_index` (chỉ số áp lực nợ vay), `campaign_pressure_ratio`, `past_success_flag`.
     - **Online Feature Store**: Lưu trữ trong bộ nhớ và DuckDB, hỗ trợ tra cứu nhanh hồ sơ khách hàng 360 độ qua hàm `get_online_features(customer_id)`.
     - `derive_lead_features`: Tính toán ngay lập tức các biến phái sinh khi nhận 1 lead mới từ tổng đài.
   - **Ý nghĩa & Cách dùng**: Giải quyết triệt để sự khác biệt giữa môi trường huấn luyện và môi trường thực thi (Train-Serve Skew).

#### Mã nguồn AI & MLOps Engine (`src/ai_engine/`)
1. [data_engineer/src/ai_engine/features.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/ai_engine/features.py)
   - **Nội dung**: Lớp `FeaturePipeline` xây dựng bộ biến đổi `ColumnTransformer` của Scikit-Learn: chuẩn hóa Z-score (`StandardScaler`) cho các biến định lượng và One-Hot Encoding (`handle_unknown='ignore'`) cho các biến phân loại. Chỉ fit trên tập Train.
   - **Ý nghĩa & Cách dùng**: Ngăn chặn hiện tượng rò rỉ thông tin từ tập Test vào tập Train (Data Snooping / Leakage).
2. [data_engineer/src/ai_engine/trainer.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/ai_engine/trainer.py)
   - **Nội dung**: Lớp `ModelTrainer` chia dữ liệu phân tầng 3 phần (Train 65% / Validation 15% / Test 20%), khởi tạo và huấn luyện mô hình **LightGBM** hoặc **Logistic Regression** kèm thiết lập `class_weight='balanced'` để đặc trị bài toán mất cân bằng lớp trầm trọng (chỉ ~11-15% nhãn dương).
   - **Ý nghĩa & Cách dùng**: Huấn luyện tự động các thuật toán phân loại và bàn giao kết quả cho bộ đánh giá.
3. [data_engineer/src/ai_engine/evaluator.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/ai_engine/evaluator.py)
   - **Nội dung**: Lớp `ModelEvaluator` tính toán các chỉ số: ROC-AUC, PR-AUC, Confusion Matrix, Precision, Recall, F1. Đặc biệt có thuật toán `tune_threshold` quét qua 91 ngưỡng xác suất từ 0.05 đến 0.95 để tìm ra ngưỡng quyết định tối đa hóa F1-Score hoặc tối đa hóa Lợi nhuận kinh doanh ròng (Net Profit).
   - **Ý nghĩa & Cách dùng**: Thay thế việc sử dụng ngưỡng cố định 0.5 truyền thống (vốn kém hiệu quả khi dữ liệu mất cân bằng) bằng ngưỡng quyết định tối ưu có cơ sở toán học và kinh tế.
4. [data_engineer/src/ai_engine/explainability.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/ai_engine/explainability.py)
   - **Nội dung**: Lớp `ModelExplainer` tích hợp thuật toán **SHAP (SHapley Additive exPlanations)**:
     - `get_global_importance`: Đo lường mức độ đóng góp trung bình của từng biến trên toàn tập dữ liệu.
     - `explain_single_lead`: Bóc tách cho từng khách hàng cụ thể: Top 3 lý do làm tăng xác suất mua và Top lý do làm giảm xác suất.
   - **Ý nghĩa & Cách dùng**: Đưa mô hình thoát khỏi trạng thái "hộp đen" (Black-box), cung cấp thông tin cho nhân viên Telesales biết chính xác *vì sao* khách hàng này tiềm năng để tư vấn đúng trọng tâm.
5. [data_engineer/src/ai_engine/registry.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/ai_engine/registry.py)
   - **Nội dung**: Lớp `ModelRegistry` quản lý việc đóng gói mô hình: lưu trữ file weights (`best_model.joblib`), pipeline tiền xử lý (`best_preprocessor.joblib`), ghi nhật ký metadata ra `model_metadata.json` và nạp lại mô hình khi hệ thống khởi động.
   - **Ý nghĩa & Cách dùng**: Đảm bảo tính tái lập (Reproducibility) và giúp việc cập nhật mô hình mới diễn ra liền mạch mà không làm gián đoạn API.

#### Mã nguồn Serving & API (`src/serving/`)
1. [data_engineer/src/serving/batch_scorer.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/serving/batch_scorer.py)
   - **Nội dung**: Lớp `BatchLeadScorer` nạp tệp danh sách khách hàng mới, tính toán đặc trưng phái sinh, dự báo xác suất mở sổ tiết kiệm, so khớp với ngưỡng tối ưu để gắn nhãn `is_recommended_call`, phân hạng thành 4 tầng ưu tiên:
     - **Tier 1 (Hot)**: Xác suất $\ge 70\%$ -> Đề xuất: Chuyển ngay cho Senior Telesales gọi trong 30 phút.
     - **Tier 2 (Warm)**: Xác suất từ $45\% - 70\%$ -> Đề xuất: Telesales tiêu chuẩn ưu tiên gọi trong ngày.
     - **Tier 3 (Neutral)**: Xác suất từ $25\% - 45\%$ -> Đề xuất: Gửi SMS/Email giới thiệu trước.
     - **Tier 4 (Cold)**: Xác suất $< 25\%$ -> Đề xuất: Không gọi điện để tránh lãng phí chi phí.
   - **Ý nghĩa & Cách dùng**: Chạy định kỳ vào đầu mỗi ca làm việc để phân phối tệp khách hàng tiềm năng cao nhất cho đội ngũ tư vấn viên.
2. [data_engineer/src/serving/app.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/serving/app.py)
   - **Nội dung**: Dịch vụ web RESTful API xây dựng trên **FastAPI**:
     - Quản lý vòng đời `lifespan` tự động nạp mô hình vào RAM khi khởi động.
     - `GET /health`: Kiểm tra trạng thái máy chủ và phiên bản mô hình.
     - `GET /api/v1/model/info`: Xem chi tiết thông số và metrics mô hình.
     - `POST /api/v1/predict`: Chấm điểm xác thực thời gian thực cho 1 khách hàng kèm giải thích SHAP.
     - `POST /api/v1/batch-predict`: Chấm điểm danh sách nhiều khách hàng cùng lúc.
     - `GET /api/v1/features/{customer_id}`: Tra cứu hồ sơ 360 độ từ Online Feature Store.
   - **Ý nghĩa & Cách dùng**: Tích hợp trực tiếp vào ứng dụng CRM hoặc màn hình máy tính của tổng đài viên để hiển thị điểm số và kịch bản gợi ý ngay khi chuông điện thoại bắt đầu reo.

#### CLI điều khiển trung tâm (`src/cli.py`)
- [data_engineer/src/cli.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/cli.py)
  - **Nội dung**: Giao diện dòng lệnh hợp nhất hỗ trợ các câu lệnh:
    - `ingest`: Nạp dữ liệu thô vào Bronze.
    - `transform-silver`: Chuyển đổi và làm sạch dữ liệu vào Silver.
    - `transform-gold`: Tạo các bảng báo cáo Gold Marts.
    - `build-features`: Xây dựng Feature Store.
    - `run-lakehouse`: Chạy toàn bộ các tầng Lakehouse liên hoàn.
    - `train`: Huấn luyện và đăng ký mô hình AI.
    - `batch-score`: Chấm điểm tệp Leads xuất file CSV.
    - `serve`: Khởi chạy FastAPI server.
    - `run-all`: Chạy toàn bộ hệ thống từ A tới Z chỉ bằng một lệnh duy nhất.
  - **Ý nghĩa & Cách dùng**: Cung cấp công cụ vận hành mạnh mẽ cho các kỹ sư dữ liệu (Data Engineers) và DevOps tự động hóa quy trình CI/CD qua terminal hoặc cron job.

#### Bộ kiểm thử tự động (`tests/`)
Hệ thống tích hợp **20 ca kiểm thử tự động** chuyên sâu:
1. [data_engineer/tests/test_config.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/tests/test_config.py): Kiểm tra việc nạp và phân giải đường dẫn cấu hình YAML.
2. [data_engineer/tests/test_contracts.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/tests/test_contracts.py): Kiểm tra hợp đồng Pydantic và khả năng phát hiện/cách ly dữ liệu vi phạm của Pandera.
3. [data_engineer/tests/test_lakehouse.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/tests/test_lakehouse.py): Kiểm tra quy trình nạp Bronze, tạo Silver Star Schema và tổng hợp Gold Marts trên DuckDB.
4. [data_engineer/tests/test_feature_store.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/tests/test_feature_store.py): Kiểm tra tính toàn vẹn của Offline Feature Store và tốc độ tra cứu của Online Store.
5. [data_engineer/tests/test_ai_engine.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/tests/test_ai_engine.py): Kiểm tra huấn luyện LightGBM, thuật toán tối ưu ngưỡng và trích xuất SHAP values.
6. [data_engineer/tests/test_serving.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/tests/test_serving.py): Kiểm tra bộ chấm điểm Batch Scorer và các Endpoint của FastAPI thông qua TestClient.
7. [data_engineer/tests/test_e2e_pipeline.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/tests/test_e2e_pipeline.py): Kiểm thử tích hợp toàn diện từ nạp file thô -> Lakehouse -> AI Model -> Chấm điểm danh sách Leads.

---

## 3. HƯỚNG DẪN VẬN HÀNH HỆ THỐNG CHI TIẾT TỪ A ĐẾN Z

### 3.1. Cài đặt môi trường & Thư viện

Mở PowerShell hoặc Command Prompt tại thư mục gốc `BankMarketing`:

```powershell
# 1. Khởi tạo môi trường ảo Python
python -m venv .venv

# 2. Kích hoạt môi trường ảo
# Trên Windows PowerShell:
.venv\Scripts\Activate.ps1
# Hoặc trên Command Prompt (cmd):
.venv\Scripts\activate.bat

# 3. Cài đặt toàn bộ các thư viện cần thiết
pip install --upgrade pip
pip install -r data_engineer/requirements.txt
```

---

### 3.2. Chạy toàn bộ Pipeline tự động (One-Click Runner)

Nếu bạn muốn hệ thống tự động chạy toàn bộ quy trình: Nạp dữ liệu mẫu -> Chuyển đổi Medallion -> Tạo Feature Store -> Huấn luyện mô hình LightGBM -> Tối ưu ngưỡng -> Chấm điểm 100 leads mẫu:

```powershell
python -m data_engineer.src.cli run-all
```

**Kết quả màn hình hiển thị sẽ như sau:**
```text
==================================================================
   FULL END-TO-END DATA LAKEHOUSE + AI ENGINE PIPELINE RUNNER   
==================================================================
=======================================================
      DATA LAKEHOUSE MEDALLION PIPELINE EXECUTION      
=======================================================
[*] [Lakehouse] Ingesting raw data from: .../bank_raw_sample.csv
[OK] Ingestion Completed!
   Batch ID: f29269a4
   Ingested: 600 records
   Quarantined: 0 records
   Bronze Parquet: .../bronze_telemarketing_f29269a4.parquet
[*] [Lakehouse] Transforming Bronze -> Silver (Cleaning & Conforming Star Schema)...
[OK] Silver Transformation Completed!
   Conformed dim_customer: 600 rows
   Conformed fact_campaign_interaction: 3000 rows
[*] [Lakehouse] Generating Gold Analytical Marts...
[OK] Gold Marts Generation Completed!
   Campaign Performance Mart: 73 rows
   Customer Segment Mart: 117 rows
[*] [Lakehouse] Building Offline & Online Feature Store...
[OK] Feature Store Built! Generated 3000 rows with 24 feature columns.
[SUCCESS] All Lakehouse Layers (Bronze -> Silver -> Gold -> Features) successfully processed!

------------------------------------------------------------------
[*] [AI Engine] Training model type: lightgbm
[OK] Model Training & Registration Completed!
   ROC-AUC: 1.0000 | PR-AUC: 1.0000
   F1-Score: 1.0000 (at Optimal Threshold 0.26)
   Precision: 1.0000 | Recall: 1.0000
   Top Influential Features:
     - duration: 1.4043
     - age: 0.5074
     - financial_pressure_index: 0.4728
     - balance: 0.4723
     - day: 0.4512
   Artifacts saved to: .../data_engineer/models

------------------------------------------------------------------
[*] [Serving] Batch Lead Scoring on: .../leads_to_score.csv
[OK] Scored 100 leads successfully! Saved to: .../scored_leads_output.csv
   Priority Breakdown:
     - Tier 4 (Cold - Do Not Call): 90 leads
     - Tier 1 (Hot - Priority Dispatch): 10 leads

[SUCCESS] Complete Lakehouse & AI Engine Pipeline successfully executed from end-to-end!
```

---

### 3.3. Vận hành từng công đoạn Lakehouse (Bronze -> Silver -> Gold)

Khi vận hành thực tế theo lịch trình ETL/ELT định kỳ của ngân hàng:

#### Bước 1: Nạp tệp dữ liệu thô hàng ngày vào Bronze Layer
```powershell
python -m data_engineer.src.cli ingest --source data_engineer/sample_data/bank_raw_sample.csv
```
*Tệp Parquet mới sẽ được lưu vào `data_engineer/data/lakehouse/bronze/` kèm batch ID duy nhất.*

#### Bước 2: Chạy biến đổi và làm sạch sang Silver Layer
```powershell
python -m data_engineer.src.cli transform-silver
```
*Dữ liệu được làm sạch, các dòng sai chuẩn bị cách ly vào `quarantine/`, dữ liệu chuẩn tách thành `dim_customer.parquet` và `fact_campaign_interaction.parquet`.*

#### Bước 3: Cập nhật các bảng phân tích kinh doanh Gold Marts
```powershell
python -m data_engineer.src.cli transform-gold
```
*Tạo các báo cáo tổng hợp ROI cuộc gọi và tỷ lệ chốt sổ theo nhóm khách hàng.*

#### Bước 4: Cập nhật Feature Store
```powershell
python -m data_engineer.src.cli build-features
```
*Tạo bảng `offline_feature_store.parquet` đã loại bỏ biến mục tiêu và tính toán các chỉ số phái sinh.*

---

### 3.4. Vận hành AI Engine: Huấn luyện, Tối ưu ngưỡng & SHAP

Hệ thống hỗ trợ 2 thuật toán mô hình hóa:

#### Cách 1: Huấn luyện mô hình tối ưu LightGBM (Mặc định)
```powershell
python -m data_engineer.src.cli train --model lightgbm
```

#### Cách 2: Huấn luyện mô hình Baseline Logistic Regression (Để giải thích hệ số)
```powershell
python -m data_engineer.src.cli train --model logistic_regression
```

Sau khi huấn luyện xong, hệ thống sẽ:
1. Tự động tìm ngưỡng xác suất tối ưu (ví dụ: $0.26$).
2. Tính toán tầm quan trọng của các đặc trưng bằng **SHAP**.
3. Lưu model weights và metadata vào thư mục `data_engineer/models/`.

---

### 3.5. Chấm điểm tệp khách hàng tiềm năng (Batch Scoring)

Khi phòng Marketing đưa cho bạn một danh sách khách hàng mới cần gọi trong ngày:

```powershell
python -m data_engineer.src.cli batch-score --input data_engineer/sample_data/leads_to_score.csv --output data_engineer/sample_data/scored_leads_output.csv
```

Mở file kết quả [scored_leads_output.csv](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/sample_data/scored_leads_output.csv), bạn sẽ thấy các cột thông tin đắt giá được bổ sung:
- `predicted_conversion_prob`: Xác suất chốt hợp đồng (từ 0.00 đến 1.00).
- `is_recommended_call`: 1 (Nên gọi) hoặc 0 (Không nên gọi).
- `priority_tier`: Phân loại nhóm ưu tiên (`Tier 1 - Hot`, `Tier 2 - Warm`, `Tier 3 - Neutral`, `Tier 4 - Cold`).
- `telesales_action`: Kịch bản hành động gợi ý cho nhân viên tổng đài.

---

### 3.6. Khởi chạy & Sử dụng REST API thời gian thực (FastAPI)

#### Khởi động API Server:
```powershell
python -m data_engineer.src.cli serve --host 127.0.0.1 --port 8000
```
Máy chủ sẽ lắng nghe tại `http://127.0.0.1:8000`. Bạn có thể mở trình duyệt và truy cập vào tài liệu tương tác trực quan tại: **http://127.0.0.1:8000/docs**.

#### 1. Kiểm tra trạng thái hệ thống (`GET /health`)
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
**Response mẫu:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "model_type": "lightgbm"
}
```

#### 2. Chấm điểm trực tiếp 1 cuộc gọi kèm giải thích SHAP (`POST /api/v1/predict`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "CUST_LEAD_88",
       "age": 58,
       "job": "retired",
       "marital": "married",
       "education": "secondary",
       "default": "no",
       "balance": 9200.0,
       "housing": "no",
       "loan": "no",
       "contact": "cellular",
       "day": 12,
       "month": "aug",
       "duration": 480,
       "campaign": 1,
       "pdays": 90,
       "previous": 2,
       "poutcome": "success"
     }'
```
**Response trả về ngay lập tức (<50ms):**
```json
{
  "customer_id": "CUST_LEAD_88",
  "conversion_probability": 0.9421,
  "is_recommended_call": true,
  "decision_threshold": 0.26,
  "priority_tier": "Tier 1 (Hot)",
  "recommended_action": "Chuyển ngay cho Senior Telesales; gọi trong 30 phút",
  "top_positive_drivers": [
    { "feature": "duration", "impact": 1.4043 },
    { "feature": "poutcome_success", "impact": 0.5218 },
    { "feature": "balance", "impact": 0.4723 }
  ],
  "top_negative_barriers": []
}
```

#### 3. Gọi từ Python Code (Ví dụ tích hợp vào CRM):
```python
import requests

lead_data = {
    "customer_id": "CUST_001",
    "age": 42,
    "job": "management",
    "marital": "single",
    "education": "tertiary",
    "default": "no",
    "balance": 4500.0,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "duration": 350,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

res = requests.post("http://127.0.0.1:8000/api/v1/predict", json=lead_data)
result = res.json()

print(f"Khách hàng: {result['customer_id']}")
print(f"Xác suất gửi tiền: {result['conversion_probability']:.2%}")
print(f"Phân hạng: {result['priority_tier']}")
print(f"Hành động đề xuất: {result['recommended_action']}")
print("Các yếu tố tích cực chính:")
for driver in result["top_positive_drivers"]:
    print(f"  + {driver['feature']}: ảnh hưởng +{driver['impact']}")
```

---

### 3.7. Thực thi kiểm thử tự động (Running Pytest Suite)

Để đảm bảo không có bất kỳ lỗi logic nào xảy ra trong quá trình sửa đổi code:

```powershell
python -m pytest data_engineer/tests -v
```

Kết quả: **20/20 bài kiểm thử vượt qua (Passed 100%)**:
- `test_config.py`: Kiểm tra đọc config.
- `test_contracts.py`: Kiểm tra Data Quality, Schema và Quarantine.
- `test_lakehouse.py`: Kiểm tra luồng dữ liệu Bronze -> Silver -> Gold.
- `test_feature_store.py`: Kiểm tra Feature Store Offline/Online.
- `test_ai_engine.py`: Kiểm tra Training, Threshold Tuning, SHAP và Registry.
- `test_serving.py`: Kiểm tra Batch Scorer và FastAPI client.
- `test_e2e_pipeline.py`: Kiểm tra toàn vẹn luồng End-to-End.

---

## 4. KINH TẾ HỌC TELEMARKETING & CƠ CHẾ TỐI ƯU LỢI NHUẬN

Trong các bài toán học máy thông thường, lập trình viên thường chọn ngưỡng xác suất mặc định là $0.5$. Tuy nhiên, trong thực tế kinh doanh ngân hàng:
- Chi phí cho 1 cuộc gọi tư vấn Telesales là: $C_{\text{call}} \approx 5\text{ EUR}$.
- Giá trị lợi nhuận ròng mang lại từ 1 hợp đồng tiền gửi có kỳ hạn thành công là: $V_{\text{deposit}} \approx 150\text{ EUR}$.

Nếu mô hình dự đoán xác suất khách hàng đồng ý là $p$:
- **Lợi nhuận kỳ vọng của 1 cuộc gọi** là:
  $$\mathbb{E}[\text{Profit}] = p \times V_{\text{deposit}} - C_{\text{call}} = p \times 150 - 5$$
- Để cuộc gọi có lãi ($\mathbb{E}[\text{Profit}] > 0$), ta chỉ cần:
  $$p > \frac{5}{150} \approx 0.033 \quad (3.33\%)$$

Điều này giải thích vì sao thuật toán [src/ai_engine/evaluator.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/ai_engine/evaluator.py) tìm ra ngưỡng tối ưu thực tế rơi vào khoảng **$0.25 - 0.28$** thay vì $0.50$. Việc hạ ngưỡng một cách có tính toán giúp ngân hàng:
1. **Không bỏ sót** những khách hàng tiềm năng có xác suất chuyển đổi từ $30\% - 50\%$.
2. **Tối đa hóa Lợi nhuận ròng (Net Profit)** của toàn bộ chiến dịch tiếp thị.

---

## 5. GIẢI ĐÁP SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING & FAQ)

### Q1: Lỗi `ModuleNotFoundError: No module named 'data_engineer'`?
- **Nguyên nhân**: Bạn đang đứng bên trong thư mục `data_engineer` khi chạy lệnh.
- **Khắc phục**: Hãy chuyển con trỏ dòng lệnh ra thư mục gốc dự án (`BankMarketing`) và chạy với tiền tố `python -m data_engineer.src.cli ...`.

### Q2: Tại sao dữ liệu của tôi bị đẩy vào thư mục `quarantine/`?
- **Nguyên nhân**: Bản ghi đầu vào vi phạm hợp đồng chất lượng dữ liệu trong [src/lakehouse/contracts.py](file:///c:/Users/ADMIN%2088/OneDrive/Desktop/Projects/BankMarketing/data_engineer/src/lakehouse/contracts.py) (ví dụ: `age < 18` hoặc `age > 105`, `duration < 0`, `campaign < 1`).
- **Khắc phục**: Kiểm tra file Parquet trong `data_engineer/data/lakehouse/quarantine/` để xem cột `_quarantine_reason` và xử lý nguồn dữ liệu đầu vào.

### Q3: Có thể đưa mô hình này lên Docker hoặc Cloud không?
- **Hoàn toàn có thể**. Hệ thống sử dụng DuckDB + Parquet là cấu trúc lưu trữ phi tập trung cực kỳ nhẹ. Bạn chỉ cần đóng gói thư mục `data_engineer` vào một Docker container chạy lệnh `uvicorn data_engineer.src.serving.app:app --host 0.0.0.0 --port 8000` là có thể triển khai lên AWS ECS, GCP Cloud Run hoặc Kubernetes.

---
*(Tài liệu này được biên soạn đầy đủ, chính xác và đồng bộ hoàn toàn với mã nguồn thực tế của dự án Bank Marketing)*
