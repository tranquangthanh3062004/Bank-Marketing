# 🏦 Bank Marketing Term Deposit Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20Imbalanced--Learn%20%7C%20SHAP-orange.svg)](https://scikit-learn.org/)
[![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-green.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dự án phân tích dữ liệu chuyên sâu (EDA) và xây dựng mô hình Học máy (Machine Learning) nhằm **dự đoán khả năng khách hàng đăng ký gửi tiền tiết kiệm có kỳ hạn (Term Deposit)** trong các chiến dịch Telemarketing của ngân hàng. Dự án được triển khai bài bản theo chuẩn quy trình **CRISP-DM**.

---

## 📌 Mục Lục
- [1. Bối Cảnh & Bài Toán Kinh Doanh](#1-bối-cảnh--bài-toán-kinh-doanh)
- [2. Bộ Dữ Liệu (Dataset Overview)](#2-bộ-dữ-liệu-dataset-overview)
- [3. Quy Trình Triển Khai (CRISP-DM Pipeline)](#3-quy-trình-triển-khai-crisp-dm-pipeline)
- [4. Kết Quả & Insight Kinh Doanh](#4-kết-quả--insight-kinh-doanh)
- [5. Cấu Trúc Thư Mục (Project Structure)](#5-cấu-trúc-thư-mục-project-structure)
- [6. Hướng Dẫn Cài Đặt & Chạy](#6-hướng-dẫn-cài-đặt--chạy)
- [7. Đóng Góp & Bản Quyền](#7-đóng-góp--bản-quyền)

---

## 1. Bối Cảnh & Bài Toán Kinh Doanh

### 1.1. Business Problem
Trong các chiến dịch tiếp thị qua điện thoại (Telemarketing), tỷ lệ khách hàng đồng ý gửi tiết kiệm kỳ hạn thường rất thấp (~11.7%). Việc gọi ngẫu nhiên hoặc dàn trải dẫn đến:
- Lãng phí chi phí vận hành và thời gian của đội ngũ tư vấn viên (Telesales).
- Gây phiền toái cho khách hàng không có nhu cầu, ảnh hưởng đến trải nghiệm thương hiệu.

### 1.2. Mục Tiêu Dự Án (Project Objectives)
- Xây dựng mô hình phân loại nhị phân (**Binary Classification**) để chấm điểm xác suất chuyển đổi của từng khách hàng.
- Xếp hạng và ưu tiên tệp khách hàng tiềm năng cao (Lead Scoring / Prioritization).
- Phân tích nguyên nhân gốc rễ và mức độ ảnh hưởng của từng đặc trưng kinh tế - xã hội đến quyết định gửi tiền thông qua **SHAP & Feature Importance**.

### 1.3. Success Criteria
- **Kỹ thuật**: Mô hình đạt diện tích dưới đường cong ROC-AUC cao, tối ưu hóa $F_1\text{-score}$ và cân bằng Precision/Recall thông qua việc điều chỉnh ngưỡng phân loại (Threshold Tuning).
- **Kinh doanh**: Tăng tỷ lệ chuyển đổi (Conversion Rate) trên mỗi 100 cuộc gọi, giảm thiểu số lượng cuộc gọi không hiệu quả.

---

## 2. Bộ Dữ Liệu (Dataset Overview)

Dữ liệu được thu thập từ các chiến dịch tiếp thị trực tiếp của một định chế tài chính ngân hàng (UCI Bank Marketing Dataset).

| STT | Tên Biến | Nhóm | Kiểu Dữ Liệu | Ý Nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `age` | Nhân khẩu học | Numerical | Độ tuổi khách hàng |
| 2 | `job` | Nhân khẩu học | Categorical | Nghề nghiệp (`admin.`, `technician`, `services`, `management`,...) |
| 3 | `marital` | Nhân khẩu học | Categorical | Tình trạng hôn nhân (`married`, `single`, `divorced`) |
| 4 | `education` | Nhân khẩu học | Categorical | Học vấn (`primary`, `secondary`, `tertiary`, `unknown`) |
| 5 | `default` | Tài chính cá nhân | Binary | Có nợ xấu/vỡ nợ tín dụng hay không (`yes`, `no`) |
| 6 | `balance` | Tài chính cá nhân | Numerical | Số dư bình quân năm trong tài khoản (đơn vị: Euro) |
| 7 | `housing` | Tài chính cá nhân | Binary | Có khoản vay mua nhà hay không (`yes`, `no`) |
| 8 | `loan` | Tài chính cá nhân | Binary | Có khoản vay tiêu dùng cá nhân hay không (`yes`, `no`) |
| 9 | `contact` | Chiến dịch | Categorical | Kênh liên lạc (`cellular`, `telephone`, `unknown`) |
| 10 | `day` | Chiến dịch | Numerical | Ngày liên lạc cuối cùng trong tháng |
| 11 | `month` | Chiến dịch | Categorical | Tháng liên lạc cuối cùng (`jan`, `feb`,..., `dec`) |
| 12 | `duration` | Chiến dịch | Numerical | Thời lượng cuộc gọi gần nhất (giây) |
| 13 | `campaign` | Chiến dịch | Numerical | Số lần liên lạc với khách hàng trong chiến dịch này |
| 14 | `pdays` | Chiến dịch trước | Numerical | Số ngày trôi qua từ lần liên lạc trước (`-1`: chưa từng liên lạc) |
| 15 | `previous` | Chiến dịch trước | Numerical | Số lần liên lạc trước chiến dịch hiện tại |
| 16 | `poutcome` | Chiến dịch trước | Categorical | Kết quả chiến dịch trước (`success`, `failure`, `unknown`, `other`) |
| **Target** | **`y`** | **Mục tiêu** | **Binary** | **Khách hàng có đăng ký gửi tiết kiệm không? (`yes`/`no`)** |

---

## 3. Quy Trình Triển Khai (CRISP-DM Pipeline)

```mermaid
flowchart LR
    A[1. Business Understanding] --> B[2. Data Understanding & EDA]
    B --> C[3. Data Preparation & Engineering]
    C --> D[4. Modeling & Resampling]
    D --> E[5. Evaluation & Explainability]
    E --> F[6. Business Deployment & Actionable Insights]
```

### Bước 1: Data Understanding & Khám phá dữ liệu (EDA)
- **Kiểm tra chất lượng dữ liệu**: Thống kê số lượng bản ghi, kiểu dữ liệu, các giá trị `unknown` (tập trung ở `poutcome`, `contact`, `education`, `job`) và khẳng định không có dòng trùng lặp (duplicates).
- **Phân tích mất cân bằng lớp (Class Imbalance)**: Lớp dương (`y = yes`) chỉ chiếm ~11.7%, lớp âm (`y = no`) chiếm ~88.3%.
- **Phân tích đơn biến & đa biến**: 
  - Đánh giá phân phối phân vị của các biến định lượng (`balance`, `duration`, `age`, `campaign`).
  - Kiểm định Chi-Square ($\chi^2$) và phân tích Crosstab kiểm tra tương quan giữa từng biến định tính với biến mục tiêu.

### Bước 2: Data Preparation & Tiền xử lý
- **Làm sạch dữ liệu**: Xử lý các giá trị khuyết thiếu/unknown hợp lý, phát hiện các bản ghi ngoại lai (outliers).
- **Mã hóa đặc trưng (Encoding)**: Áp dụng One-Hot Encoding cho các biến phân loại danh nghĩa và Binary Encoding cho các biến nhị phân.
- **Lựa chọn đặc trưng (Feature Selection)**:
  - Tính điểm **Mutual Information (MI)** và phân tích ma trận tương quan để loại bỏ các đặc trưng nhiễu hoặc có điểm thông tin bằng 0.
  - Ngăn ngừa rò rỉ dữ liệu (**Data Leakage**): Kiểm tra và cô lập biến mục tiêu trước khi đưa vào pipeline.
- **Phân chia tập dữ liệu**: Phân tách dữ liệu thành `Train / Validation / Test` theo tỷ lệ chuẩn.
- **Chuẩn hóa & Xử lý mất cân bằng**:
  - Áp dụng `StandardScaler` trên tập huấn luyện (tránh data snooping).
  - Ứng dụng kỹ thuật `SMOTE` (Synthetic Minority Over-sampling Technique) để cân bằng mẫu lớp thiểu số trên tập Train.

### Bước 3: Modeling & Đánh giá mô hình
- **Mô hình thử nghiệm**: Logistic Regression Baseline (với khả năng diễn giải trọng số Coefficients trực quan).
- **Đánh giá hiệu năng**:
  - Đo lường diện tích dưới đường cong **ROC-AUC** trên tập Validation.
  - Xây dựng ma trận nhầm lẫn (**Confusion Matrix**) chi tiết dạng tần số và tỷ lệ phần trăm.
- **Tối ưu hóa ngưỡng quyết định (Threshold Tuning)**:
  - Phân tích đường cong **Precision-Recall Trade-off**.
  - Tìm kiếm ngưỡng tối ưu giúp cực đại hóa điểm số $F_1\text{-score}$ để cân bằng giữa việc bỏ sót khách hàng tiềm năng (False Negatives) và gọi nhầm khách hàng không có nhu cầu (False Positives).

### Bước 4: Giải thích mô hình (Model Explainability)
- Ứng dụng **SHAP (SHapley Additive exPlanations)** để giải thích tầm quan trọng toàn cục (Global Importance) và cục bộ (Local Impact) của từng biến đầu vào.

---

## 4. Kết Quả & Insight Kinh Doanh

| Yếu Tố | Insight Thực Tế | Khuyến Nghị Hành Động (Actionable Recommendations) |
| :--- | :--- | :--- |
| **Duration (Thời lượng cuộc gọi)** | Là biến có sức mạnh dự báo lớn nhất. Thời lượng gọi càng dài tỷ lệ chuyển đổi càng tăng mạnh. | Đào tạo kịch bản telesales giữ chân khách hàng trong 3 phút đầu; phân loại khách hàng có dấu hiệu quan tâm để chăm sóc sâu. |
| **Poutcome (Chiến dịch trước)** | Nhóm khách hàng từng có kết quả `success` ở chiến dịch trước đạt tỷ lệ chuyển đổi **> 60%**. | Ưu tiên gọi ngay cho nhóm khách hàng cũ đã từng phản hồi tích cực trong các chiến dịch trước. |
| **Housing Loan (Vay mua nhà)** | Khách hàng đang có khoản vay mua nhà có xu hướng **ít gửi tiết kiệm hơn** đáng kể (tác động âm). | Hạn chế chào mời sản phẩm tiết kiệm kỳ hạn lớn cho nhóm đang chịu gánh nặng nợ vay mua nhà; thay vào đó giới thiệu các gói quản lý dòng tiền hoặc bảo hiểm. |
| **Contact (Kênh liên lạc)** | Kênh di động (`cellular`) đem lại tỷ lệ phản hồi và thành công cao hơn rõ rệt so với điện thoại cố định (`telephone`) hoặc `unknown`. | Chuẩn hóa cơ sở dữ liệu số điện thoại di động của khách hàng; loại bỏ các đầu số cố định hoặc không xác thực. |
| **Balance (Số dư tài khoản)** | Khách hàng có số dư tích lũy ổn định và cao có tỷ lệ đồng ý mở sổ tiết kiệm cao hơn. | Thiết lập bộ lọc số dư tối thiểu trong phân khúc khách hàng mục tiêu trước khi bàn giao data cho Telesales. |

---

## 5. Cấu Trúc Thư Mục (Project Structure)

```text
BankMarketing/
├── .gitignore               # Cấu hình bỏ qua các file tạm, môi trường ảo, cache
├── pyproject.toml           # Cấu hình gói và khai báo dependencies chính
├── requirements.txt         # Danh sách thư viện Python chi tiết
├── README.md                # Tài liệu tổng quan và hướng dẫn dự án
├── docs/                    # Tài liệu báo cáo, chỉ số thống kê & checklist EDA
├── notebooks/               # Jupyter Notebooks thực nghiệm phân tích & mô hình
├── data_engineer/           # HỆ THỐNG DATA LAKEHOUSE CHUẨN AI ENGINE (Production Standard)
│   ├── config/              # Cấu hình Lakehouse (Bronze, Silver, Gold, DuckDB) & Model
│   ├── data/lakehouse/      # Lưu trữ Parquet phân tầng Medallion & DuckDB catalog
│   ├── sample_data/         # Tệp dữ liệu mẫu chuẩn UCI Bank Marketing
│   ├── models/              # Model Registry lưu trữ LightGBM weights & metadata
│   ├── src/                 # Mã nguồn: lakehouse (ETL/Feature Store), ai_engine, serving (FastAPI)
│   ├── tests/               # Bộ 20 bài test tự động bao phủ toàn diện
│   └── README.md            # Hướng dẫn chi tiết vận hành Data Lakehouse & AI Engine
└── src/                     # Core package ban đầu
    └── __init__.py
```

---

## 6. Hướng Dẫn Cài Đặt & Chạy

### 6.1. Yêu Cầu Hệ Thống
- Python `3.10` trở lên.
- Quản lý môi trường: `venv`, `conda` hoặc `poetry`.

### 6.2. Các Bước Cài Đặt

1. **Clone repository về máy**:
   ```bash
   git clone https://github.com/tranquangthanh3062004/Bank-Marketing.git
   cd Bank-Marketing
   ```

2. **Khởi tạo và kích hoạt môi trường ảo**:
   - *Trên Windows (PowerShell)*:
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - *Trên macOS / Linux*:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Cài đặt các gói phụ thuộc (Dependencies)**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Khởi chạy Jupyter Notebook / JupyterLab**:
   ```bash
   jupyter lab
   ```
   Mở file [`notebooks/BankMarketing.ipynb`](notebooks/BankMarketing.ipynb) để xem toàn bộ quá trình phân tích và huấn luyện mô hình.

---

## 7. Đóng Góp & Bản Quyền

- **Tác giả**: Tran Quang Thanh ([@tranquangthanh3062004](https://github.com/tranquangthanh3062004))
- **Giấy phép**: Phân phối theo giấy phép [MIT License](LICENSE).