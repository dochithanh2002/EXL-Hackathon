# Báo cáo tổng thể: chuyển đổi model M2 từ R sang Python

_Ngày 13/11/2026 · Dự án EXL Hackathon, Challenge 4_

## 1. Tóm tắt

- **Việc chuyển đổi là đúng.** Bản Python do agent sinh ra (`runs/M2/output/m2_taiwan_pd.py`) cho kết quả giống hệt script R gốc. Hệ số lệch tối đa khoảng 1e-13, xác suất vỡ nợ (PD) lệch tối đa khoảng 1e-14 trên toàn bộ 30.000 dòng, và không có quyết định duyệt/từ chối nào bị đổi. Điều này đúng trên **6 cách chia dữ liệu khác nhau** (seed 1, 42, 777, 12345, 99999, 2026), không chỉ một lần chia may mắn.
- **Model R gốc không tái lập được.** Script gốc chia dữ liệu ngẫu nhiên mà không đặt seed, nên mỗi lần chạy cho hệ số và AUC khác nhau. Có hệ số còn đổi dấu (`education2`). Đây là lỗi của model gốc, không phải lỗi chuyển đổi.
- **Bộ agent R→Python** chạy theo 9 bước, có 13 guardrail và một bộ quy tắc chuyển đổi (rulebook), kết thúc bằng cổng phê duyệt của con người. Hiện model M2 đang ở trạng thái **chờ phê duyệt** (`AWAITING_HUMAN_APPROVAL`).
- **Kết luận kiểm định:** model hoạt động tốt (AUC test 0,740, hiệu chỉnh tốt) nhưng **không được duyệt ở dạng hiện tại**, vì dùng trực tiếp giới tính và tình trạng hôn nhân làm biến đầu vào. Biến thể B (bỏ hai biến này) được duyệt có điều kiện và không mất độ chính xác.
- Mọi thứ chạy **hoàn toàn trên máy local** (R 4.6.1, Python 3.13), không cần tài khoản Azure.

## 2. Bối cảnh và model M2

| Hạng mục | Chi tiết |
|---|---|
| Dữ liệu | UCI #350 "Default of Credit Card Clients" (Đài Loan, 4–9/2005), 30.000 khách hàng, tỷ lệ vỡ nợ 22,1%. Bản trên Kaggle (`uciml/...`) là bản sao của bộ UCI này. |
| Nguồn code | Notebook Kaggle `netzone/default-credit-and-credit-score-prediction` (Apache-2.0), lưu tại `legacy_models/m2_taiwan/model.R` và **không được sửa**. |
| Thuật toán | Hồi quy logistic (`stats::glm`, binomial/logit), 11 biến, 15 hệ số. |
| Biến đầu vào | limit_bal, sex, education, marriage, pay_0, pay_2, pay_3, pay_5, pay_6, pay_amt1, pay_amt2 |
| Tiền xử lý | Gộp education 0/4/5/6 → 4; marriage 0 → 3; pay_* giá trị −2/−1/0 → 0 (vòng lặp theo vị trí cột 6..11); chuyển sex, education, marriage, biến mục tiêu thành factor. |
| Chia dữ liệu | `caret::createDataPartition`, 70/30 phân tầng theo biến mục tiêu, **không có seed**. |

## 3. Bộ agent chuyển đổi R → Python (`r2py_agent/`)

### 3.1 Quy trình 9 bước

Mỗi bước có thể chạy lại và tiếp tục từ chỗ dừng. Trạng thái lưu ở `runs/<model_id>/state.json`.

| # | Bước | Việc làm |
|---|---|---|
| 1 | INTAKE | Băm SHA-256 config, rulebook, script R, dung sai và dữ liệu; đóng băng dung sai. |
| 2 | CAPTURE | Chạy script R **nguyên bản** trong harness `r_capture.R` (đặt seed trước khi chạy), xuất hệ số, dự đoán, cách chia và metadata. |
| 3 | SCAN | Quét tĩnh code R + hồ sơ schema dữ liệu, xác định các quy tắc bị kích hoạt (M2: 17 quy tắc). |
| 4 | ANALYSE | LLM phân tích logic R, kết quả được kiểm tra theo schema. |
| 5 | PLAN | LLM lập kế hoạch ánh xạ từng cấu trúc R sang Python, có trích dẫn quy tắc; kiểm tra độ phủ. |
| 6 | GENERATE | LLM sinh module Python theo một giao diện cố định. |
| 7 | TEST & REPAIR | Guardrail tĩnh → chạy cô lập → kiểm tra parity. Nếu lỗi, gửi bằng chứng lại cho LLM sửa, tối đa 3 vòng rồi chuyển cho người. |
| 8 | PACKAGE | Đóng gói module, `requirements.txt` cố định phiên bản, test pytest, báo cáo, `manifest.json` (hash từng file). |
| 9 | HUMAN GATE | Trạng thái chờ phê duyệt cho đến khi có người chạy lệnh `approve`. |

### 3.2 Guardrails

| Mã | Nội dung |
|---|---|
| G-01 | Chỉ được import các package cho phép (M2: numpy, pandas, statsmodels, scipy). |
| G-02 | Cấm: estimator có phạt (sklearn), mọi RNG hoặc chia lại dữ liệu, shell, mạng, eval, ghi file. |
| G-03 | Phải có đủ các hàm giao diện và hằng `TARGET`. |
| G-04 | Cú pháp Python hợp lệ. |
| G-05 | Tên hệ số Python phải trùng khớp tên hệ số R. |
| G-06 | Chạy trong tiến trình cô lập (`python -I`), xoá credential khỏi môi trường, giới hạn thời gian 300 giây. |
| G-07 | Giữ nguyên mọi dòng dữ liệu, đúng thứ tự gốc. |
| G-08 | Tối đa 3 vòng sửa, sau đó chuyển cho người. |
| G-09 | Bắt buộc có người phê duyệt trước khi phát hành. |
| G-10 | Mọi phản hồi LLM được kiểm tra theo schema có kiểu. |
| G-11 | LLM chỉ thấy tên cột, kiểu dữ liệu và tập giá trị, **không bao giờ thấy dòng dữ liệu**. |
| G-12 | Đầu vào bất biến: hash lúc intake được kiểm tra lại trước khi đóng gói. |
| G-13 | Kế hoạch phải trích dẫn mọi quy tắc MUST bị kích hoạt. |

### 3.3 Rulebook và bài học từ lần chạy M2 (v1.1)

`r2py_agent/rulebook.yaml` có 20 quy tắc (đọc file, chỉ số cột, mã hoá lại, factor, RNG, ước lượng, dự đoán, package, quản trị). Từ lần chạy M2:

- **R-EST-02 được sửa.** Lần đầu dùng statsmodels với `tol=1e-12`: hệ số khớp, nhưng sai số chuẩn lệch 7,7e-6 (vượt dung sai 1e-6). Nguyên nhân: R dừng IRLS ở `epsilon=1e-8` và lấy sai số chuẩn từ vòng lặp cuối. Quy tắc mới yêu cầu tái hiện đúng đường đi của `glm.fit`; sau đó sai số chuẩn khớp đến 5e-15.
- **Sửa harness:** module được đăng ký vào `sys.modules` trước khi chạy (lỗi do `@dataclass`, không phải lỗi model).
- `requirements.txt` chỉ ghim các package thực sự được import.

### 3.4 Kết quả lần chạy M2

- LLM: claude-opus-5-5 qua backend `manual`, 3 lần gọi (analyse, plan, generate).
- **Đạt parity ở vòng 1** (giới hạn 3 vòng). 17/17 quy tắc MUST được trích dẫn trong kế hoạch.
- Đầu ra: `runs/M2/output/` gồm `m2_taiwan_pd.py`, `requirements.txt`, `tests/test_parity.py`, `reference/`, `manifest.json`, `MIGRATION_REPORT.md`, `HUMAN_REVIEW.md`.

## 4. Kiểm chứng tính tương đương (parity)

### 4.1 Phương pháp

Vấn đề: script R chia ngẫu nhiên không seed, nên không thể so sánh trực tiếp. Cách giải quyết:

1. **Đóng băng cách chia mà không sửa code gốc.** `r_capture.R` gọi `set.seed(seed)` rồi chạy `model.R` nguyên bản. `createDataPartition` vì thế luôn chọn cùng các dòng.
2. **Lưu kết quả R** với độ chính xác đầy đủ (17 chữ số): `split.csv`, `coefficients.csv`, `predictions.csv`, `meta.json`.
3. **Python fit trên đúng các dòng đó**: đọc `split.csv` thay vì tự chia (guardrail G-02 cấm chia lại).
4. **So sánh** bằng `r2py_agent/parity.py`: tên hệ số, hệ số, sai số chuẩn, tập dòng, biến mục tiêu, cách chia, PD trên mọi dòng, deviance.

Dung sai (đóng băng trong `migration.yaml`, agent không được nới): hệ số và sai số chuẩn 1e-6 (tương đối), PD 1e-8 (tuyệt đối), deviance 1e-6.

Đã kiểm tra thêm: chạy capture hai lần với seed 2026 cho file **giống hệt từng byte**, và giống bản tham chiếu đã lưu.

### 4.2 Kết quả trên 6 seed

Test mới `r2py_agent/tests/test_multiseed_parity.py`: **6/6 đạt** (10,8 giây). Cả bộ test agent: **23/23 đạt**.

| Seed | Kiểm tra | Lệch hệ số tối đa | Lệch SE tối đa | Lệch PD tối đa | AUC test (R) | AUC test (Python) | Quyết định bị đổi (ngưỡng 0,5) |
|---|---|---|---|---|---|---|---|
| 1 | 8/8 | 1,7e-13 | 8,1e-15 | 8,7e-15 | 0,7560815205 | 0,7560815205 | 0 |
| 42 | 8/8 | 2,8e-13 | 9,4e-15 | 5,9e-15 | 0,7718163868 | 0,7718163868 | 0 |
| 777 | 8/8 | 5,5e-14 | 8,7e-15 | 1,1e-14 | 0,7520720667 | 0,7520720667 | 0 |
| 12345 | 8/8 | 1,7e-14 | 5,6e-15 | 3,7e-15 | 0,7578163682 | 0,7578163682 | 0 |
| 99999 | 8/8 | 1,9e-14 | 5,5e-15 | 3,2e-15 | 0,7567433759 | 0,7567433759 | 0 |
| 2026 | 8/8 | 3,0e-14 | 5,0e-15 | 8,8e-15 | 0,7402449543 | 0,7402449184 | 0 |

Mức lệch thực tế nhỏ hơn dung sai hàng triệu lần, gần giới hạn của số thực double.

**Về chênh lệch AUC 3,6e-8 ở seed 2026:** có 556 dòng test trùng điểm với dòng khác (khách hàng có đặc điểm giống nhau). R cho một cặp bằng nhau tuyệt đối, Python lệch ở chữ số thứ 15, nên cách xếp hạng khi tính AUC khác nhau đúng một nửa cặp (1/(số cặp dương×âm)×0,5 = 3,58e-8). Làm tròn PD đến 12 chữ số thì AUC của R và Python đều là **0,7402449542619647**. Đây là sai số làm tròn, không phải lỗi chuyển đổi.

### 4.3 Một nhánh duy nhất: agent sinh, validator kiểm lại

| Bước | So sánh | Ý nghĩa |
|---|---|---|
| Agent (`runs/M2`) | Python ↔ **`model.R` gốc** (chạy nguyên trạng) | Bằng chứng chuyển đổi đúng, trong harness của agent. |
| Validator (`src/m2/parity.py`) | **Module do agent sinh** ↔ reference R đã đóng băng, kiểm hash với manifest | Kiểm lại độc lập bằng code và dung sai riêng, kèm đối chiếu scikit-learn. Kết quả: 14/14 đạt. |

Toàn bộ kiểm định, fairness và monitoring đều chạy trên chính module `runs/M2/output/m2_taiwan_pd.py` (SHA-256 được ghi vào mọi kết quả). Bản rebuild viết tay (`src/m2/rebuild.py`, `run_reference.R`) đã bị bỏ.

## 5. Phát hiện về model R gốc

Chạy `model.R` hai lần liên tiếp (không seed):

| | Lần 1 | Lần 2 |
|---|---|---|
| Hệ số chặn | −1,3014 | −1,3038 |
| sex2 | −0,1334 | −0,1471 |
| education2 | **−0,0205** | **+0,0322** |
| AUC test | 0,7619 | 0,7607 |

Trên 6 seed, `education2` dao động từ −0,039 đến +0,040 và `pay_0` từ 0,864 đến 0,944. Trên 200 seed, AUC test trung bình 0,759 ± 0,005 (khoảng 95%: 0,748–0,769). Seed 2026 (AUC 0,740) là một cách chia khá bất lợi.

Bản Python tái hiện đúng từng giá trị này, kể cả dấu. Muốn đưa vào vận hành thì phải cố định seed hoặc dùng cách chia đã lưu.

## 6. Kiểm định model và công bằng (fairness)

Nguồn: `reports/m2/validation_report_M2.md`, khung PRA SS1/23, Equality Act 2010, FCA Consumer Duty.

| Hạng mục | Đánh giá | Bằng chứng |
|---|---|---|
| Tái lập & bản Python | 🟠 | Parity đạt; code gốc không có seed. |
| Khả năng phân biệt | 🟢 | AUC test 0,740 (CI 95%: 0,727–0,753), Gini 0,480, KS 0,381. |
| Hiệu chỉnh & back-test | 🟢 | PD trung bình 21,9% so với thực tế 22,1%; Hosmer–Lemeshow p = 0,12; 0/10 nhóm bị cảnh báo. |
| Ổn định | 🟠 | PSI train→test 0,002, train→tháng dịch chuyển 0,083; CSI pay_0 = 0,62 do cách mã hoá dữ liệu. |
| Đặc tả model | 🟠 | pay_0 phi tuyến (p < 0,001); mô hình gradient boosting thách thức đạt AUC cao hơn 3,3 điểm. |
| Công bằng về kết quả | 🟢 | Mọi tỷ lệ tác động khác biệt ≥ 0,80: giới tính 0,933, tuổi 0,914, học vấn 0,907, hôn nhân 0,951. |
| Công bằng về pháp lý | 🔴 | **Giới tính và tình trạng hôn nhân là biến đầu vào trực tiếp.** |

**Các phát hiện chính:**

- **F1 (Cao):** dùng giới tính (sex2 = −0,17) và hôn nhân (marriage2 = −0,25) làm biến đầu vào. Khuyến nghị dùng biến thể B (bỏ hai biến này): AUC tăng 0,0015, tỷ lệ tác động theo giới cải thiện 0,024.
- **F2 (Trung bình):** mã không có tài liệu (education 0/5/6, marriage 0) bị gộp vào "khác", tạo hệ số giả education4 = −1,18.
- **F3 (Trung bình):** mã trạng thái trả nợ khác nhau giữa các tháng (mã 1 xuất hiện 3.688 lần ở PAY_0 nhưng gần như không có ở PAY_2–PAY_6).
- **F4 (Trung bình):** không tái lập được (không seed); ngưỡng cắt được tối ưu trên tập test. Cách chia đã được đóng băng.
- **F5 (Trung bình):** model thiếu đặc tả; nên cân nhắc phân nhóm/WoE cho các biến trễ hạn.
- **F6 (Thấp):** các biến còn lại dự đoán được tuổi ≤ 25 (AUC 0,78); nhóm 46–55 bị dự đoán thấp hơn 2,3 điểm phần trăm ở biến thể B.
- **F7 (Thấp):** dữ liệu Đài Loan 2005, không đại diện cho danh mục tại Anh.

**Kết luận kiểm định:** 🔴 không duyệt ở dạng hiện tại; biến thể B được duyệt với điều kiện xử lý F2–F5.

## 7. Hạn chế và bước tiếp theo

**Hạn chế của phần kiểm chứng:**

- Parity chứng minh bản Python **giống** model gốc, không chứng minh model gốc **tốt**. Các rủi ro ở mục 6 được chuyển sang nguyên vẹn (quy tắc GOV-01).
- Test multi-seed mới chưa được thử với một bản Python cố tình sai. Tuy vậy, hàm so sánh `parity.compare()` đã được kiểm chứng bắt lỗi bởi các test đột biến trong `test_guardrails.py`.
- Test do agent sinh ra (`runs/M2/output/tests/test_parity.py`) ghi cứng đường dẫn tuyệt đối tới dữ liệu, nên sẽ lỗi trên máy khác.
- Không có ghi chép nào về việc tải dữ liệu và chuyển `.xls` sang `.csv`.
- Thư mục dự án chưa phải git repo, nên chưa có lịch sử thay đổi.

**Bước tiếp theo đề xuất:**

1. Người phụ trách rà soát theo checklist trong `runs/M2/output/HUMAN_REVIEW.md` và chạy lệnh `approve`.
2. Sửa test do agent sinh ra để dùng đường dẫn tương đối (sửa ở template trong agent, không sửa file đầu ra).
3. Thêm script `data/fetch.sh` tải dữ liệu UCI, chuyển đổi và kiểm tra hash.
4. Nếu sau này so sánh AUC trực tiếp, làm tròn PD trước hoặc dùng dung sai khoảng 1e-6.
5. Xây dựng lại theo biến thể B và xử lý F2–F5 trước khi dùng.

## 8. Cách chạy lại

Toàn bộ chuỗi (discover → document → migrate → validate → monitor → tổng kết), một lệnh:

```bash
./run_m2.sh            # hoặc: .venv/bin/python -m src.factory legacy_models/m2_taiwan/migration.yaml
./run_m2.sh --fresh    # chạy lại từ đầu, LLM replay từ cache
```

Kết quả tổng hợp: `reports/inventory/FACTORY_RUN_M2.md`. Mọi bước đều ghi vào MLflow (experiment `m2-taiwan-default`).

Chỉ chạy test:

```bash
.venv/bin/python -m pytest -q r2py_agent/tests runs/M2/output/tests src/tests
```

Chạy lại agent và phê duyệt:

```bash
.venv/bin/python -m r2py_agent run legacy_models/m2_taiwan/migration.yaml
.venv/bin/python -m r2py_agent status legacy_models/m2_taiwan/migration.yaml
.venv/bin/python -m r2py_agent approve legacy_models/m2_taiwan/migration.yaml --reviewer "<tên>" --decision approve --comment "..."
```

Xem kết quả MLflow (lưu local trong `mlflow.db`):

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```
