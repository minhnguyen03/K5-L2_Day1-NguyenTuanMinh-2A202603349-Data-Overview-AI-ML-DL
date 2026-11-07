# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. class_id = 468
class_name = cab,
rank = 1,
score = 0.510915,
taxonomy_name = ImageNet-1K

2. Record này nói về cả bức ảnh, không phải một vật thể riêng biệt trong ảnh.

Lý do:
-Trường task có giá trị image_classification, cho biết mô hình đang thực hiện bài toán phân loại toàn bộ hình ảnh.
-Mô hình YOLO11n Classification (yolo11n-cls.pt) dự đoán nhãn cho toàn bộ ảnh đầu vào.
-Record không có các trường như bounding_box, x, y, width, height để xác định vị trí của từng vật thể.

3. Danh sách tên lớp (class list) không phải do model tự nghĩ ra, mà được xác định bởi taxonomy và checkpoint đi kèm mô hình.

Cụ thể trong trường hợp này:
- model_file = yolo11n-cls.pt: checkpoint chứa trọng số và thông tin phục vụ phân loại.
- taxonomy_name = ImageNet-1K: bộ danh mục gồm 1.000 lớp được sử dụng để phân loại.
- class_id = 468: mã định danh của lớp trong danh sách.
- class_name = cab: tên lớp tương ứng với mã 468.

4. Cần ghi cả class_id, class_name và taxonomy_name để định danh lớp phân loại chính xác, tránh nhầm lẫn và thuận tiện cho việc xử lý dữ liệu.
- class_id = 468: Mã số định danh của lớp, giúp máy tính dễ dàng tham chiếu và xử lý.
- class_name = cab: Tên lớp giúp con người dễ đọc và hiểu kết quả dự đoán.
- taxonomy_name = ImageNet-1K: Cho biết lớp đó thuộc hệ thống danh mục nào, giúp xác định đúng ý nghĩa của mã lớp.

5. Guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn bộ bức ảnh khi có nhiều vật thể hoặc chủ đề.

Cụ thể cần xác định:
- Đối tượng chính: Chọn vật thể hoặc chủ đề nổi bật nhất trong ảnh.
- Mức độ nổi bật: Ưu tiên đối tượng chiếm diện tích lớn, nằm ở vị trí trung tâm hoặc thu hút sự chú ý nhất.
- Ngữ cảnh ảnh: Nếu ảnh thể hiện một chủ đề chung, chọn lớp phản ánh đúng chủ đề đó.
- Trường hợp nhiều đối tượng ngang nhau: Quy định cách ưu tiên hoặc tiêu chí phân xử thống nhất.
- Trường hợp không xác định được: Có quy định lớp unknown hoặc other nếu taxonomy hỗ trợ.

6. Không được lấy score của model làm Ground Truth vì score chỉ thể hiện mức độ tự tin của mô hình khi dự đoán, không chứng minh dự đoán đó đúng.

Cụ thể:
- score là kết quả do model tính toán dựa trên hình ảnh đầu vào.
- Ground Truth là nhãn thực tế được xác định thông qua dữ liệu gốc hoặc người gán nhãn theo guideline.
- Model có thể dự đoán sai nhưng vẫn có score cao.
- Nếu dùng score làm Ground Truth thì sẽ không thể đánh giá khách quan độ chính xác của model.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. class_name = bowl
score = 0.499948
bbox_xyxy = [59.29, 288.92, 133.46, 328.98]
bbox_width = 74.18
bbox_height = 40.06

2. Góc trên-trái: Nằm tại tọa độ (59.29, 288.92) pixel tính từ góc trên bên trái của ảnh.
Góc dưới-phải: Nằm tại tọa độ (133.46, 328.98) pixel.
Chiều rộng: 74.18 pixel.
Chiều cao: 40.06 pixel.

3. Ngưỡng 0.20: 17 predictions.
Ngưỡng 0.60: 6 predictions.

4. Khi hạ ngưỡng (ví dụ từ 0.60 xuống 0.20), mô hình giữ lại nhiều Bounding Box hơn, vì vậy reviewer phải dành thêm thời gian kiểm tra và xác minh từng prediction.

Cụ thể:
- Kiểm tra các hộp có thực sự bao quanh vật thể hay không.
- Phát hiện các hộp nhận diện sai (False Positive).
- Kiểm tra các vật thể bị nhận diện trùng lặp.
- Xác minh class có đúng với vật thể không.
- Đối chiếu Bounding Box với ảnh thực tế và chỉnh sửa nếu cần.

5. Quy tắc Bounding Box: Người gán nhãn phải vẽ khung bao sát với ranh giới vật thể, bao phủ toàn bộ phần vật thể nhìn thấy được, không lấy thừa quá nhiều vùng nền xung quanh và không cắt mất phần vật thể đang hiển thị. Mỗi vật thể được gán một Bounding Box riêng và đúng với nhãn lớp tương ứng.

6. Một ví dụ là chiếc bát (bowl) ở sát mép dưới bên trái ảnh, bị cắt một phần khỏi khung hình.

Tôi sẽ đưa trường hợp này lên hỏi Lab Coach (escalation) để xác nhận cách xử lý vật thể bị cắt mép, chẳng hạn chỉ vẽ Bounding Box cho phần nhìn thấy hay xử lý theo quy tắc khác.

Lý do: Đây là trường hợp cần thống nhất cách gán nhãn để tránh mỗi annotator xử lý khác nhau. Sau khi Lab Coach xác nhận, quy tắc chung sẽ được bổ sung vào guideline để áp dụng nhất quán cho các ảnh tiếp theo.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. instance_id = kitchen-009
class_name = person
score = 0.431437
polygon_point_count = 65
3–5 điểm đầu của polygon_xy:
- [1.0, 264.0]
- [1.0, 309.0]
- [5.0, 309.0]
- [6.0, 308.0]
- [15.0, 308.0]

2. So với Bounding Box (box) chỉ xác định vùng hình chữ nhật bao quanh vật thể, Polygon cung cấp thêm thông tin về đường viền và hình dạng thực tế của vật thể.

Cụ thể:
- Box: Chỉ có 4 tọa độ xác định hình chữ nhật bao quanh vật thể, có thể chứa cả vùng nền không thuộc vật thể.
- Polygon: Gồm nhiều điểm tọa độ tạo thành đường bao sát theo biên dạng vật thể, giúp xác định chính xác vùng pixel thuộc vật thể hơn.
- Ứng dụng: Phân biệt vật thể với nền, nhận diện hình dạng không đều và đo diện tích hoặc đường viền vật thể.

3. instance_id là mã định danh duy nhất cho một đối tượng cụ thể (instance) trong kết quả phân đoạn, giúp phân biệt các đối tượng được phát hiện, kể cả khi chúng có cùng class_name.
- instance_id = kitchen-009: mã riêng của đối tượng này trong dữ liệu kitchen.
- class_id = 0: đối tượng thuộc lớp person.
- Nếu ảnh có 3 người, mỗi người có thể có instance_id khác nhau nhưng cùng class_id = 0.

4. Vẽ polygon sát theo đường viền thực tế của vật thể, bao phủ toàn bộ phần vật thể nhìn thấy được, không lấy thêm vùng nền xung quanh và không bỏ sót các phần vật thể có thể quan sát được; các điểm polygon phải nối liên tục, tạo thành một vùng khép kín.

5. Tôi chọn escalation (báo Lab Coach để xin hướng dẫn).

Lý do: Đây là trường hợp biên vật thể không rõ ràng. Cần xác nhận cách xử lý vùng bị che khuất hoặc không thể xác định chính xác đường viền trước khi đưa ra quy tắc thống nhất cho người gán nhãn. Không nên tự suy đoán vì có thể tạo ra mask không nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp đại diện cho toàn bộ ảnh. Đơn vị nhãn là một ảnh – một nhãn theo quy tắc của bài toán. | JSON: Ảnh traffic có 5 dự đoán. cab đứng thứ nhất với score=0.510915, tiếp theo là minibus=0.164284, police_van=0.085848. PNG: Ảnh có nhiều phương tiện giao thông nên khó xác định lớp nào đại diện phù hợp nhất cho toàn ảnh. | Quan sát toàn bộ ảnh, xác định nội dung chính và chọn nhãn theo taxonomy, guideline. Không mặc định chọn nhãn có score cao nhất nếu chưa đối chiếu ảnh. | Kiểm tra nhãn có đúng với nội dung ảnh không; đặc biệt xem xét trường hợp ảnh có nhiều đối tượng hoặc nhiều lớp có thể đại diện. |
| Phát hiện vật thể | Mỗi đối tượng có một bounding box hình chữ nhật, kèm class_id, class_name và tọa độ bbox_xyxy. Đơn vị nhãn là một đối tượng – một bounding box. | JSON/PNG: Với ảnh kitchen, ở threshold 0.35 mô hình phát hiện 11 đối tượng; giảm xuống 0.20 tăng lên 17, còn tăng lên 0.60 giảm còn 6. PNG: Có nhiều box tập trung ở khu vực bàn bếp phía dưới bên trái, một số box nhỏ và gần nhau, gây khó khăn khi xác định vật thể, box trùng lặp hoặc box chưa chính xác.| Vẽ box bao quanh từng đối tượng, gán đúng lớp, không bỏ sót phần nhìn thấy được. Kiểm tra các box nhỏ, chồng lấn và vật thể ở sát mép ảnh. Escalation nếu chưa rõ quy tắc xử lý vật thể bị che hoặc bị cắt. | Kiểm tra class, tọa độ và kích thước box; xem box có bao đủ vật thể, có lấy thừa nền, bỏ sót đối tượng hoặc phát hiện trùng lặp không. |
| Instance segmentation | Mỗi đối tượng có một mask riêng, biểu diễn bằng polygon gồm nhiều điểm tọa độ. Đơn vị nhãn là một instance – một mask. | JSON: Record kitchen-009 có class_name=person, score=0.431437, polygon_point_count=65. Năm điểm đầu của polygon là [1,264], [1,309], [5,309], [6,308], [15,308]. PNG/JSON: Đối tượng người nằm gần mép trái ảnh; đường biên cần được kiểm tra kỹ tại vùng tiếp giáp với mép ảnh và nền. | Vẽ polygon sát đường viền thực tế của vật thể, bao phủ phần nhìn thấy được, không lấy thừa nền và tạo mask khép kín. Nếu biên mờ hoặc không rõ cách xử lý vùng bị che thì escalation. | Kiểm tra mask có bám sát đường viền vật thể không; có thiếu hoặc thừa vùng pixel không; polygon có khép kín và gắn đúng instance, đúng class không. |

## 5. An toàn dữ liệu

1. Không chia sẻ, sao chép hoặc tiết lộ dữ liệu, hình ảnh và thông tin của bài lab ra bên ngoài khi chưa được cấp phép; chỉ sử dụng chúng đúng mục đích và theo hướng dẫn bảo mật.

2. Tôi sẽ dừng xử lý và báo ngay cho Lab Coach hoặc người phụ trách dự án (Project Lead) để được hướng dẫn, không tự ý tiếp tục xử lý dữ liệu ngoài phạm vi.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
