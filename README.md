# CloneTTS - Sao Chép Giọng Đọc Đa Ngôn Ngữ 🎙️🧠

**Tác giả:** Lý Trần

**🚀 Chạy thử trên Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/drive/1nmdU6vTKRBnjRxHDKudFxdCGbIB4Q3RM?usp=sharing](https://colab.research.google.com/drive/1ttXQ5GuMYm4ZPMrWNLFarm56efz8wDGt))

CloneTTS Giọng Đa Ngôn Ngữ là một ứng dụng web sử dụng Gradio, cung cấp giao diện thân thiện để tạo giọng nói, chuyển đổi giọng nói, và quản lý quy trình xử lý âm thanh nâng cao dựa trên mô hình Chatterbox của Resemble AI.

## Tính năng nổi bật

- **Quản lý dự án tập trung:**  
  Tạo, chọn và quản lý workspace riêng biệt. Mọi file đầu vào, file xử lý và kết quả sẽ được sắp xếp tự động vào đúng thư mục trong dự án.

- **Sinh giọng nói (Single Generation):**
    - **Text-to-Speech (TTS):** Sinh giọng nói chất lượng cao từ văn bản, có thể dùng file tham chiếu để clone giọng.
    - **Voice Conversion (VC):** Chuyển đổi đặc trưng giọng nói của file nguồn sang tham chiếu.
    - **Quét tham số (Parameter Sweep):** Sinh nhiều phiên bản cùng lúc với các giá trị tham số khác nhau (ví dụ: Temperature, Pace...).

- **Xử lý hàng loạt (Batch Processing):**
    - Xử lý cả thư mục văn bản hoặc âm thanh chỉ với một lần bấm.
    - Có thể ghép tất cả file âm thanh sinh ra thành một file duy nhất.

- **Chuẩn bị dữ liệu:**
    - **Tách văn bản:** Tự động chia nhỏ file văn bản dài thành nhiều đoạn phù hợp với mô hình.
    - **Tách file âm thanh:** Chia nhỏ file âm thanh thành các đoạn ngắn hơn, ưu tiên tách ở đoạn im lặng.

- **Chỉnh sửa & hoàn thiện quy trình:**
    - **Regenerate Audio:** Xem lại từng file audio, chỉnh sửa & thay thế nhanh chóng.
    - **Trình soạn thảo văn bản trực tiếp:** Sửa văn bản nguồn ngay trên giao diện, lưu lại dễ dàng.

## Yêu cầu cài đặt

- **Python:** >=3.8 (Khuyến nghị 3.11)
- **Git**
- **FFmpeg**
- **GPU CUDA** (khuyến nghị, chạy CPU sẽ rất chậm)

## Hướng dẫn cài đặt nhanh

### 1. Clone dự án

```bash
git clone https://github.com/ltteamvn/CloneTTS
cd CloneTTS
```

### 2. Tạo môi trường ảo Python

```bash
python3.11 -m venv toolkit
source toolkit/bin/activate    # Trên Linux/macOS
# .\toolkit\Scripts\activate   # Trên Windows
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Lưu ý: Nếu bạn dùng GPU Nvidia 10 series hoặc AMD, cần tự cài torch phù hợp.

### 4. Chạy ứng dụng

```bash
python app.py
```

Truy cập địa chỉ xuất hiện trên terminal (thường là http://127.0.0.1:7860) để sử dụng giao diện web.

## Quy trình sử dụng điển hình

1. **Tạo project** ở tab Projects.
2. **Chuẩn bị dữ liệu:**  
   - Upload văn bản/audio vào thư mục dự án tương ứng.  
   - Sử dụng tab Data Preparation để tách nhỏ file nếu cần.
3. **Sinh audio:**  
   - Vào tab Batch Generation hoặc Single Generation để sinh file âm thanh mong muốn.
4. **Chỉnh sửa & hoàn thiện:**  
   - Vào Edit Project Data để chỉnh sửa file text hoặc thay thế từng file audio.

## Một số lưu ý

- Thư mục dự án sẽ tự động lưu trữ toàn bộ file đầu vào, file xử lý và kết quả theo cấu trúc rõ ràng.
- Khi chuyển giọng, file tham chiếu (reference voice) nên ngắn hơn hoặc bằng 40 giây.
- Source Audio có thể dài hơn 40s, chương trình sẽ tự động chia nhỏ và ghép lại kết quả.

## Đóng góp & liên hệ

Nếu bạn gặp lỗi hoặc muốn đóng góp ý kiến, hãy tạo issue hoặc liên hệ trực tiếp với tác giả.

---

Chúc bạn sử dụng hiệu quả công cụ này!
