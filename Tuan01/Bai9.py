import numpy as np
import torch

# Danh sách các bộ hệ số đa thức cần huấn luyện
danh_sach_he_so = [
    {"ten": "Bài 9: y = x² + 2x + 3", "goc": [1, 2, 3]},
    {"ten": "Bài 11: y = 2x² - 3x + 5", "goc": [2, -3, 5]}
]

for bai in danh_sach_he_so:
    print(f"\n=================== {bai['ten']} ===================")
    
    # 1. Sinh dữ liệu từ đa thức
    true_coeffs = bai['goc']
    polynomial = np.poly1d(true_coeffs)
    N = 20

    X = np.random.randn(N, 1) * 5
    Y = polynomial(X)

    # 2. Tạo ma trận đặc trưng XX = [X^2, X, 1]
    XX = np.hstack([X * X, X, np.ones_like(X)])

    # 3. Khởi tạo trọng số ngẫu nhiên w và chuyển dữ liệu sang Tensor
    w = torch.randn(3, 1, requires_grad=True)
    x = torch.tensor(XX, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)

    # 4. Khởi tạo Optimizer NAdam
    optimizer = torch.optim.NAdam([w], lr=0.1)

    print("Initial coefficients (Hệ số ban đầu):")
    print(w.detach().numpy().ravel())

    # 5. Vòng lặp huấn luyện (Training Loop)
    for epoch in range(2000):
        optimizer.zero_grad()                   # Bước 1: Xóa gradient cũ
        y_pred = x @ w                          # Dự đoán
        mse = torch.mean(torch.square(y - y_pred)) # Tính hàm mất mát MSE
        mse.backward()                          # Bước 2: Tính gradient cho w
        optimizer.step()                        # Bước 3: Cập nhật trọng số w

    # 6. Đánh giá và so sánh kết quả
    w_learned = w.detach().numpy().ravel()
    true_arr = np.array(true_coeffs, dtype=float)
    sai_so = np.abs(w_learned - true_arr)

    print("\nLearned coefficients:")
    print(f"[{w_learned[0]:.4f}, {w_learned[1]:.4f}, {w_learned[2]:.4f}]")

    print("\n--- SO SÁNH HỆ SỐ ---")
    print(f"Hệ số gốc (True):        [{true_arr[0]:.4f}, {true_arr[1]:.4f}, {true_arr[2]:.4f}]")
    print(f"Hệ số học được (Learned): [{w_learned[0]:.4f}, {w_learned[1]:.4f}, {w_learned[2]:.4f}]")
    print(f"Mức độ sai số tuyệt đối:  [{sai_so[0]:.4f}, {sai_so[1]:.4f}, {sai_so[2]:.4f}]")

# -------------------------------------------------------------
# GIẢI THÍCH VAI TRÒ CÁC HÀM:
# 1. optimizer.zero_grad(): Xóa gradient cũ tích lũy từ bước lặp trước về 0.
# 2. mse.backward(): Kích hoạt Backpropagation để tính đạo hàm (gradient) của mse theo w và lưu vào w.grad.
# 3. optimizer.step(): Dựa vào w.grad, thuật toán NAdam sẽ cập nhật giá trị mới cho w để giảm dần lỗi mse.