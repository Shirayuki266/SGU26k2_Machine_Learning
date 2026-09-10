import random 
import torch 
A = torch.tensor(random.random(), requires_grad=True) 
B = torch.tensor(random.random(), requires_grad=True) 
C = torch.tensor(random.random(), requires_grad=True) 
D = torch.tensor(random.random(), requires_grad=True) 
EPOCHS = 2000 
optimizer = torch.optim.NAdam([A, B, C, D], lr=0.01) 
for _ in range(EPOCHS): 
    y1 = A + B - 9 
    y2 = C - D - 1 
    y3 = A + C - 8 
    y4 = B - D - 2 
    sqerr = y1*y1 + y2*y2 + y3*y3 + y4*y4 
    optimizer.zero_grad() 
    sqerr.backward() 
    optimizer.step() 
print("A =", A) 
print("B =", B)
print("C =", C) 
print("D =", D)

# Lấy giá trị vô hướng (scalar)
a_val, b_val, c_val, d_val = A.item(), B.item(), C.item(), D.item()

print("--- NGHIỆM THU ĐƯỢC TỪ AUTOGRAD ---")
print(f"A = {a_val}")
print(f"B = {b_val}")
print(f"C = {c_val}")
print(f"D = {d_val}")

# -------------------------------------------------------------
# ĐÁP ỨNG YÊU CẦU 1: Kiểm tra nghiệm với 4 phương trình
print("\n--- KIỂM TRA THAY VÀO 4 PHƯƠNG TRÌNH ---")
print(f"1) A + B = {a_val + b_val} (Mục tiêu: 9)")
print(f"2) C - D = {c_val - d_val} (Mục tiêu: 1)")
print(f"3) A + C = {a_val + c_val} (Mục tiêu: 8)")
print(f"4) B - D = {b_val - d_val} (Mục tiêu: 2)")

print("\n--- NHẬN XÉT ---")
print("Hệ phương trình bị phụ thuộc tuyến tính (hạng ma trận rank = 3 < số ẩn = 4).")
print("Phương trình (4) thực chất được rút ra từ (1) + (2) - (3): (A+B) + (C-D) - (A+C) = B - D = 2.")
print("Do đó, hệ chỉ có 3 phương trình độc lập cho 4 ẩn -> Hệ có vô số nghiệm phụ thuộc 1 tham số t (ví dụ D = t).")