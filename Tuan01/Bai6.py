import torch
a = torch.randn(2, 3)
b = torch.randn(2, 3)
print("a:", a)
print("\nb:", b)
print("\na + b:", a + b)
print("\na / b:", a / b)
print("\na^2:", a ** 2)
print("\nMatrix multiplication:", a @ b.T)
print("\n======Phân biệt phép toán a * b với a @ b.T===============")
print("a * b: Là phép nhân từng phần tử tương ứng. Yêu cầu a và b phải cùng shape (2, 3). Kết quả trả về có shape (2, 3).")
print("a:", a)
print("\nb:", b)
print("\nChi tiết:\n")
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        val_a = a[i, j].item()
        val_b = b[i, j].item()
        res = val_a * val_b
        print(f"Vị trí [{i+1},{j+1}]: {val_a:.2f} * {val_b:.2f} = {res:.2f}")
print("\n=> a * b = ", a*b)
print("\n==========================================================")
print("a @ b.T: Là phép nhân ma trận. Yêu cầu số cột của a phải bằng số hàng của b.T. Kết quả trả về có shape (2, 2).")
print("a:", a)
print("\nb:", b)
b_T = b.T
print("\nb.T= ", b_T)
print("\nChi Tiết:\n")
for i in range(a.shape[0]):
    for j in range(b_T.shape[1]):
        phep_tinh = " + ".join([f"{a[i, k]:.2f}*{b_T[k, j]:.2f}" for k in range(a.shape[1])])
        gia_tri = (a[i] * b_T[:, j]).sum().item()
        print(f"Vị trí [{i+1},{j+1}]: {phep_tinh} = {gia_tri:.2f}")
print("=> a @ b.T = ", a @ b.T)

kq = a @ b.T
print("\nShape của a @ b.T là:", kq.shape)