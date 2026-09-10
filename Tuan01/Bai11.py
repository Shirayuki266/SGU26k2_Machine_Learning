import numpy as np
import torch
X = torch.arange(24)
print("\nGốc (24 phần tử):\n", X)
print("\nReshape 4 x 6:\n", X.reshape(4, 6))
print("\nReshape 2 x 12:\n", X.reshape(2, 12))
print("\nReshape 2 x 3 x 4:\n", X.reshape(2, 3, 4))

A = torch.randn(3, 4)
B = torch.randn(3, 4)

print("Tensor A:\n", A)
print("\nTensor B:\n", B)
print("\nA + B = \n", A + B)
print("\nA * B = \n", A * B)
print("\nA / B = \n", A / B)

# Thống kê theo CỘT -> triệt tiêu chiều hàng (dim=0)
dim=0
print("\nTrung bình theo cột:\n", torch.mean(A, dim))
print("\nĐộ lệch chuẩn theo cột:\n", torch.std(A, dim))

x = torch.tensor(2.0, requires_grad=True)
y = 2 * x**2 + 5 * x + 3
y.backward()

grad_autograd = x.grad.item()
print(f"\nĐạo hàm tự động: {grad_autograd}")

# y = 2x^2 + 5x + 3
#y' = 4x + 5
#y'(2) = 4*2 + 5 = 13
