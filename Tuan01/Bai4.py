import torch
a = torch.randn(3, 4, 5)
print("Original tensor:")
print(a)
print("\nTensor a[1]:")
print(a[1])
print("\nTensor a[1:, 2:4]:")
print(a[1:, 2:4])
print("\nKích thước của a[1]:", a[1].shape)
print("\nKích thước của a[1:, 2:4]:", a[1:, 2:4].shape)
print("\nSố chiều: ", a.ndim)
hpt = a[:, :, -2:]
print("\nHai phần tử cuối trên chiều thứ 3:\n", hpt)