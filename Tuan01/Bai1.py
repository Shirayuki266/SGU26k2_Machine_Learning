import torch
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
print(a)
print("\nKích thước:", a.size())
print("Số chiều:", a.ndim)
print("Kiểu dữ liệu:", a.dtype)
print("Kích thước: 2 hàng, 3 cột, số chiều: 2, kiểu dữ liệu: int32")