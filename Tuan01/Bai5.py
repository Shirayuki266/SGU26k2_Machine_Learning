import torch
a = torch.randn(3, 4)
print("Original:")
print(a)
print("\nFlatten:")
print(a.ravel())
print("\nReshape to 3 x 2 x 2:")
print(a.reshape(3, 2, 2))

print("Số phần tử ban đầu (3x4):", a.numel())
print("Số phần tử sau khi reshape (3x2x2):", a.reshape(3, 2, 2).numel())

b = a.reshape(2, 6)
print("Reshape to 2 x 6:\n", b)
print("Shape của b:", b.shape)