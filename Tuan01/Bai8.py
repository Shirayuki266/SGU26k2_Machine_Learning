import torch
x = torch.tensor(3.6, requires_grad=True)
y = x * x
y.backward()
print("\nx =", x)
print("\ny =", y)
print("\nx.grad =", x.grad)

x.grad.zero_() 
# hoặc khởi tạo lại bằng x = torch.tensor(3.6, requires_grad=True)
print("\nThay y = 3*x**2 + 2*x + 1")
y = 3 * x**2 + 2 * x + 1
y.backward()
print("\nx =", x)
print("\ny =", y)
print("\nx.grad =", x.grad)