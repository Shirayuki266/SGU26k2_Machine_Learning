import torch
a = torch.randn(3, 4)
print(a)
for i in [0,1]:
    dim = i
    if dim == 0:
        chieu = "cột"
    else:  
        chieu = "hàng"
    print("\nCác phép toán thống kê theo chiều: ", chieu, "(dim = ", dim, "):")
    print("\nMean:")
    print(torch.mean(a, dim))
    print("\nStandard deviation:")
    print(torch.std(a, dim))
    print("\nCumulative sum:")
    print(torch.cumsum(a, dim))
    print("\n")
print("\n" + "="*20)
print("NHẬN XÉT SỰ KHÁC NHAU:")
print(f"- dim=0 (theo Cột): Gộp các hàng lại. Mean/Std trả về Vector {a.shape[1]} phần tử. Cumsum cộng dồn từ trên xuống dưới.")
print(f"- dim=1 (theo Hàng): Gộp các cột lại. Mean/Std trả về Vector {a.shape[0]} phần tử. Cumsum cộng dồn từ trái sang phải.")