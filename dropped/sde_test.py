import torch

# 创建一个张量
x = torch.tensor([1, 2, 3])

# 尝试使用.numpy()方法
try:
    x.numpy()
    print("您的torch具有.numpy()功能")
except AttributeError:
    print("您的torch不具有.numpy()功能")