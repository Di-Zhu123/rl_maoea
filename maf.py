import torch
from evox.core import Problem
from evox.operators.sampling import uniform_sampling
import math

class MaFTestSuit(Problem):
    """
    Base class for MaF test suite problems in multi-objective optimization.
    Inherit this class to implement specific MaF problem variants.
    :param d: Number of decision variables.
    :param m: Number of objectives.
    :param ref_num: Number of reference points used in the problem.
    """
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        super().__init__()
        self.d = d
        self.m = m
        self.ref_num = ref_num
        self.sample, _ = uniform_sampling(self.ref_num * self.m, self.m)
        self.device = self.sample.device

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to evaluate the objective values for given decision variables.
        :param X: A tensor of shape (n, d), where n is the number of solutions and d is the number of decision variables.
        :return: A tensor of shape (n, m) representing the objective values for each solution.
        """
        raise NotImplementedError()

    def pf(self):
        """
        Return the Pareto front for the problem.
        :return: A tensor representing the Pareto front.
        """
        f = self.sample / 2
        return f


class MaF1(MaFTestSuit):
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        super().__init__(d, m, ref_num)
        # 默认参数设置
        if self.d is None:
            self.d = self.m + 9 if self.m is not None else 12
        if self.m is None:
            self.m = 3

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        
        # 计算g值
        g = torch.sum((X[:, m-1:] - 0.5)**2, dim=1, keepdim=True)
        
        # 构建cumprod项
        ones_col = torch.ones((n, 1), device=X.device)
        cumprod_term = torch.cumprod(
            torch.cat([ones_col, X[:, :m-1]], dim=1), 
            dim=1
        )
        cumprod_term = torch.flip(cumprod_term, dims=[1])  # 反转
        
        # 构建反转项
        reversed_X = torch.flip(X[:, :m-1], dims=[1])  # 反转前m-1列
        reversed_term = torch.cat([
            ones_col, 
            1 - reversed_X
        ], dim=1)
        
        # 重复g并计算目标函数
        repeat_g = (1 + g).expand(-1, m)
        f = repeat_g - repeat_g * cumprod_term * reversed_term
        
        return f

    def pf(self):
        # 生成参考点并反转
        f = 1 - self.sample
        return f

class MaF2(MaFTestSuit):
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        super().__init__(d, m, ref_num)
        # 处理默认参数
        if self.d is None:
            self.d = self.m + 9 if self.m is not None else 12
        if self.m is None:
            self.m = 3

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        device = X.device
        g = torch.zeros((n, m), device=device)
        total_vars = d - m + 1
        interval = total_vars // m

        for i in range(m):
            # 计算每个g_i对应的变量区间
            if i < m - 1:
                start = m - 1 + i * interval
                end = start + interval
                # 处理最后一个块可能超出的情况
                if i == m - 2:
                    end = d
            else:
                start = m - 1 + (m - 1) * interval
                end = d

            # 创建掩码并处理变量
            mask = torch.zeros_like(X, dtype=torch.bool, device=device)
            mask[:, start:end] = True
            temp = torch.where(mask, X, 0.5).div(2).add(0.25)
            
            # 计算g_i
            g[:, i] = torch.sum((temp - 0.5).pow(2), dim=1)

        # 计算目标函数
        f1_part = torch.cat([
            torch.ones(n, 1, device=device),
            torch.cos((X[:, :m-1] / 2 + 0.25) * math.pi / 2)
        ], dim=1)
        f1 = torch.flip(torch.cumprod(f1_part, dim=1), dims=[1])

        f2_part = torch.cat([
            torch.ones(n, 1, device=device),
            torch.sin(
            (torch.flip(X[:, :m-1], dims=[1]) / 2 + 0.25) * math.pi / 2)
        ], dim=1)
        f2 = f2_part

        return (1 + g) * f1 * f2
    # g (85,5) f1 (85,5) f2 (85,5) 

    def pf(self):
        m = self.m
        ref_num = self.ref_num * m
        r = self.sample.clone()  # 使用基类预生成的参考点
        
        # 初始化c矩阵
        c = torch.zeros((r.shape[0], m-1), device=self.device)
        
        # 逐点计算c值
        for i in range(r.shape[0]):
            for j in range(2, m):  # 修改循环范围为range(2, m)
                # 计算连乘项
                product_dims = slice(m-j+1, m-1)
                product = torch.prod(c[i, product_dims]) if j > 2 else 1.0
                temp = r[i, j-1] / r[i, 0] * product
                c[i, m-j] = torch.sqrt(1 / (1 + temp**2))
        
        # 根据目标数处理c矩阵
        if m > 5:
            angle1 = math.cos(math.pi/8)
            angle2 = math.cos(3*math.pi/8)
            c = c * (angle1 - angle2) + angle2
        else:
            # 筛选符合条件的点
            mask = torch.all((c >= math.cos(3*math.pi/8)) & 
                             (c <= math.cos(math.pi/8)), dim=1)
            c = c[mask]
        
        # 构建Pareto前沿
        f_part = torch.cat([
            torch.ones(c.shape[0], 1, device=self.device),
            c[:, :m-1]
        ], dim=1)
        f = torch.flip(torch.cumprod(f_part, dim=1), dims=[1]) * torch.cat([
            torch.ones(c.shape[0], 1, device=self.device),
            torch.sqrt(1 - c[:, m-2::-1].pow(2))
        ], dim=1)
        
        return f
    
class MaF3(MaFTestSuit):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        self.d = self.m + 9 if self.d is None else self.d
        self.m = 3 if self.m is None else self.m

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            (d - m + 1) 
            + torch.sum(
                (X[:, m-1:] - 0.5).pow(2) 
                - torch.cos(20 * torch.pi * (X[:, m-1:] - 0.5)),
                dim=1, keepdim=True
            )
        )
        
        f1 = torch.cat([
            torch.ones(n, 1, device=self.device),
            torch.cos(X[:, :m-1] * torch.pi / 2)
        ], dim=1)
        f1 = torch.flip(torch.cumprod(f1, dim=1), [1])
        
        f2 = torch.cat([
            torch.ones(n, 1, device=self.device),
            torch.sin(torch.flip(X[:, :m-1], [1]) * torch.pi / 2)
        ], dim=1)
        
        f = (1 + g) * f1 * f2
        f = torch.cat([f[:, :-1]**4, f[:, -1:]**2], dim=1)
        return f

    def pf(self):
        m = self.m
        ref_num = self.ref_num * m
        r = self.sample.pow(2).clone()
        temp = torch.sum(torch.sqrt(r[:, :-1]), dim=1, keepdim=True) + r[:, -1:]
        f = r / torch.cat([temp.pow(2).expand(-1, m-1), temp], dim=1)
        return f

class MaF4(MaFTestSuit):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        self.d = self.m + 9 if self.d is None else self.d
        self.m = 3 if self.m is None else self.m

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            (d - m + 1) 
            + torch.sum(
                (X[:, m-1:] - 0.5).pow(2) 
                - torch.cos(20 * torch.pi * (X[:, m-1:] - 0.5)),
                dim=1, keepdim=True
            )
        )
        
        f1 = torch.cat([
            torch.ones(n, 1, device=self.device),
            torch.cos(X[:, :m-1] * torch.pi / 2)
        ], dim=1)
        f1 = torch.flip(torch.cumprod(f1, dim=1), [1])
        
        f2 = torch.cat([
            torch.ones(n, 1, device=self.device),
            torch.sin(torch.flip(X[:, :m-1], [1]) * torch.pi / 2)
        ], dim=1)
        
        f = (1 + g) * f1 * f2
        f = f * (2 ** torch.arange(1, m+1, device=self.device))
        return f

    def pf(self):
        m = self.m
        ref_num = self.ref_num * m
        r = self.sample.clone()
        r_normalized = r / torch.norm(r, dim=1, keepdim=True)
        f = (1 - r_normalized) * (2 ** torch.arange(1, m+1, device=self.device))
        return f

class MaF5(MaFTestSuit):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        self.d = self.m + 9 if self.d is None else self.d
        self.m = 3 if self.m is None else self.m

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        X = X.clone()
        X[:, :m-1] = X[:, :m-1].pow(10)
        g = torch.sum((X[:, m-1:] - 0.5).pow(2), dim=1, keepdim=True)
        
        f1 = torch.cat([
            torch.ones(n, 1, device=self.device),
            torch.cos(X[:, :m-1] * torch.pi / 2)
        ], dim=1)
        f1 = torch.flip(torch.cumprod(f1, dim=1), [1])
        
        f2 = torch.cat([
            torch.ones(n, 1, device=self.device),
            torch.sin(torch.flip(X[:, :m-1], [1]) * torch.pi / 2)
        ], dim=1)
        
        f = (1 + g) * f1 * f2
        f = f * (2 ** torch.arange(m, 0, -1, device=self.device))
        return f

    def pf(self):
        m = self.m
        ref_num = self.ref_num * m
        r = self.sample.clone()
        r_normalized = r / torch.norm(r, dim=1, keepdim=True)
        f = r_normalized * (2 ** torch.arange(m, 0, -1, device=self.device))
        return f

class MaF6(MaFTestSuit):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        self.d = self.m + 9 if self.d is None else self.d
        self.m = 3 if self.m is None else self.m

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        i = 2  # 固定参数
        g = torch.sum((X[:, m-1:] - 0.5).pow(2), dim=1, keepdim=True)
        
        # 更新X的特定维度
        temp = g.repeat(1, m - i)
        X[:, i-1:m-1] = (1 + 2 * temp * X[:, i-1:m-1]) / (2 + 2 * temp)
        
        f1 = torch.cat([
            torch.ones(n, 1, device=self.device),
            torch.cos(X[:, :m-1] * torch.pi / 2)
        ], dim=1)
        f1 = torch.flip(torch.cumprod(f1, dim=1), [1])
        
        f2 = torch.cat([
            torch.ones(n, 1, device=self.device),
            torch.sin(torch.flip(X[:, :m-1], [1]) * torch.pi / 2)
        ], dim=1)
        
        f = (1 + 100 * g) * f1 * f2
        f = f / (2 ** 0.5) ** torch.arange(
            torch.maximum(torch.tensor(m - i, device=self.device), torch.tensor(0)),
            -1,
            -1,
            device=self.device
        )
        return f

    def pf(self):
        m = self.m
        i = 2
        ref_num = self.ref_num * m
        r = self.sample[:, :i].clone()
        r_normalized = r / torch.norm(r, dim=1, keepdim=True)
        
        if r_normalized.shape[1] < m:
            pad_cols = m - r_normalized.shape[1]
            r_normalized = torch.cat([
                torch.zeros(n, pad_cols, device=self.device), 
                r_normalized
            ], dim=1)
            
        f = r_normalized / (2 ** 0.5) ** torch.arange(
            torch.maximum(torch.tensor(m - i, device=self.device), torch.tensor(0)),
            -1,
            -1,
            device=self.device
        ).unsqueeze(0)
        return f

class MaF7(MaFTestSuit):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        self.d = self.m + 9 if self.d is None else self.d
        self.m = 3 if self.m is None else self.m

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 1 + 9 * torch.mean(X[:, m-1:], dim=1, keepdim=True)
        f = torch.zeros((n, m), device=self.device)
        f[:, :m-1] = X[:, :m-1]
        term = torch.sum(
            (f[:, :m-1] / (1 + g)) * (1 + torch.sin(3 * torch.pi * f[:, :m-1])),
            dim=1, keepdim=True
        )
        f[:, m-1:] = (1 + g) * (m - term)
        return f

    def pf(self):
        interval = torch.tensor([0.0, 0.251412, 0.631627, 0.859401], device=self.device)
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        
        x = self.sample.clone()
        x = torch.where(
            x <= median, 
            x * (interval[1] - interval[0]) / median + interval[0],
            (x - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
        )
        
        last_col = 2 * (self.m - torch.sum(
            x / 2 * (1 + torch.sin(3 * torch.pi * x)), 
            dim=1, keepdim=True
        ))
        return torch.cat([x, last_col], dim=1)