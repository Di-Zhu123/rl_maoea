import torch

def hv_normalized(objs: torch.Tensor, optimum: torch.Tensor, num_sample: int = 100000) -> float:
    """
    计算归一化后的超体积（hv），结果在 [0,1] 范围内。
    
    :param objs: 目标点，形状为 (n_points, n_objs)。
    :param optimum: 每个目标的最大值（参考点），形状为 (n_objs, )。
    :param num_sample: 蒙特卡罗采样数量。
    :return: 估计的归一化超体积 hv。
    """
    # 计算每个目标的下界 fmin（取 objs 各维最小值与 0 的较小值）
    fmin = torch.min(torch.cat([objs.min(dim=0).values.unsqueeze(0), torch.zeros(1, objs.size(1))], dim=0), dim=0).values
    fmax = optimum  # 上界为每个目标的最大值

    # 对目标值进行归一化
    scaled_objs = (objs - fmin) / ((fmax - fmin) * 1.1)
    # 剔除归一化后超出 [0,1] 的点
    scaled_objs = scaled_objs[(scaled_objs <= 1).all(dim=1)]
    
    # 单位超立方体的参考点
    ref_point = torch.ones(scaled_objs.size(1))
    
    # 蒙特卡罗采样：在 [0,1]^d 内均匀采样
    samples = torch.rand(num_sample, scaled_objs.size(1))
    # 判断每个样本是否被至少一个归一化后的点支配
    # 对于最小化问题：一个样本被点支配当且仅当对于某个点，样本在所有维度上 >= 该点的值
    dominated = torch.any(torch.all(samples.unsqueeze(1) >= scaled_objs.unsqueeze(0), dim=2), dim=1)
    
    # 单位超立方体的体积为 1，hv 即为被支配样本的比例
    hv = dominated.float().mean().item()
    return hv

# 示例调用：
# objs 为 (n_points, n_objs) 的 Tensor，optimum 为各维度最大值
# hv_value = hv_normalized(objs, optimum)
