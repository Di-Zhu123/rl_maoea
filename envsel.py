import torch
from sde import cal_fitness
import numpy as np
# # 使用 @torch.jit.ignore 来忽略使用 numpy 的部分
# @torch.jit.ignore
def environmental_selection(popobjs: torch.Tensor, n_sel: int):
    # 计算适应度
    fitness, dominance = cal_fitness(popobjs)
    # print("Fitness values:", fitness)
    # 环境选择初始过滤
    next_mask = fitness < 1.0
    num_next = torch.sum(next_mask).item()
    
    if num_next < n_sel:
        # 按适应度排序补充个体
        _, sorted_indices = torch.sort(fitness)
        selected_indices = sorted_indices[:n_sel]
    elif num_next > n_sel:
        # 需要截断K个个体
        k = num_next - n_sel
        # 获取需要截断的子集
        filtered_objs = popobjs[next_mask]
        del_mask = truncation(filtered_objs, k)
        # 合并索引
        next_indices = torch.where(next_mask)[0]
        remaining_indices = next_indices[~del_mask]
        selected_indices = remaining_indices
    else:
        selected_indices = torch.where(next_mask)[0].long()
    
    # 返回选中的目标值和适应度
    selected_popobjs = popobjs[selected_indices]
    selected_fitness = fitness[selected_indices]
    
    return selected_popobjs, selected_fitness, selected_indices

def truncation(popobjs: torch.Tensor, k: int):
    n = popobjs.size(0)
    del_mask = torch.zeros(n, dtype=torch.bool, device=popobjs.device)
    distance = torch.full((n, n), float('inf'), device=popobjs.device)
    
    
    # 计算支配后的距离矩阵
    spopobjs = torch.maximum(popobjs.unsqueeze(1), popobjs.unsqueeze(0))
    distance = torch.norm(popobjs.unsqueeze(1) - spopobjs, dim=2)
    

    # 循环删除密度最大的个体
    while torch.sum(del_mask) < k:
        remain_indices = torch.where(~del_mask)[0]
        temp_dist = distance[remain_indices][:, remain_indices]
        temp_dist = torch.sort(temp_dist, dim=1).values
        _, rank = torch.sort(temp_dist.sum(dim=1))
        del_mask[remain_indices[rank[0]]] = True
    return del_mask

# # cannot use numpy because of @jit_class
# def truncation_wrong(popobjs: torch.Tensor, k: int):
#     n = popobjs.size(0)
#     del_mask = torch.zeros(n, dtype=torch.bool)
#     distance = torch.full((n, n), float('inf'))
    
#     # 计算支配后的距离矩阵
#     spopobjs = torch.maximum(popobjs.unsqueeze(1), popobjs.unsqueeze(0))
#     distance = torch.norm(popobjs.unsqueeze(1) - spopobjs, dim=2)
    
#     # 初始化距离和堆
#     remain_indices = torch.arange(n)
#     temp_dist = distance[remain_indices][:, remain_indices]
#     temp_dist = torch.sort(temp_dist, dim=1).values
#     dist_sums = temp_dist.sum(dim=1).detach().cpu().numpy()

#     heap = [(dist_sums[i], i) for i in range(n)]
#     heapq.heapify(heap)
    
#     # 循环删除密度最大的个体
#     while torch.sum(del_mask) < k:
#         while True:
#             _, idx = heapq.heappop(heap)
#             if not del_mask[idx]:
#                 break
#         del_mask[idx] = True
        
#         # 更新距离和堆
#         remain_indices = torch.where(~del_mask)[0]
#         temp_dist = distance[remain_indices][:, remain_indices]
#         temp_dist = torch.sort(temp_dist, dim=1).values
#         dist_sums = temp_dist.sum(dim=1).detach().cpu().numpy()
#         heap = [(dist_sums[i], remain_indices[i].item()) for i in range(len(remain_indices))]
#         heapq.heapify(heap)
    
#     return del_mask


# def update_heap(distance, remain_indices):
#     temp_dist = distance[remain_indices][:, remain_indices]
#     temp_dist = torch.sort(temp_dist, dim=1).values
#     dist_sums = temp_dist.sum(dim=1).detach().cpu().numpy()
#     heap = [(dist_sums[i], remain_indices[i].item()) for i in range(len(remain_indices))]
#     heapq.heapify(heap)
#     return heap

# def truncation(popobjs: torch.Tensor, k: int):
#     n = popobjs.size(0)
#     del_mask = torch.zeros(n, dtype=torch.bool)
#     distance = torch.full((n, n), float('inf'))

#     # 计算支配后的距离矩阵
#     spopobjs = torch.maximum(popobjs.unsqueeze(1), popobjs.unsqueeze(0))
#     distance = torch.norm(popobjs.unsqueeze(1) - spopobjs, dim=2)
    
#     # 初始化距离和堆
#     remain_indices = torch.arange(n)
#     temp_dist = distance[remain_indices][:, remain_indices]
#     temp_dist = torch.sort(temp_dist, dim=1).values
#     dist_sums = temp_dist.sum(dim=1).detach().cpu().numpy()

#     heap = [(dist_sums[i], i) for i in range(n)]
#     heapq.heapify(heap)
    
#     # 循环删除密度最大的个体
#     while torch.sum(del_mask) < k:
#         while True:
#             _, idx = heapq.heappop(heap)
#             if not del_mask[idx]:
#                 break
#         del_mask[idx] = True
        
#         # 更新距离和堆
#         remain_indices = torch.where(~del_mask)[0]
#         heap = update_heap(distance, remain_indices)
    
#     return del_mask

if __name__ == '__main__':
    # 测试用例
    # popobjs = torch.tensor([
    #     [0.2, 0.3],
    #     [0.5, 0.6],
    #     [0.8, 0.9],
    #     [1.2, 1.3],
    #     [1.5, 1.6],
    #     [0.1, 0.2],
    #     [0.4, 0.5],
    #     [0.7, 0.8],
    #     [1.0, 1.1]
    # ])
    popobjs = torch.tensor([
        # Pareto Rank 1 (最优前沿)
        [0.1, 0.2],   # 被所有后续前沿支配
        [0.2, 0.1],   # 
        [0.15, 0.15], # 

        # Pareto Rank 2
        [0.3, 0.3],   # 被Rank1支配，支配Rank3及以下
        [0.35, 0.25], # 
        [0.25, 0.35], # 

        # Pareto Rank 3
        [0.4, 0.4],   # 被Rank1/2支配，支配Rank4
        [0.45, 0.35], # 
        [0.35, 0.45], # 

        # Pareto Rank 4
        [0.5, 0.5],   # 被所有前面支配
        [0.55, 0.45], # 
        [0.45, 0.55], # 

        # 被完全支配的个体
        [0.6, 0.6],   # 被所有前面支配
        [0.7, 0.7],   # 
        [0.8, 0.8],   # 

        # 随机分布的中间个体
        [0.65, 0.55], # 被多个个体支配
        [0.55, 0.65], # 
        [0.75, 0.65], # 
        [0.65, 0.75], # 
        [0.85, 0.75]  # 
    ])
    selected_pop, selected_fitness, selected_indices = environmental_selection(popobjs, 10)
    print("Selected objs:", selected_pop)
    print("Selected fitness:", selected_fitness)
    print("Selected indices:", selected_indices)
#     Fitness values: tensor([  0.4497,  19.4497,  37.4497,  54.4878,  70.4878,  85.4878,  99.5000,
#         112.5000, 124.5000, 135.5000, 145.5000, 154.5000, 162.5000, 180.5000,
#         189.5000, 169.5000, 175.5000, 184.5000, 187.5000, 190.5000])
# Selected objs: tensor([[0.1000, 0.2000],
#         [0.2000, 0.1000],
#         [0.1500, 0.1500],
#         [0.3000, 0.3000],
#         [0.3500, 0.2500],
#         [0.2500, 0.3500],
#         [0.4000, 0.4000],
#         [0.4500, 0.3500],
#         [0.3500, 0.4500],
#         [0.5000, 0.5000]])
# Selected fitness: tensor([  0.4497,  19.4497,  37.4497,  54.4878,  70.4878,  85.4878,  99.5000,
#         112.5000, 124.5000, 135.5000])
# test correct