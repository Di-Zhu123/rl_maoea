import numpy as np

def CalFitness(PopObj, kappa):
    N = PopObj.shape[0]
    # 归一化处理
    min_vals = np.min(PopObj, axis=0)
    max_vals = np.max(PopObj, axis=0)
    range_vals = max_vals - min_vals
    PopObj_norm = (PopObj - min_vals) / range_vals  # 归一化到[0,1]

    # 计算支配关系矩阵I
    I = np.max(PopObj_norm[:, np.newaxis, :] - PopObj_norm, axis=2)

    # 计算最大支配距离C
    C = np.max(np.abs(I))

    # 计算适应度值
    exponents = -I / (C * kappa)
    Fitness = np.sum(-np.exp(exponents), axis=1) + 1

    return Fitness, I, C

def cal_ibea(popobj):
    fitness, _, _ = CalFitness(popobj, 0.05)
    tot_fitness = np.sum(fitness)
    return tot_fitness
    

if __name__ == '__main__':
    pass