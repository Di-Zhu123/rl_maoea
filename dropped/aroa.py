import torch
from scipy.spatial.distance import pdist, squareform

def AROA(N, dim, lb, ub, fobj, maxEvals):
    c = torch.tensor(0.95)  # 转换为Tensor
    fr1 = torch.tensor(0.15)
    fr2 = torch.tensor(0.6)
    p1 = torch.tensor(0.2)
    p2 = torch.tensor(0.8)
    Ef = torch.tensor(0.4)
    tr1 = torch.tensor(0.9)
    tr2 = torch.tensor(0.85)
    tr3 = torch.tensor(0.9)

    
    
    tmax = int(torch.ceil(torch.tensor((maxEvals - N) / (2 * N))))
    evalCounter = 0
    Convergence_curve = torch.zeros(1, tmax)
    lb_tensor = torch.tensor(lb)
    ub_tensor = torch.tensor(ub)
    
    X = torch.rand(N, dim) * (ub - lb) + lb
    X, F, evalCounter = evaluate_population(X, fobj, ub, lb, evalCounter, maxEvals)
    fbest, ibest = torch.min(F, dim=0)
    xbest = X[ibest.item(), :].clone()
    X_memory = X.clone()
    F_memory = F.clone()
    
    for t in range(1, tmax+1):
        D = squareform(pdist(X.numpy(), 'sqeuclidean'))
        D = torch.from_numpy(D).float()
        m = tanh(t, tmax, [-2, 7])
        
        for i in range(N):
            Dimax = torch.max(D[i, :])
            k = int(torch.floor(torch.tensor((1 - t/tmax)*N))) + 1
            _, neighbors = torch.sort(D[i, :])
            
            delta_ni = torch.zeros(1, dim)
            for j in neighbors[:k]:
                I = 1 - (D[i, j]/Dimax)
                s = torch.sign(F[j] - F[i])
                delta_ni += c * (X_memory[i, :] - X_memory[j, :]) * I * s
            ni = delta_ni / N
            
            if torch.rand(1) < p1:
                bi = m * c * (torch.rand(1, dim) * xbest - X_memory[i, :])
                bi = bi[0,:]
            else:
                bi = m * c * (xbest - X_memory[i, :])
            
            if torch.rand(1) < p2:
                if torch.rand(1) > 0.5*t/tmax + 0.25:
                    u1 = (torch.rand(1, dim) > tr1).float()
                    ri = u1 * torch.normal(mean=torch.zeros(1, dim), std=fr1*(1 - t/tmax)*(ub - lb))
                else:
                    u2 = (torch.rand(1, dim) > tr2).float()
                    w = index_roulette_wheel_selection(F, k)
                    Xw = X_memory[w, :]
                    if torch.rand(1) < 0.5:
                        ri = fr2 * u2 * (1 - t/tmax) * torch.sin(2 * torch.pi * torch.rand(1, dim)) * torch.abs(torch.rand(1, dim) * Xw - X_memory[i, :])
                    else:
                        ri = fr2 * u2 * (1 - t/tmax) * torch.cos(2 * torch.pi * torch.rand(1, dim)) * torch.abs(torch.rand(1, dim) * Xw - X_memory[i, :])
            else:
                u3 = (torch.rand(1, dim) > tr3).float()
                ri = u3 * (2 * torch.rand(1, dim) - 1) * (ub - lb)

            # print(ni[0,:].shape, bi.shape, ri[0,:].shape, X[i, :].shape)
            # torch.Size([100]) torch.Size([100]) torch.Size([100]) torch.Size([100])

            X[i, :] += ni[0,:] + bi + ri[0,:]
        
        X, F, evalCounter = evaluate_population(X, fobj, ub, lb, evalCounter, maxEvals)
        fbest_candidate, ibest_candidate = torch.min(F, dim=0)
        if fbest_candidate < fbest:
            fbest = fbest_candidate
            xbest = X[ibest_candidate.item(), :].clone()
        
        X, F = memory_operator(X, F, X_memory, F_memory)
        X_memory = X.clone()
        F_memory = F.clone()
        
        CF = torch.tensor((1 - t/tmax) ** 3)
        if torch.rand(1) < Ef:
            u4 = (torch.rand(N, dim) < Ef).float()
            X += CF * u4 * (torch.rand(N, dim) * (ub - lb) + lb)
        else:
            r7 = torch.rand(1)
            X += (CF * (1 - r7) + r7) * (X[torch.randperm(N), :] - X[torch.randperm(N), :])
        
        X, F, evalCounter = evaluate_population(X, fobj, ub, lb, evalCounter, maxEvals)
        fbest_candidate, ibest_candidate = torch.min(F, dim=0)
        if fbest_candidate < fbest:
            fbest = fbest_candidate
            xbest = X[ibest_candidate.item(), :].clone()
        
        X, F = memory_operator(X, F, X_memory, F_memory)
        X_memory = X.clone()
        F_memory = F.clone()
        Convergence_curve[0, t-1] = fbest
    
    return fbest, xbest, Convergence_curve

def evaluate_population(X, fobj, ub, lb, evalCounter, maxEvals):
    N, dim = X.shape
    F = torch.full((N,), float('inf'))
    X = torch.max(lb, torch.min(ub, X))
    
    for i in range(N):
        if evalCounter >= maxEvals:
            break
        F[i] = fobj(X[i, :])
        evalCounter += 1
    
    return X, F, evalCounter

def memory_operator(X, F, X_memory, F_memory):
    Inx = (F_memory < F).unsqueeze(-1)
    X = torch.where(Inx, X_memory, X)
    F = torch.where(Inx.squeeze(), F_memory, F)
    return X, F

def tanh(t, tmax, bounds):
    z = torch.tensor((t - tmax) / tmax * (bounds[1] - bounds[0]) + bounds[0])
    return 0.5 * ((torch.exp(z) - 1) / (torch.exp(z) + 1) + 1)

def index_roulette_wheel_selection(F, k):
    fitness = F[:k]
    weights = torch.max(fitness) - fitness
    weights = torch.cumsum(weights / torch.sum(weights), dim=0)
    return roulette_wheel_selection(weights)

def roulette_wheel_selection(weights):
    r = torch.rand(1).item()
    for idx in range(len(weights)):
        if r <= weights[idx].item():  # 转换为float进行比较
            return idx
    return len(weights)-1
'''
# # 主调用示例
# if __name__ == "__main__":
#     SearchAgents_no = 40
#     Max_FES = 30000
#     NoRepeats = 30
#     dim = 100
#     lb = -5.12 * torch.ones(dim)
#     ub = 5.12 * torch.ones(dim)
    
#     outputF = []
#     outputX = []
    
#     def F22(x):
#         return torch.sum(x**2).item()
    
#     for _ in range(NoRepeats):
#         Best_score, Best_pos, cg_curve = AROA(
#             SearchAgents_no, dim, lb, ub, F22, Max_FES
#         )
#         outputF.append(Best_score)
#         outputX.append(Best_pos)
    
#     outputF = torch.tensor(outputF)
#     print(f"Minimum: {torch.min(outputF)}")
#     print(f"Mean: {torch.mean(outputF)}")
#     print(f"Std: {torch.std(outputF)}")
'''
import torch

# Fixing the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
# Running the algorithm
SearchAgents_no = 40
Max_FES = 30000
dim = 100
lb = -5.12 * torch.ones(dim)
ub = 5.12 * torch.ones(dim)

def F22(x):
    return torch.sum(x**2).item()

Best_score, Best_pos, cg_curve = AROA(SearchAgents_no, dim, lb, ub, F22, Max_FES)
print(Best_score, Best_pos)

import matplotlib.pyplot as plt
# 绘制 cg_curve 数据
plt.plot(cg_curve)
plt.xlabel('itee')
plt.ylabel('score')
plt.title('cd_curve')
plt.savefig('cg_curve.png')