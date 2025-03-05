import torch
import math
def cal_fitness(PopObj):
    N = PopObj.size(0)
    
    # Detect the dominance relation between each two solutions
    Dominate = torch.zeros((N, N), dtype=torch.bool)
    for i in range(N-1):
        for j in range(i+1, N):
            domij = torch.any(PopObj[i,:] < PopObj[j,:])
            domji = torch.any(PopObj[i,:] > PopObj[j,:])
            if domij:
                Dominate[i,j] = True
            elif domji:
                Dominate[j,i] = True

    # Calculate S(i)
    S = Dominate.sum(dim=1)
    
    # Calculate R(i)
    R = torch.zeros(N)
    for i in range(N):
        R[i] = S[Dominate[:,i]].sum()
    
    # Calculate the shifted distance between each two solutions
    Distance = torch.full((N, N), float('inf'))
    for i in range(N):
        SPopObj = torch.maximum(PopObj, PopObj[i,:].repeat(N, 1))
        for j in range(N):
            if j != i:
                Distance[i, j] = torch.norm(PopObj[i,:] - SPopObj[j,:])

    # Calculate D(i)
    Distance, idx = torch.sort(Distance, dim=1)
    aa=math.floor(math.sqrt(N))
    test =Distance[:, math.floor(math.sqrt(N))-1] 
    D = 1.0 / (Distance[:, math.floor(math.sqrt(N))-1] + 2)
    # D = 1. / (Distance[torch.arange(N), idx[:, int(torch.sqrt(torch.tensor(N)).item())]] + 2)
    
    # Calculate the fitnesses
    Fitness = R + D
    return Fitness, Dominate

if __name__ == '__main__':
# Test case
    PopObj = torch.tensor([
        [0.2, 0.3],
        [0.5, 0.6],
        [0.8, 0.9],
        [1.2, 1.3],
        [1.5, 1.6],
        [0.1, 0.2],
        [0.4, 0.5],
        [0.7, 0.8],
        [1.0, 1.1],
        [1.3, 1.4]
    ])
    # print(PopObj.shape)
    # Call the cal_fitness function to calculate fitness values
    Fitness, _ = cal_fitness(PopObj)

    # Display the fitness values
    print("Fitness values:")
    print(Fitness)
    # tensor([ 9.4125, 24.5000, 35.5000, 42.5000, 45.5000,  0.3898, 17.4670, 30.5000,
            # 39.5000, 44.5000])
    # test correct