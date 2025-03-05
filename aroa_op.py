import torch
from scipy.spatial.distance import pdist, squareform
from pymoo.core.population import Population

def aroaop(pop, t, maxEvals, action, ub, lb):

    """
    Generate a new population using the AROA operator for one generation.
    
    Args:
        pop (Population): Current population (Pymoo Population object).
        t (int): Current generation number.
        maxEvals (int): Maximum number of evaluations.
        action (np.ndarray): 1D array of scalar fitness metrics for each individual.
    
    Returns:
        off (Population): New population after one generation of AROA evolution.
    """
    # Algorithm parameters
    c = torch.tensor(0.35)  # Convergence factor
    fr1 = torch.tensor(0.15)  # Random factor 1
    fr2 = torch.tensor(0.6)   # Random factor 2
    p1 = torch.tensor(0.2)    # Probability for best individual influence
    p2 = torch.tensor(0.8)    # Probability for random movement
    tr1 = torch.tensor(0.9)   # Threshold 1 for random perturbation
    tr2 = torch.tensor(0.85)  # Threshold 2 for trigonometric perturbation
    tr3 = torch.tensor(0.9)   # Threshold 3 for uniform perturbation

    # Extract decision variables from population
    X = torch.tensor(pop.get("X"), dtype=torch.float32)
    N, dim = X.shape  # N: population size, dim: problem维度

    # Convert action to tensor
    action = torch.tensor(action, dtype=torch.float32)

    # Calculate maximum iterations (for scaling purposes)
    tmax = int(torch.ceil(torch.tensor((maxEvals - N) / (2 * N))))
    if tmax < 1:
        tmax = 1
    # added in 2.28
    if t/tmax>=1:
        tmax = t

    # Compute distance matrix
    D = squareform(pdist(X.numpy(), 'sqeuclidean'))
    D = torch.from_numpy(D).float()

    # Compute m using tanh function
    m = tanh(t, tmax, [-2, 7])

    # Update each individual
    for i in range(N):
        Dimax = torch.max(D[i, :])  # Maximum distance for individual i
        k = int(torch.floor(torch.tensor((1 - t/tmax) * N))) + 1  # Number of neighbors
        
        k=2
        
        _, neighbors = torch.sort(D[i, :])  # Sort neighbors by distance

        # Neighbor-based movement (ni)
        delta_ni = torch.zeros(1, dim)
        for j in neighbors[:k]:
            I = 1 - (D[i, j] / Dimax)  # Distance-based influence
            s = torch.sign(action[j] - action[i])  # Direction based on action
            if np.abs(action[j] - action[i])<0.001:
                s=0
            delta_ni += c * (X[i, :] - X[j, :]) * I * s
        ni = delta_ni / k

        # Best individual influence (bi)
        if torch.rand(1) < p1:
            bi = m * c * (torch.rand(1, dim) * X[torch.argmin(action), :] - X[i, :])
            bi = bi[0, :]
        else:
            bi = m * c * (X[torch.argmin(action), :] - X[i, :])

        # Random movement (ri)
        if torch.rand(1) < p2:
            # if torch.rand(1) > 0.5 * t / tmax + 0.25:
            if 1:
                u1 = (torch.rand(1, dim) > tr1).float()
                std = fr1 * (1 - t/tmax) * (ub - lb)  # Assuming ub, lb are global bounds
                ri = u1 * torch.normal(mean=torch.zeros(1, dim), std=std)
            else:
                u2 = (torch.rand(1, dim) > tr2).float()
                w = index_roulette_wheel_selection(action, k)
                Xw = X[w, :]
                if torch.rand(1) < 0.5:
                    ri = fr2 * u2 * (1 - t/tmax) * torch.sin(2 * torch.pi * torch.rand(1, dim)) * torch.abs(torch.rand(1, dim) * Xw - X[i, :])
                else:
                    ri = fr2 * u2 * (1 - t/tmax) * torch.cos(2 * torch.pi * torch.rand(1, dim)) * torch.abs(torch.rand(1, dim) * Xw - X[i, :])
            ri = ri[0, :]
        else:
            u3 = (torch.rand(1, dim) > tr3).float()
            ri = u3 * (2 * torch.rand(1, dim) - 1) * (ub - lb)  # Assuming ub, lb are global bounds
            ri = ri[0, :]

        # Update individual position
        X[i, :] += ni[0, :] + bi + ri
        X[i, :]  = torch.clamp(X[i, :] , min=torch.from_numpy(lb), max=torch.from_numpy(ub))
    # Create new population with updated positions
    off = Population.new("X", X.numpy())

    return off

# Helper functions
def tanh(t, tmax, bounds):
    """Compute the tanh scaling factor."""
    z = torch.tensor((t - tmax) / tmax * (bounds[1] - bounds[0]) + bounds[0])
    return 0.5 * ((torch.exp(z) - 1) / (torch.exp(z) + 1) + 1)

def index_roulette_wheel_selection(action, k):
    """Select an index using roulette wheel selection based on action values."""
    fitness = action[:k]
    weights = torch.max(fitness) - fitness  # Invert for minimization
    weights = torch.cumsum(weights / torch.sum(weights), dim=0)
    return roulette_wheel_selection(weights)

def roulette_wheel_selection(weights):
    """Perform roulette wheel selection."""
    r = torch.rand(1).item()
    for idx in range(len(weights)):
        if r <= weights[idx].item():
            return idx
    return len(weights) - 1

from pymoo.core.population import Population
import numpy as np


if __name__ == "__main__":
    # Example setup
    N = 40
    dim = 100
    X = np.random.uniform(-5.12, 5.12, (N, dim))
    pop = Population.new("X", X)
    t = 1
    maxEvals = 30000
    action = np.random.rand(N)  # Dummy action values
    ub = 5.12 * np.ones(dim)  # Upper bounds
    lb = -5.12 * np.ones(dim)  # Lower bounds
    # Run aroaop
    off = aroaop(pop, t, maxEvals, action, ub, lb)
    print(off.get("X").shape)  # Should print (40, 100)