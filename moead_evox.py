import math
from typing import Callable, Optional

import torch

from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary_half
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, minimum
from aroa_op import aroaop, aroaop_single

def pbi(f: torch.Tensor, w: torch.Tensor, z: torch.Tensor):
    # f (8, 5) w (8, 5) z (9,5)
    norm_w = torch.linalg.norm(w, dim=1)
    f = f - z

    d1 = torch.sum(f * w, dim=1) / norm_w
    # d1
    d2 = torch.linalg.norm(f - (d1[:,None] * w / norm_w[:,None]), dim=1)
    return d1 + 5 * d2


class MOEAD_evox(Algorithm):
    """
    Implementation of the Original MOEA/D algorithm.

    :references:
        - "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," IEEE Transactions on Evolutionary Computation.
          `Link <https://ieeexplore.ieee.org/document/4358754>`_


    :note: This implementation is based on the original paper and may not be the most efficient implementation. It can not be traced by JIT.

    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
        max_gen: int = 100,  # 添加 max_gen 参数
    ):
        """Initializes the MOEA/D algorithm.

        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions in the optimization problem.
        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param selection_op: The selection operation for evolutionary strategy (optional).
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param crossover_op: The crossover operation, defaults to `simulated_binary_half` if not provided (optional).
        :param device: The device on which computations should run (optional). Defaults to PyTorch's default device.
        """
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        if device is None:
            device = torch.get_default_device()
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]
        # write to self
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)
        
        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op
        self.device = device
        

        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary_half

        w, _ = uniform_sampling(self.pop_size, self.n_objs)

        self.pop_size = w.size(0)
        self.n_neighbor = int(math.ceil(self.pop_size / 10))

        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        neighbors = torch.cdist(w, w)
        self.neighbors = torch.argsort(neighbors, dim=1, stable=True)[:, : self.n_neighbor]
        self.w = w

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.z = Mutable(torch.zeros((self.n_objs,), device=device))
        self.action = None  # 添加 action 属性
        self.gen = Mutable(torch.tensor(0, dtype=int, device=device))  # 添加 gen 属性
        self.max_gen = max_gen  # 添加 max_gen 属性

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0)[0]

    def step(self):
        """Perform a single optimization step of the workflow."""
        self.gen = self.gen + torch.tensor(1)
        for i in range(self.pop_size): 
            # Note：一次生成一个子代
            parents = self.neighbors[i][torch.randperm(self.n_neighbor, device=self.device)]
            # todo: 
            mask = parents != i
            others = parents[mask]
            i_elements = parents[~mask]

            # 拼接为新 tensor，其中 i 元素位于最后
            parents = torch.cat([others, i_elements])
            crossovered = aroaop_single(self.pop[parents], t=self.gen, maxEvals=self.max_gen, action=self.action[parents], ub=self.ub, lb=self.lb, mode='evox')
            parents = parents[:-1]
            
            # crossovered = aroaop(self.pop, t=self.gen, maxEvals=self.max_gen, action=self.action, ub=self.ub, lb=self.lb, mode='evox')
            # crossovered = self.crossover(self.pop[parents[:2]])
            offspring = self.mutation(crossovered, self.lb, self.ub)
            offspring = clamp(offspring, self.lb, self.ub)
            off_fit = self.evaluate(offspring)

            self.z = minimum(self.z, off_fit)

            g_old = pbi(self.fit[parents], self.w[parents], self.z)
            g_new = pbi(off_fit, self.w[parents], self.z)

            self.fit[parents[g_old >= g_new]] = off_fit
            self.pop[parents[g_old >= g_new]] = offspring
            
    def load_action(self, action):
        """加载外部动作"""
        self.action = action
