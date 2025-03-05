import jax
import jax.numpy as jnp
import evox
from evox.operators import selection, mutation, crossover
from evox.problems.numerical.dtlz import DTLZ2
from evox.metrics.hv import hv
import matplotlib.pyplot as plt
import os
from evox.workflows import StdWorkflow, EvalMonitor
import matplotlib.pyplot as plt
from evox.core import Algorithm, jit_class, Parameter, Mutable
from evox.operators import selection, mutation, crossover
from evox.problems.numerical.dtlz import DTLZ2
from evox.metrics.hv import hv
import torch
from evox.algorithms import MOEAD
from sde import cal_fitness
from envsel import environmental_selection
import time

@jit_class
class SPEA2SDE(Algorithm):
    def __init__(
        self,
        n_objs: int,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        tournament_size: int = 2,
        cr: float = 0.9,
        eta_c: float = 20,
        eta_m: float = 20,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.n_objs = Parameter(n_objs)
        self.pop_size = Parameter(pop_size)
        self.tournament_size = Parameter(tournament_size)
        self.cr = Parameter(cr)
        self.eta_c = Parameter(eta_c)
        self.eta_m = Parameter(eta_m)
        self.device = device

        # Convert bounds to tensors
        self.lb = lb.to(self.device)
        self.ub = ub.to(self.device)


        population = torch.rand(self.pop_size, self.n_objs, device=self.device)
        population = self.lb + (self.ub -self.lb) * population  # 缩放到 [0, 1]


        # Initialize fitness
        # fitness = torch.empty(pop_size, device=device)
        
        # Register mutable parameters
        self.population = Mutable(population)
        # self.fitness = Mutable(fitness)
        # self.f = self.evaluate(self.population)
        # self.fitness = self._calculate_fitness(self.f)
        self.f=torch.empty((pop_size,self.n_objs), device=self.device)
        self.fitness=torch.empty(pop_size, device=self.device)


    def step(self):
        # print(1)
        self.f = self.evaluate(self.population)
        self.fitness, dominance = self._calculate_fitness(self.f)
        # fitness = self.evaluate(self.population)
        # Tournament selection
        selected = selection.tournament_selection(
            n_round=self.pop_size,
            fitness=self.fitness,
            tournament_size=self.tournament_size
        )
        parents = self.population[selected]
        
        # Simulated binary crossover
        offspring = crossover.simulated_binary(
            x=parents
        )
        
        # Polynomial mutation
        offspring = mutation.polynomial_mutation(
            x=offspring,
            lb=self.lb,
            ub=self.ub
        )
        
        # Evaluate offspring
        offspring_f = self.evaluate(offspring)
        
        # Merge populations
        merged_f = torch.cat([self.f, offspring_f])
        merged_pop = torch.cat([self.population, offspring])
        # merged_fitness = self._calculate_fitness(merged_f)

        # torch.save(merged_fitness, '11.pt')

        # Environmental selection
        # selected_indices = self._environmental_selection(merged_fitness)

        _, _, selected_indices = self._environmental_selection(merged_f)

        # print('selected_indices:',selected_indices)
        # torch.save(selected_indices, '11.pt')
        new_pop = merged_pop[selected_indices]
        # new_fitness = merged_fitness[selected_indices]
        new_f = merged_f[selected_indices]
        
        # new_fitness = self._calculate_fitness(new_f)
        self.population = new_pop
        # self.fitness = new_fitness
        self.f = new_f
        
    


    def _calculate_fitness(self, population):
        
        return cal_fitness(population)

    def _environmental_selection(self, fitness):
        return environmental_selection(fitness, self.pop_size)

if __name__ == "__main__":
    # 问题配置
    
    n_objs = 3
    pop_size = 100
    max_gen = 1000
    problem = DTLZ2(m=n_objs)
    # 定义边界
    lb = torch.zeros(n_objs)
    ub = torch.ones(n_objs)
    
    # 初始化算法
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algorithm = SPEA2SDE(
        n_objs=n_objs,
        pop_size=pop_size,
        lb=lb,
        ub=ub,
        tournament_size=2,
        cr=0.9,
        eta_c=20,
        eta_m=20,
        device=device
    )
    # algorithm = MOEAD(
    #     n_objs=n_objs,
    #     pop_size=pop_size,
    #     lb=lb,
    #     ub=ub,
    #     device='cpu'
    # )
    
    # 创建Workflow
    monitor = EvalMonitor(full_sol_history=True)
    workflow = StdWorkflow()
    workflow.setup(algorithm=algorithm, problem=problem, monitor=monitor)
    HVs=[]
    step = 10
    # 运行优化
    for gen in range(max_gen):
        workflow.step()
        
        # 每10代计算并记录超体积
        if gen % step == 0:
            ref_point = torch.tensor([1] * n_objs, device=algorithm.device)
            monitor_vars = workflow.get_submodule("monitor")
            latest_solution = monitor_vars.latest_solution
            latest_fitness = monitor_vars.latest_fitness
            latest_solution = latest_solution.to(algorithm.device)
            latest_fitness = latest_fitness.to(algorithm.device)
            
            hv_metric = hv(latest_fitness, ref_point, num_sample=1000, device=algorithm.device)
            print(f"Generation {gen}: HV = {hv_metric}")

            HVs.append(hv_metric)
    
    # 保存最终结果
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pareto_front.png")
    
    # final_pop = workflow.get_population().cpu().numpy()
    final_solution = latest_solution.cpu().numpy()
    final_fitness = latest_fitness.cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.scatter(final_fitness[:, 0], final_fitness[:, 1], c='blue', label='Pareto Front')
    plt.title("SPEA2SDE on DTLZ2")
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Pareto前沿图已保存至: {output_file}")

    # 绘制并保存超体积曲线
    hv_output_file = os.path.join(output_dir, "hv_curve.png")
    plt.figure(figsize=(8, 6))
    HVs_cpu = [hv.cpu().numpy() for hv in HVs]
    plt.plot(range(0, max_gen, step), HVs_cpu, marker='o', linestyle='-', color='r')
    plt.title("Hypervolume over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.savefig(hv_output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"超体积曲线已保存至: {hv_output_file}")