# the dependent packages or functions in this example
import time

import torch

from evox.algorithms import PSO, RVEA
# from rvea_evox import RVEA
from evox.problems.numerical import Ackley, DTLZ2
from evox.workflows import EvalMonitor, StdWorkflow
from evox.metrics import hv, igd

def filter_nan_rows(pop_f: torch.Tensor) -> torch.Tensor:
    """
    过滤掉二维张量中全为 NaN 的行
    Args:
        pop_f: shape (85, 5) 的二维张量
    Returns:
        过滤后的张量（保留非全 NaN 行）
    """
    # 创建掩码：标记非全 NaN 的行
    mask = ~torch.all(torch.isnan(pop_f), dim=1)
    
    # 使用掩码过滤行
    filtered_pop_f = pop_f[mask]
    
    return filtered_pop_f
# Initiate an algorithm
dim=5
k=5
pop_size = 85
max_gen=5000
d = dim + k - 1
ref = torch.tensor([1.1]*dim)
problem = DTLZ2(d=d,m=dim)
# problem = DTLZ2()
algorithm = RVEA(
    pop_size=pop_size,
    n_objs = dim,
    lb=torch.tensor([0]*d),
    ub=torch.tensor([1]*d),
    max_gen=max_gen,
)

# Initiate a problem

# Set an monitor
monitor = EvalMonitor()
# Initiate an workflow
workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
)
# compiled_step = torch.compile(workflow.step)
start = time.time()

# Run the workflow
workflow.init_step()
for i in range(max_gen//pop_size):
    # compiled_step() # torch.compile version of workflow.step()
    workflow.step()
    if (i + 1) % 10 == 0:
        run_time = time.time() - start
        # top_fitness = monitor.topk_fitness
        pop = workflow.algorithm.pop
        pop_f = workflow.algorithm.fit
        pop_f = filter_nan_rows(pop_f)
        hv_metric = hv(pop_f, ref)
        final_igd = igd(pop_f, workflow.problem.pf())
        print('hv',hv_metric, 'igd', final_igd, 'time', run_time)
        
