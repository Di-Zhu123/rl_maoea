# the dependent packages or functions in this example
import time

import torch

# from evox.algorithms import PSO, RVEA
from rvea_evox import RVEA
from evox.problems.numerical import Ackley, DTLZ2
from evox.workflows import EvalMonitor, StdWorkflow
from evox.metrics import hv, igd
# Initiate an algorithm
dim=5
k=5
pop_size = 91
max_gen=10000
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
compiled_step = torch.compile(workflow.step)
start = time.time()

# Run the workflow
workflow.init_step()
for i in range(100):
    compiled_step() # torch.compile version of workflow.step()
    # workflow.step()
    if (i + 1) % 10 == 0:
        run_time = time.time() - start
        # top_fitness = monitor.topk_fitness
        pop = workflow.algorithm.pop
        pop_f = workflow.algorithm.fit
        hv_metric = hv(pop_f, ref)
        print('hv',hv_metric, 'time', run_time)
        