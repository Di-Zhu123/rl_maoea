# from rvea import RVEA
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
import numpy as np

dim=2
k=5
problem = get_problem("dtlz2", n_var=dim + k - 1, n_obj=dim)

ref_dirs = get_reference_directions("das-dennis", dim, n_partitions=90)
n_generations =2000
algorithm = RVEA(ref_dirs,pop_size=91, n_offsprings=91)
algorithm.setup(problem, termination=('n_gen', n_generations), seed=42, verbose=False)
        
for i in range(n_generations):
    algorithm.next()
    # len_test = len(algorithm.pop)
    # print(f"gen: {i+1}")
    # print(f"n_gen: {algorithm.n_gen}")
    # print(f"n_eval: {algorithm.evaluator.n_eval}")
    # print(f"n_offsprings: {algorithm.n_offsprings}")
# res = minimize(problem,
#                algorithm,
#                termination=('n_gen', 400),
#                seed=1,
#                verbose=False)
# res = minimize(problem,
#                algorithm,
#                ('n_gen', 100),
#                seed=1,
#                verbose=False)
# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, color="red")
# # plot.show()
# plot.save("output_filename0.png")
dim=2
pf = problem.pareto_front()
ind = IGD(pf)
f = algorithm.pop.get("F")
f = algorithm.result().F
igd = ind(f)
print(igd)
ind2 = HV(ref_point=np.ones(dim))
hv = ind2(f)
print(hv)
# plt.scatter(f[:, 0], f[:, 1])
# plt.xlabel('Objective 1')
# plt.ylabel('Objective 2')
# plt.title('Population Objectives')
# plt.savefig('res/pop_scatter.png')
# plt.close()