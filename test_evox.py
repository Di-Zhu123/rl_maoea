import time
import numpy as np
import torch
from evox.algorithms import RVEA
from evox.problems.numerical import DTLZ1, DTLZ2, DTLZ5, DTLZ7
from evox.workflows import StdWorkflow, EvalMonitor
from evox.metrics import hv, igd


def test_evox(env, problem_name, dim, model, n_generations=100, n_runs=10, reward_mode='igdhv'):
    """
    使用EvoX框架测试多目标优化算法性能

    Args:
        problem_name (str): 问题名称 ('dtlz1', 'dtlz2', 'dtlz5', 'dtlz7', 'convex_dtlz2')
        dim (int): 目标维度
        n_generations (int): 最大进化代数
        n_runs (int): 独立运行次数
        ref_point (list): 参考点（HV计算使用）
    Returns:
        avg_hv (float): 平均超体积指标
        std_hv (float): HV标准差
        avg_igd (float): 平均IGD指标
        std_igd (float): IGD标准差
    """
    hvs = []
    igds = []
    k = 5  # DTLZ标准参数
    d = dim + k - 1
    pop_size = 85

    
    # 获取真实Pareto前沿
    problem_class = {
        'dtlz1': DTLZ1,
        'dtlz2': DTLZ2,
        'dtlz5': DTLZ5,
        'dtlz7': DTLZ7
    }[problem_name]

    true_pf = problem_class(d=d, m=dim).pf()

    for run in range(n_runs):

        # 初始化问题和算法
        problem = problem_class(d=d, m=dim)
        algorithm = RVEA(
            pop_size=pop_size,
            n_objs=dim,
            lb=torch.zeros(d),
            ub=torch.ones(d),
            max_gen=n_generations,
        )
        
        monitor = EvalMonitor()
        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=problem,
            monitor=monitor
        )

        # 运行优化过程
        workflow.init_step()
        # obs = env.reset()
        vec_env = model.get_env()
        initial = vec_env.envs[0].reset()
        obs = initial[0] #(85,5).在训练时，action和obs没有在dummpy下；但是测试时是在dummpy下的
        obs = np.array([obs])

        # start_time = time.time()
        for i in range(n_generations):
            if i % 200 == 0:
                print(i, '/', n_generations)
            if i == n_generations - 1:
                vec_env = model.get_env()
                test_env = vec_env.envs[0].unwrapped.gym_env
                algo = test_env.workflow.algorithm
                pop_last = algo.pop
                ref = test_env.ref_point
                front = algo.fit
                # ideal = front.min(axis=0)
                # nadir = front.max(axis=0)
                # front = normalize(front, ideal, nadir+0.1)
                # front = (front - ideal) / (nadir - ideal)
                final_hv = hv(ref, front)
                final_igd = igd(front, true_pf)

            action, _ = model.predict(obs, deterministic=True)
            #(85,)
            obs, _, done, _ = env.step(action)
        print(f"Run {run+1}: HV={final_hv:.4f}, IGD={final_igd:.4f}")
        hvs.append(final_hv)
        igds.append(final_igd)
    # 计算统计结果
    avg_hv = np.mean(hvs)
    std_hv = np.std(hvs)
    avg_igd = np.mean(igds)
    std_igd = np.std(igds)

    return avg_hv, std_hv, avg_igd, std_igd

if __name__ == "__main__":
    # 测试参数设置
    PROBLEM = 'dtlz2'
    DIM = 2
    N_GENERATIONS = 1000
    N_RUNS = 10

    # 设置参考点
    if PROBLEM == 'dtlz7':
        ref_point = [15.0] * DIM
    elif PROBLEM == 'dtlz1':
        ref_point = [400.0] * DIM
    else:
        ref_point = [1.1] * DIM

    avg_hv, std_hv, avg_igd, std_igd = test_evox(
        problem_name=PROBLEM,
        dim=DIM,
        n_generations=N_GENERATIONS,
        n_runs=N_RUNS,
    )

    print(f"\n{N_RUNS}次运行结果:")
    print(f"平均HV: {avg_hv:.4f} ± {std_hv:.4f}")
    print(f"平均IGD: {avg_igd:.4f} ± {std_igd:.4f}")