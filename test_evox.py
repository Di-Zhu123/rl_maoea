import time
import numpy as np
import torch
from rvea_evox import RVEA
from evox.problems.numerical import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from evox.workflows import StdWorkflow, EvalMonitor
from evox.metrics import hv, igd
from sbx import PPO
from hv_norm import hv_normalized


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
    problem_dict = {
        'dtlz1': DTLZ1,
        'dtlz2': DTLZ2,
        'dtlz3': DTLZ3,
        'dtlz4': DTLZ4,
        'dtlz5': DTLZ5,
        'dtlz6': DTLZ6,
        'dtlz7': DTLZ7,
    }
    if problem_name == 'all':
        problem_name = 'dtlz2'
    problem_class = problem_dict[problem_name]

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
            monitor=monitor,
            device='cuda'
        )

        # 运行优化过程
        workflow.init_step()
        # obs = env.reset()
        vec_env = model.get_env()
        initial = vec_env.envs[0].reset()
        obs = initial[0] #(85,5).在训练时，action和obs没有在dummpy下；但是测试时是在dummpy下的
        obs = np.array([obs])

        start_time = time.time()
        for i in range(n_generations):
            if i % 1000 == 0:
                print(i, '/', n_generations)
            # if i == n_generations - 1:
            if 1:
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
                # final_hv = hv(ref, front)
                optimum = true_pf.max(dim=0).values
                final_hv = hv_normalized(front, optimum)
                final_igd = igd(front, true_pf)
                # print(f"Run {run+1}: HV={final_hv:.4f}, IGD={final_igd:.4f}, time={time.time()-start_time:.2f}s")
                # if final_hv >10:
                #     front_max=torch.max(front)

            action, _ = model.predict(obs, deterministic=True)
            #(85,)
            obs, _, done, _ = env.step(action)
        print(f"Run {run+1}: HV={final_hv:.4f}, IGD={final_igd:.4f}, time={time.time()-start_time:.2f}s")
        hvs.append(final_hv)
        igds.append(final_igd)
    # 计算统计结果
    avg_hv = np.mean(hvs)
    std_hv = np.std(hvs)
    avg_igd = np.mean(igds)
    std_igd = np.std(igds)

    return avg_hv, std_hv, avg_igd, std_igd

if __name__ == "__main__":
    from env_evox import EvoXContiEnv, to_sci_string
    date = '0309'
    # reward_mode = 'hv_Percentage'
    # reward_mode = 'ibea'
    # reward_mode = 'igdhv'
    reward_mode = 'log_smooth' # 1.8251, 0.0142, 0.0361, 0.0119, 0.0022, -0.0050, -0.0456, 0.0001, 0.0034, 0.0099, -0.0119, 0.0292, 0.0468, 0.0142, -0.0059, 0.0062, 0.0326, 0.0018, -0.0120, 0.0156, -0.0277, -0.0069, 0.0032, -0.0166, 0.0130, 0.0137
    pop_size = 85
    # problem = 'dtlz2'
    # problem = 'maf2'
    problem = 'all'
    dim = 5
    n_generations = 5000
    # n_generations = 500
    n_steps = 20
    total_timesteps = 4e5

    test_step = 2e4
    N_test = int(total_timesteps // test_step)
    # print(to_sci_string(n_generations), to_sci_string(total_timesteps))

    save_path = "ppo_"+ reward_mode[0:4]+"_"+problem+"_d"+str(dim)+"_g"+to_sci_string(n_generations)+"_ns"+str(n_steps)+'_'+to_sci_string(int(total_timesteps))+'_'+date

    env = EvoXContiEnv(problem, dim,  n_generations=n_generations, pop_size=pop_size, reward_mode =reward_mode)
    # policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps, policy_kwargs=policy_kwargs) 
    # model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps) 
    # model = PPO.load("/home/zhu_di/workspace/rl_maoea/models/19pj_log__all_d5_g5e3_ns20_4e5_0308.zip", env=env, device='cuda') 
    # avg_hv 2.2611856 std_hv 5.8645525 avg_igd 1.1093122 std_igd 1.4770586
    model = PPO.load("/home/zhu_di/workspace/rl_maoea/models/pj_19log__all_d5_g5e3_ns20_4e5_0310.zip", env=env, device='cuda') 
    avg_hvs =[]
    avg_igds = []
    hv_tolence = 0.003
    igd_tolence = 0.003
    # for i in range(N_test):
    for i in [0]:
        # i=0
        # test_step = 200
        # model.learn(total_timesteps=test_step, tb_log_name=str(i+1)+save_path, log_interval=1, reset_num_timesteps=False)
        # model.save("models/"+str(i)+save_path)
        vec_env = model.get_env()
        # # vec_env = RecurrentPPO.load(str(i)+save_path)
        # # print(vec_env.unwrapped.envs[0].rewards)
        # print(i)
        
        avg_hv, std_hv, avg_igd, std_igd = test_evox(vec_env, problem, dim, model, n_generations=n_generations, n_runs=10, reward_mode=reward_mode)
        print('avg_hv', avg_hv, 'std_hv', std_hv, 'avg_igd', avg_igd, 'std_igd', std_igd)
        # with open("metric_log/metric_ppo.txt", "a") as file:
        #     file.write(str(i)+save_path)
        #     file.write(f"avg_hv: {avg_hv}")
        #     file.write(f"std_hv: {std_hv}")
        #     file.write(f"avg_igd: {avg_igd}")
        #     file.write(f"std_igd: {std_igd}\n")
        # if len(avg_hvs) > 0:
        #     if np.abs(avg_hv-avg_hvs[-1])<hv_tolence and np.abs(avg_igd-avg_igds[-1])<igd_tolence:
        #         print('hv and igd converge')
        #         break
        # avg_hvs.append(avg_hv)
        # avg_igds.append(avg_igd)