import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import copy
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from pymoo.util.normalization import normalize

def test_contrib(env, problem, dim, model, n_episodes=10, reward_mode='hv_Percentage', n_generations=100):
    """
    测试保存的RL模型并输出最终HV指标
    
    参数:
    problem (str): 测试问题名称（如'dtlz2'）
    dim (int): 目标维度
    model_path (str): 模型保存路径
    n_episodes (int): 测试的独立运行次数
    reward_mode (str): 奖励模式（'hv'或'hv_Percentage'）
    
    返回:
    avg_hv (float): 平均HV值
    std_hv (float): HV值的标准差
    """
    hvs = []
    igds = []
    # env = DummyVecEnv([lambda: env])
    for episode in range(n_episodes):

        # 创建测试环境
        
        # 加载预训练模型
        
        obs = env.reset()
        done = [False]
        k=5
        
        vec_env = model.get_env()
        ref_dirs_pf = vec_env.envs[0].unwrapped.gym_env.ref_dirs_pf
        pf = get_problem(problem, n_var=dim+k-1, n_obj=dim).pareto_front(ref_dirs_pf)
        # while not done[0]:
        for i in range(n_generations):
            # here to save the final result
            if i == n_generations - 1:
                vec_env = model.get_env()
                test_env = vec_env.envs[0].unwrapped.gym_env
                algo = test_env.algorithm
                pop_last = algo.pop
                ref = test_env.ref_point_pymoo
                front = algo.result().F
                # ideal = front.min(axis=0)
                # nadir = front.max(axis=0)
                # front = normalize(front, ideal, nadir+0.1)
                # front = (front - ideal) / (nadir - ideal)
                final_hv = HV(ref_point=ref)(front)
                igd = IGD(pf)(front)

            action, _ = model.predict(obs, deterministic=True)
            #(1,91)
            obs, _, done, _ = env.step(action)
            # print(f"Gen: {env.envs[0].current_gen}, HV: {env.envs[0].indicator_last}")
        print('episode',episode,'igd',igd,'hv',final_hv)
    
        # plt.scatter(f[:, 0], f[:, 1])
        # plt.xlabel('Objective 1')
        # plt.ylabel('Objective 2')
        # plt.title('Population Objectives')
        # plt.savefig('res/pop_scatter.png')
        # plt.close()
        # 获取最终HV值
        # final_hv = indicator_last
        # 获取IGD值
        hvs.append(final_hv)
        igds.append(igd)
        env.close()   


    env.reset()
    # 计算统计结果
    avg_hv = np.mean(hvs)
    std_hv = np.std(hvs)
    
    avg_igd = np.mean(igds)
    std_igd = np.std(igds)


    # print(f"\n测试结果 ({n_episodes}次运行):")
    # print(f"平均HV: {avg_hv:.4f}")
    # print(f"标准差: {std_hv:.4f}")
    # print(f"平均IGD: {avg_igd:.4f}")
    # print(f"标准差: {std_igd:.4f}")
    
    return avg_hv, std_hv, avg_igd, std_igd

if __name__ == "__main__":
    from env_conti import conti_env
    # 测试参数设置
    PROBLEM = 'dtlz3'
    DIM = 2
    MODEL_PATH = "ppo_mlp_model_hv_percentage"  # 根据实际保存的模型路径修改
    MODEL_PATH = "conti_ppo_hv_Percentage_dtlz2_d2_gen500_nstep20"
    MODEL_PATH = "/home/zhu_di/workspace/rl_maoea/19ppo_igdh_dtlz2_d2_g2e3_ns10_4e4_0304"
    N_EPISODES = 2
    n_generations = 2000
    reward_mode = 'igdhv'
    env = conti_env(problem=PROBLEM, 
                dim=DIM, 
                n_generations=n_generations, 
                repetitions=1, 
                pop_size=91, 
                step_times=10, 
                reward_mode=reward_mode)
    # 执行测试
    model = PPO.load(MODEL_PATH)
    avg_hv, std_hv, avg_igd, std_igd = test_contrib(env, PROBLEM, DIM, model, N_EPISODES, reward_mode, n_generations)
    print('avg_hv', avg_hv, 'std_hv', std_hv, 'avg_igd', avg_igd, 'std_igd', std_igd)