import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import maoea_env
import copy
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
import matplotlib.pyplot as plt


def test_rl_model(problem, dim, model_path, n_episodes=10, reward_mode='hv_Percentage', n_generations=100):
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
    env = maoea_env(problem=problem, 
                    dim=dim, 
                    n_generations=n_generations, 
                    repetitions=1, 
                    pop_size=91, 
                    step_times=10, 
                    reward_mode=reward_mode)
    env = DummyVecEnv([lambda: env])
    for _ in range(n_episodes):
        # 创建测试环境
        
        # 加载预训练模型
        model = PPO.load(model_path)
        obs = env.reset()
        done = [False]
        
        # while not done[0]:
        for i in range(n_generations):
            # here to save the final result
            if i == n_generations - 1:
                pop_last = env.envs[0].algorithm.pop
                indicator_last = env.envs[0].indicator_last
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            # print(f"Gen: {env.envs[0].current_gen}, HV: {env.envs[0].indicator_last}")
        
        pf = get_problem(problem, n_var=2 * dim, n_obj=dim).pareto_front()
        ind = IGD(pf)
        f = pop_last.get("F")
        igd = ind(f)
        plt.scatter(f[:, 0], f[:, 1])
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Population Objectives')
        plt.savefig('res/pop_scatter.png')
        plt.close()
        # 获取最终HV值
        final_hv = indicator_last
        # 获取IGD值
        hvs.append(final_hv)
        igds.append(igd)
        env.close()
    
    
    # 绘制并保存HV图
    plt.plot(hvs)
    plt.xlabel('Episode')
    plt.ylabel('HV')
    plt.title('HV over Episodes')
    plt.savefig('res/hv_plot.png')
    plt.close()

    # 绘制并保存IGD图
    plt.plot(igds)
    plt.xlabel('Episode')
    plt.ylabel('IGD')
    plt.title('IGD over Episodes')
    plt.savefig('res/igd_plot.png')
    plt.close()



    # 计算统计结果
    avg_hv = np.mean(hvs)
    std_hv = np.std(hvs)
    
    avg_igd = np.mean(igds)
    std_igd = np.std(igds)


    print(f"\n测试结果 ({n_episodes}次运行):")
    print(f"平均HV: {avg_hv:.4f}")
    print(f"标准差: {std_hv:.4f}")
    print(f"平均IGD: {avg_igd:.4f}")
    print(f"标准差: {std_igd:.4f}")
    
    return avg_hv, std_hv

if __name__ == "__main__":
    # 测试参数设置
    PROBLEM = 'dtlz2'
    DIM = 2
    MODEL_PATH = "ppo_mlp_model_hv_percentage"  # 根据实际保存的模型路径修改
    MODEL_PATH = "conti_ppo_hv_Percentage_dtlz2_d2_gen500_nstep20"
    N_EPISODES = 30
    n_generations = 300
    reward_mode = 'hv_Percentage'
    # 执行测试
    test_rl_model(PROBLEM, DIM, MODEL_PATH, N_EPISODES, reward_mode, n_generations)