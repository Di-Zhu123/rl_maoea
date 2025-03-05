import gym
from gym import spaces
import numpy as np
import torch
from pymoo.problems import get_problem
# import os
# import time
from pymoo.indicators.hv import HV
from pymoo.util.normalization import normalize
# from pymoo.algorithms.moo.rvea import RVEA
from rvea import RVEA
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from ibea import cal_ibea
from pymoo.indicators.igd import IGD
from test_contrib import test_contrib
from stable_baselines3 import PPO

# continous env
class conti_env(gym.Env):
    def __init__(self, problem, dim, n_generations=10, repetitions=1, pop_size=91, step_times=10, reward_mode='hv'):
        super(conti_env, self).__init__()
        self.problems = ['dtlz1', 'dtlz2', 'convex_dtlz2', 'dtlz5', 'dtlz7']
        self.problem = problem # self.problems中的一种或者all（采样）
        self.dim = dim
        self.n_generations = n_generations
        self.repetitions = repetitions
        self.pop_size = pop_size
        self.step_times = step_times

        # self.action_space = spaces.MultiDiscrete([pop_size] * pop_size)
        self.action_space = spaces.Box(low=0, high=1, shape=(pop_size, ))
        self.observation_space = spaces.Box(low=0, high=100, shape=(pop_size, dim))
        self.state = None
        self.reward_mode = reward_mode
        self.w1 = 10
        # self.ref_dirs = get_reference_directions("das-dennis", self.dim, n_partitions=self.pop_size-1)
        self.ref_dirs = get_reference_directions("das-dennis", self.dim, n_points=self.pop_size)
        self.ref_dirs_pf = get_reference_directions("das-dennis", self.dim, n_points=self.pop_size*10)
        self.algorithm = RVEA(ref_dirs = self.ref_dirs, pop_size=self.pop_size, n_offsprings=self.pop_size, n_generations=self.n_generations)
        self.reset()
    
    def reset(self):
        if self.problem == 'all':
            problem_train = np.random.choice(self.problems)
        else:
            problem_train = self.problem
        k=5 # DTLZ 的标准参数
        self.prob = get_problem(problem_train, n_var=self.dim + k - 1, n_obj=self.dim)
        # fl, fu = self.prob.obj_range()
        # self.ref_point_pymoo = np.ones(self.dim) * fu
        self.ref_point_pymoo = np.ones(self.dim)*1.1
        if problem_train == 'dtlz7':
            self.ref_point_pymoo = np.ones(self.dim)*15
        if problem_train == 'dtlz1':
            self.ref_point_pymoo = np.ones(self.dim)*400
        
        # print('ref_dirs shape', np.shape(ref_dirs))
        # self.algorithm = RVEA(ref_dirs = ref_dirs, pop_size=self.pop_size, n_offsprings=self.pop_size, n_generations=self.n_generations)
        self.algorithm.setup(self.prob, termination=('n_gen', self.n_generations), seed=42, verbose=False)
        self.current_gen = 0
        self.total_time = 0
        self.front = None
        self.indicator_last = -1
        self.reward = 0
        self.rewards = []
        self.truepf = self.prob.pareto_front(self.ref_dirs_pf)
        return self.get_observation()
    
    def step(self, action):
        # print('last front shape', np.shape(self.front))
        self.algorithm.load_action(action, self.current_gen, self.n_generations)
        self.algorithm.next()
        self.current_gen += 1
        
        
        # Get the front and compute hypervolume (HV)
        front = self.algorithm.result().F
        '''print('front shape', np.shape(front))
        ideal = front.min(axis=0)
        nadir = front.max(axis=0)
        self.front = normalize(front, ideal, nadir+0.1)
        front = (front - ideal) / (nadir - ideal)
        self.front = front'''
        if 0:
        # if self.indicator_last == -1:
            reward = 0
        else:
            if self.reward_mode == 'hv':
                hv_pymoo = HV(ref_point=self.ref_point_pymoo)(front)
                reward = hv_pymoo - self.indicator_last  # The reward is the true hypervolume
                self.indicator_last = hv_pymoo
            elif self.reward_mode ==  'hv_Percentage':
                hv_pymoo = HV(ref_point=self.ref_point_pymoo)(front)
                reward = hv_pymoo - self.indicator_last
                reward = reward / np.abs(self.indicator_last)
                self.indicator_last = hv_pymoo
            elif self.reward_mode ==  'ibea':
                indicator_now = cal_ibea(front)
                reward = indicator_now - self.indicator_last
                reward = reward / np.abs(self.indicator_last)
                self.indicator_last = indicator_now
            elif self.reward_mode ==  'igdhv':
                hv_pymoo = HV(ref_point=self.ref_point_pymoo)(front)
                igd_pymoo = -IGD(self.truepf)(front) # to minimize
                indicator_now = hv_pymoo + self.w1*igd_pymoo
                reward = (indicator_now - self.indicator_last)/ np.abs(self.indicator_last)
                self.indicator_last = indicator_now
        if reward > 1:
            reward = 1
        elif reward < -0.1:
            reward = -0.1
        if self.indicator_last == -1:
            reward = 0
        else:
            self.reward = reward
        self.rewards.append(reward)
        # Done condition: If the maximum number of generations is reached
        done = self.current_gen >= self.n_generations
        
        state2 = self.get_observation()
        # state2_size = np.shape(state2)
        # Return the next observation, reward, done, and info
        if 0:
            print('reward', reward, 'time', self.total_time)
        return state2, reward, done, {}
    
    def get_observation(self):
        if self.front is not None:
            # normalize all the function values for the front
            # Jim: ideal and nadir come from the entire population, so normalized F is not always on the bounds [0,1]
            # But in the extreme case it is at [0,1]
            front = self.algorithm.pop.get("F")
            # ideal = front.min(axis=0)
            # nadir = front.max(axis=0)
            # #
            # F_normalized = normalize(front, ideal, nadir+0.1)
            # self.state = F_normalized
            self.state = front
            # print('state shape', np.shape(front))
            return self.state
        else:
            self.state = np.zeros((self.pop_size, self.dim)) # (100, 3)
            return self.state

def to_sci_string(number):
    """
    将数字转换为紧凑的科学计数法字符串表示。
    
    参数:
        number (float or int): 需要转换的数字。
    
    返回:
        str: 紧凑的科学计数法字符串，例如 "2e3"。
    """
    # 使用科学计数法格式化，但保留指数部分的符号和多余的0
    scientific_str = f"{number:.0e}"
    # 移除指数部分的 '+' 符号和多余的 '0'
    base, exponent = scientific_str.split('e')
    exponent = exponent.replace('+', '').lstrip('0')  # 去掉 '+' 和前导 '0'
    return f"{base}e{exponent}"
       
if __name__ == '__main__':
    '''  date = '0303'
    # reward_mode = 'hv_Percentage'
    # reward_mode = 'ibea'
    reward_mode = 'igdhv'
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    problem = 'dtlz2'
    dim = 2
    n_generations = 2000
    n_steps = 20
    total_timesteps = 1e6
    save_path = "ppo_"+ reward_mode[0:4]+"_"+problem+"_d"+str(dim)+"_g"+str(n_generations)+"_ns"+str(n_steps)+'_'+str(int(total_timesteps))+'_'+date
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    
    env = conti_env(problem, dim,  n_generations=n_generations, repetitions=1, pop_size=91, step_times=10, reward_mode =reward_mode)
    env = DummyVecEnv([lambda: env])  # 将环境包装成vectorized环境
    
    # later try dqn
    ppo_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps, policy_kwargs=policy_kwargs) 
    ppo_model.learn(total_timesteps=total_timesteps, tb_log_name=save_path, log_interval=1, reset_num_timesteps=False)
    
    ppo_model.save(save_path)'''

    date = '0304'
    # reward_mode = 'hv_Percentage'
    # reward_mode = 'ibea'
    reward_mode = 'igdhv'

    problem = 'dtlz2'
    dim = 5
    n_generations = 10000
    n_steps = 10
    total_timesteps = 4e4

    test_step = 2e3
    N_test = int(total_timesteps // test_step)
    # print(to_sci_string(n_generations), to_sci_string(total_timesteps))

    save_path = "ppo_"+ reward_mode[0:4]+"_"+problem+"_d"+str(dim)+"_g"+to_sci_string(n_generations)+"_ns"+str(n_steps)+'_'+to_sci_string(int(total_timesteps))+'_'+date

    env = conti_env(problem, dim,  n_generations=n_generations, repetitions=1, pop_size=91, step_times=10, reward_mode =reward_mode)
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps, policy_kwargs=policy_kwargs) 

    for i in range(N_test):
    # for i in [0,1]:
        # i=0
        # test_step = 20
        model.learn(total_timesteps=test_step, tb_log_name=str(i)+save_path, log_interval=1, reset_num_timesteps=False)
        model.save(str(i)+save_path)
        vec_env = model.get_env()
        # vec_env = RecurrentPPO.load(str(i)+save_path)
        avg_hv, std_hv, avg_igd, std_igd = test_contrib(vec_env, problem, dim, model, n_episodes=5, reward_mode='igdhv', n_generations=n_generations)
        # print('avg_hv', avg_hv, 'std_hv', std_hv, 'avg_igd', avg_igd, 'std_igd', std_igd)
        with open("metric_log/metric_ppo.txt", "a") as file:
            file.write(str(i)+save_path)
            file.write(f"avg_hv: {avg_hv}")
            file.write(f"std_hv: {std_hv}")
            file.write(f"avg_igd: {avg_igd}")
            file.write(f"std_igd: {std_igd}\n")
    
    
    
    
    # if reward_mode == 'hv':
    #     ppo_model.save("ppo_mlp_model_hv")
    # elif reward_mode == 'hv_Percentage':
    #     ppo_model.save("ppo_mlp_model_hv_percentage")
    # ppo_model.save("ppo_mlp_model_hv")

