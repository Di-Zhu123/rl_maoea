import gym
from gym import spaces
import numpy as np
import torch
from pymoo.problems import get_problem
import os
import time
from pymoo.indicators.hv import HV
from pymoo.util.normalization import normalize
# from pymoo.algorithms.moo.rvea import RVEA
from rvea import RVEA
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from ibea import cal_ibea
from pymoo.indicators.igd import IGD


class maoea_env(gym.Env):
    def __init__(self, problem, dim, n_generations=10, repetitions=1, pop_size=91, step_times=10, reward_mode='hv'):
        super(maoea_env, self).__init__()
        self.problems = ['dtlz1', 'dtlz2', 'convex_dtlz2', 'dtlz5', 'dtlz7']
        self.problem = problem # self.problems中的一种或者all（采样）
        self.dim = dim
        self.n_generations = n_generations
        self.repetitions = repetitions
        self.pop_size = pop_size
        self.step_times = step_times

        self.action_space = spaces.MultiDiscrete([pop_size] * pop_size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(pop_size, dim))
        self.state = None
        self.reward_mode = reward_mode
        self.reset()
    
    def reset(self):
        if self.problem == 'all':
            problem_train = np.random.choice(self.problems)
        else:
            problem_train = self.problem
        self.prob = get_problem(problem_train, n_var=2 * self.dim, n_obj=self.dim)
        self.ref_point_pymoo = np.ones(self.dim)
        if problem_train == 'dtlz7':
            self.ref_point_pymoo = np.ones(self.dim)*15
        if problem_train == 'dtlz1':
            self.ref_point_pymoo = np.ones(self.dim)*400
        ref_dirs = get_reference_directions("das-dennis", self.dim, n_partitions=self.pop_size-1)
        # print('ref_dirs shape', np.shape(ref_dirs))
        self.algorithm = RVEA(ref_dirs = ref_dirs, pop_size=self.pop_size, n_offsprings=self.pop_size, n_generations=self.n_generations)
        self.algorithm.setup(self.prob, termination=('n_gen', self.n_generations), seed=42, verbose=False)
        self.current_gen = 0
        self.total_time = 0
        self.front = None
        self.indicator_last = -1
        self.reward = 0
        self.rewards = []
        self.truepf = self.prob.pareto_front()
        return self.get_observation()
    
    def step(self, action):
        # print('last front shape', np.shape(self.front))
        self.algorithm.load_action(action, self.current_gen, self.n_generations)
        self.algorithm.next()
        self.current_gen += 1
        
        
        # Get the front and compute hypervolume (HV)
        front = self.algorithm.result().F
        # print('front shape', np.shape(front))
        ideal = front.min(axis=0)
        nadir = front.max(axis=0)
        self.front = normalize(front, ideal, nadir+0.1)
        front = (front - ideal) / (nadir - ideal)
        # self.front = front
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
            ideal = front.min(axis=0)
            nadir = front.max(axis=0)
            #
            F_normalized = normalize(front, ideal, nadir+0.1)
            self.state = F_normalized
            # print('state shape', np.shape(front))
            return self.state
        else:
            self.state = np.zeros((self.pop_size, self.dim)) # (100, 3)
            return self.state
        
if __name__ == '__main__':
    date = '0302'
    reward_mode = 'hv_Percentage'
    # reward_mode = 'ibea'
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    problem = 'dtlz2'
    dim = 2
    n_generations = 2000
    n_steps = 30
    total_timesteps = 10e7
    env = maoea_env(problem, dim,  n_generations=n_generations, repetitions=1, pop_size=91, step_times=10, reward_mode =reward_mode)
    env = DummyVecEnv([lambda: env])  # 将环境包装成vectorized环境
    save_path = "ppo_"+ reward_mode[0:4]+"_"+problem+"_d"+str(dim)+"_g"+str(n_generations)+"_ns"+str(n_steps)+'_'+str(int(total_timesteps))+'_'+date
    # later try dqn
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    ppo_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps, policy_kwargs=policy_kwargs) 
    ppo_model.learn(total_timesteps=total_timesteps, tb_log_name=save_path, log_interval=1, reset_num_timesteps=False)
    
    ppo_model.save(save_path)
    # if reward_mode == 'hv':
    #     ppo_model.save("ppo_mlp_model_hv")
    # elif reward_mode == 'hv_Percentage':
    #     ppo_model.save("ppo_mlp_model_hv_percentage")
    # ppo_model.save("ppo_mlp_model_hv")
