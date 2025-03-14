import numpy as np
# from env_conti import conti_env, to_sci_string

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from test_contrib import test_contrib
from env_evox import EvoXContiEnv, to_sci_string
from test_evox import test_evox

import warnings
import torch
torch._dynamo.config.cache_size_limit = 512  # 增大缓存限制
torch._dynamo.config.suppress_errors = True   # 静默处理错误
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor*")
warnings.filterwarnings('ignore', category=DeprecationWarning)

date = '0308'
# reward_mode = 'hv_Percentage'
# reward_mode = 'ibea'
# reward_mode = 'igdhv'
reward_mode = 'log_smooth' # 1.8251, 0.0142, 0.0361, 0.0119, 0.0022, -0.0050, -0.0456, 0.0001, 0.0034, 0.0099, -0.0119, 0.0292, 0.0468, 0.0142, -0.0059, 0.0062, 0.0326, 0.0018, -0.0120, 0.0156, -0.0277, -0.0069, 0.0032, -0.0166, 0.0130, 0.0137
pop_size = 85
problem = 'all'
dim = 5
n_generations = 5000
n_steps = 20
total_timesteps = 4e5

test_step = 2e4
N_test = int(total_timesteps // test_step)
# print(to_sci_string(n_generations), to_sci_string(total_timesteps))

save_path = reward_mode[0:4]+"_"+problem+"_d"+str(dim)+"_g"+to_sci_string(n_generations)+"_ns"+str(n_steps)+'_'+to_sci_string(int(total_timesteps))+'_'+date

env = EvoXContiEnv(problem, dim,  n_generations=n_generations, pop_size=pop_size, reward_mode =reward_mode)
policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps, policy_kwargs=policy_kwargs) 
# model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps) 
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps)

avg_hvs =[]
avg_igds = []
hv_tolence = 0.003
igd_tolence = 0.003
for i in range(N_test):
# for i in [0,1]:
    # i=0
    # test_step = 200
    model.learn(total_timesteps=test_step, tb_log_name="pr_"+ str(i)+save_path, log_interval=1, reset_num_timesteps=False)
    model.save("models/"+"pr_"+ str(i)+save_path)
    vec_env = model.get_env()
    # vec_env = RecurrentPPO.load("models/"+str(i)+save_path)
    # todo: write test_contrib for RecurrentPPO
    
    
    # print(vec_env.unwrapped.envs[0].rewards)
    # print(i)
    # avg_hv, std_hv, avg_igd, std_igd = test_evox(vec_env, problem, dim, model, n_generations=n_generations, n_runs=5, reward_mode=reward_mode)
    # print('avg_hv', avg_hv, 'std_hv', std_hv, 'avg_igd', avg_igd, 'std_igd', std_igd)
    # with open("metric_log/metric_pr.txt", "a") as file:
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


