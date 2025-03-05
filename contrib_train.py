import numpy as np
from env_conti import conti_env, to_sci_string

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from test_contrib import test_contrib



date = '0304'
# reward_mode = 'hv_Percentage'
# reward_mode = 'ibea'
reward_mode = 'igdhv'

problem = 'dtlz2'
dim = 2
n_generations = 2000
n_steps = 20
total_timesteps = 1e6

test_step = 5e4
N_test = int(total_timesteps // test_step)
# print(to_sci_string(n_generations), to_sci_string(total_timesteps))

save_path = "pr_"+ reward_mode[0:4]+"_"+problem+"_d"+str(dim)+"_g"+to_sci_string(n_generations)+"_ns"+str(n_steps)+'_'+to_sci_string(int(total_timesteps))+'_'+date

env = conti_env(problem, dim,  n_generations=n_generations, repetitions=1, pop_size=91, step_times=10, reward_mode =reward_mode)
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)

for i in range(N_test):
# for i in [0,1]:
    # i=0
    # test_step = 20
    model.learn(total_timesteps=test_step, tb_log_name=str(i)+save_path, log_interval=1, reset_num_timesteps=False)
    model.save(str(i)+save_path)
    vec_env = model.get_env()
    # vec_env = RecurrentPPO.load(str(i)+save_path)
    avg_hv, std_hv, avg_igd, std_igd = test_contrib(vec_env, problem, dim, model, n_episodes=10, reward_mode='igdhv', n_generations=200)
    # print('avg_hv', avg_hv, 'std_hv', std_hv, 'avg_igd', avg_igd, 'std_igd', std_igd)
    with open("metric_log/metric_pr.txt", "a") as file:
        file.write(str(i)+save_path)
        file.write(f"avg_hv: {avg_hv}")
        file.write(f"std_hv: {std_hv}")
        file.write(f"avg_igd: {avg_igd}")
        file.write(f"std_igd: {std_igd}\n")
    # mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
    # print(mean_reward)

# model.save("ppo_recurrent")


