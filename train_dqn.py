from env import maoea_env

# from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3.dqn import DQN


date = '0302'
reward_mode = 'igdhv'
problem = 'dtlz2'
dim = 2
n_generations = 2000
n_steps = 30
total_timesteps = 10e7
env = maoea_env(problem, dim,  n_generations=n_generations, repetitions=1, pop_size=91, step_times=10, reward_mode =reward_mode)
env = DummyVecEnv([lambda: env])  # 将环境包装成vectorized环境
save_path = "ddqn_"+ reward_mode[0:4]+"_"+problem+"_d"+str(dim)+"_g"+str(n_generations)+"_ns"+str(n_steps)+'_'+str(int(total_timesteps))+'_'+date
# later try dqn
# policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
# ppo_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps, policy_kwargs=policy_kwargs) 

dqn_model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", train_freq=n_steps, double_q = True) 

dqn_model.learn(total_timesteps=total_timesteps, tb_log_name='dqn_models/'+save_path, log_interval=1, reset_num_timesteps=False)

dqn_model.save(save_path)