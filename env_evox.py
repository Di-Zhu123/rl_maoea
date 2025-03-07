import gym
from gym import spaces
import numpy as np
import torch
from evox.problems.numerical import DTLZ1, DTLZ2, DTLZ5, DTLZ7
from rvea_evox import RVEA
from evox.workflows import StdWorkflow, EvalMonitor
from evox.metrics import hv, igd
from stable_baselines3 import PPO
from test_evox import test_evox

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

class EvoXContiEnv(gym.Env):
    def __init__(self, problem, dim, n_generations=100, pop_size=91, reward_mode='hv'):
        super().__init__()
        self.problems = ['dtlz1', 'dtlz2', 'dtlz5', 'dtlz7']
        self.problem = problem
        self.dim = dim
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.reward_mode = reward_mode
        self.w1 = 10  # 权重参数
        self.state = None

        # 动作空间：假设为连续动作空间
        self.action_space = spaces.Box(low=0, high=1, shape=(pop_size,), dtype=np.float32)
        
        # 观察空间：适应度值矩阵
        self.observation_space = spaces.Box(low=0, high=4, shape=(pop_size, dim), dtype=np.float32)

    def _setup_problem(self):
        problem_name = self.problem
        d = self.dim + 4  # DTLZ问题参数（k=5）
        if problem_name == 'dtlz1':
            self.problem_instance = DTLZ1(d=d, m=self.dim)
        elif problem_name == 'dtlz2':
            self.problem_instance = DTLZ2(d=d, m=self.dim)
        elif problem_name == 'dtlz5':
            self.problem_instance = DTLZ5(d=d, m=self.dim)
        elif problem_name == 'dtlz7':
            self.problem_instance = DTLZ7(d=d, m=self.dim)
        else:
            raise ValueError(f"Unknown problem: {problem_name}")

    def _setup_algorithm(self):
        self.algorithm = RVEA(
            pop_size=self.pop_size,
            n_objs=self.dim,
            lb=torch.tensor([0]*self.dim),
            ub=torch.tensor([1]*self.dim),
            max_gen=self.n_generations,
        )

    def _setup_workflow(self):
        self.monitor = EvalMonitor()
        self.workflow = StdWorkflow(
            algorithm=self.algorithm,
            problem=self.problem_instance,
            monitor=self.monitor
        )
        self.compiled_step = torch.compile(self.workflow.step)

    def reset(self):
        # 初始化问题和算法
        self._setup_algorithm()
        self._setup_problem()
        self._setup_workflow()
        self.workflow.init_step()
        self.current_gen = 0
        self.indicator_last = -1
        self.reward = 0
        self.rewards = []
        self.front = None
        # 初始化指标
        self.ref_point = torch.tensor([1.1]*self.dim)
        if self.problem == 'dtlz7':
            self.ref_point = torch.tensor([15.]*self.dim)
        elif self.problem == 'dtlz1':
            self.ref_point = torch.tensor([400.]*self.dim)
        
        # 获取真实Pareto前沿
        self.truepf = self.problem_instance.pf()
        
        return self.get_observation()

    def step(self, action):
        
        # 将动作传递给算法（需要EvoX RVEA支持外部动作输入）
        # 这里需要根据实际RVEA实现调整
        self.workflow.algorithm.load_action(action)  # 需要RVEA接口支持

        # self.workflow.step()
        self.compiled_step()
        self.current_gen += 1

        # 获取当前适应度
        current_f = self.workflow.algorithm.fit

        # 计算奖励
        reward = self._compute_reward(current_f)

        # 检查是否结束
        done = self.current_gen >= self.n_generations

        # 获取归一化观察
        state2 = self.get_observation()

        self.rewards.append(reward)
        return state2, reward, done, {}

    def _compute_reward(self, current_f):
        if self.indicator_last == -1:
            return 0.0

        if self.reward_mode == 'hv':
            current_hv = hv(current_f, self.ref_point)
            reward = current_hv - self.indicator_last
            self.indicator_last = current_hv
        elif self.reward_mode == 'igdhv':
            current_hv = hv(current_f, self.ref_point)
            current_igd = igd(current_f, self.truepf)
            indicator = current_hv + self.w1 * current_igd
            reward = (indicator - self.indicator_last) / abs(self.indicator_last)
            self.indicator_last = indicator
        else:
            raise ValueError("Unsupported reward mode")

        # 限制奖励范围
        reward = np.clip(reward, -0.1, 2)
        return reward

    def get_observation(self):

        front = self.workflow.algorithm.fit
        self.state = front
        return self.state



# 示例使用方式（需配合SB3使用）
if __name__ == '__main__':
    import warnings
    import torch
    torch._dynamo.config.cache_size_limit = 512  # 增大缓存限制
    torch._dynamo.config.suppress_errors = True   # 静默处理错误
    warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor*")
    date = '0307'
    # reward_mode = 'hv_Percentage'
    # reward_mode = 'ibea'
    reward_mode = 'igdhv'
    pop_size = 85
    problem = 'dtlz2'
    dim = 5
    n_generations = 5000
    n_steps = 10
    total_timesteps = 4e4

    test_step = 2e3
    N_test = int(total_timesteps // test_step)
    # print(to_sci_string(n_generations), to_sci_string(total_timesteps))

    save_path = "ppo_"+ reward_mode[0:4]+"_"+problem+"_d"+str(dim)+"_g"+to_sci_string(n_generations)+"_ns"+str(n_steps)+'_'+to_sci_string(int(total_timesteps))+'_'+date

    env = EvoXContiEnv(problem, dim,  n_generations=n_generations, pop_size=pop_size, reward_mode =reward_mode)
    # policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps, policy_kwargs=policy_kwargs) 
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps) 

    avg_hvs =[]
    avg_igds = []
    hv_tolence = 0.003
    igd_tolence = 0.003
    for i in range(N_test):
    # for i in [0,1]:
        # i=0
        # test_step = 20
        model.learn(total_timesteps=test_step, tb_log_name=str(i)+save_path, log_interval=1, reset_num_timesteps=False)
        model.save("models/"+str(i)+save_path)
        vec_env = model.get_env()
        # vec_env = RecurrentPPO.load(str(i)+save_path)
        print(i)
        avg_hv, std_hv, avg_igd, std_igd = test_evox(vec_env, problem, dim, model, n_generations=n_generations, n_runs=2, reward_mode='igdhv')
        print('avg_hv', avg_hv, 'std_hv', std_hv, 'avg_igd', avg_igd, 'std_igd', std_igd)
        with open("metric_log/metric_ppo.txt", "a") as file:
            file.write(str(i)+save_path)
            file.write(f"avg_hv: {avg_hv}")
            file.write(f"std_hv: {std_hv}")
            file.write(f"avg_igd: {avg_igd}")
            file.write(f"std_igd: {std_igd}\n")
        if len(avg_hvs) > 0:
            if np.abs(avg_hv-avg_hvs[-1])<hv_tolence and np.abs(avg_igd-avg_igds[-1])<igd_tolence:
                print('hv and igd converge')
                break
        avg_hvs.append(avg_hv)
        avg_igds.append(avg_igd)