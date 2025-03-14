import gym
from gym import spaces
import numpy as np
import torch
from evox.problems.numerical import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from maf import MaF1, MaF2, MaF3, MaF4, MaF5, MaF6, MaF7
from moead_evox import MOEAD_evox  # 修改：使用MOEAD替代RVEA
from evox.algorithms import MOEAD
from evox.workflows import StdWorkflow, EvalMonitor
from evox.metrics import hv, igd
from stable_baselines3 import PPO
from test_evox import test_evox
import random
from hv_norm import hv_normalized


def reg_evox_moead():
    # 确保环境已注册
    try:
        gym.make('env_moead-v1')  # 修改：环境ID改为env_moead-v1
    except:
        from gym.envs.registration import register
        register(
            id='env_moead-v1',  # 修改：环境ID改为env_moead-v1
            entry_point='env_moead:EvoXMOEADEnv',  # 修改：入口点改为env_moead:EvoXMOEADEnv
        )


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

class EvoXMOEADEnv(gym.Env):  # 修改：类名改为EvoXMOEADEnv
    def __init__(self, problem, dim, n_generations=100, pop_size=91, reward_mode='hv'):
        super().__init__()
        # self.problems = ['dtlz1', 'dtlz2', 'dtlz5', 'dtlz7']
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
        self.observation_space = spaces.Box(low=0, high=1, shape=(pop_size, dim), dtype=np.float32)
        
        self.problem_class_map = {
            'dtlz1': DTLZ1,
            'dtlz2': DTLZ2,
            'dtlz3': DTLZ3,
            'dtlz4': DTLZ4,
            'dtlz5': DTLZ5,
            'dtlz6': DTLZ6,
            'dtlz7': DTLZ7,
            # 'maf1': MaF1,
            # 'maf2': MaF2,
            # 'maf3': MaF3,
            # 'maf4': MaF4,
            # 'maf5': MaF5,
            # 'maf6': MaF6,
            # 'maf7': MaF7
        }

    def _setup_problem(self):
        problem_name = self.problem
        # 处理 'all' 情况
        if problem_name == 'all':
            available_problems = list(self.problem_class_map.keys())
            selected_problem = random.choice(available_problems)
            problem_class = self.problem_class_map[selected_problem]
            # 记录当前实际使用的问题（可选）
            self._current_problem = selected_problem
        else:
            problem_class = self.problem_class_map.get(problem_name)
            if problem_class is None:
                raise ValueError(f"Unsupported problem: {problem_name}")
            self._current_problem = problem_name  # 当前实际使用的问题类型
        k=5
        d = self.dim + k-1  # DTLZ问题参数（k=5）
        self.problem_instance = problem_class(d=d, m=self.dim)        

    def _setup_algorithm(self):
        # 修改：使用MOEAD替代RVEA
        k = 5
        d = self.dim + k - 1
        self.algorithm = MOEAD_evox(
            pop_size=self.pop_size,
            n_objs=self.dim,
            lb=torch.tensor([0]*d),
            ub=torch.tensor([1]*d),
        )
        self.algorithm_baseline = MOEAD(
            pop_size=self.pop_size,
            n_objs=self.dim,
            lb=torch.tensor([0]*d),
            ub=torch.tensor([1]*d),
        )

    def _setup_workflow(self):
        self.monitor = EvalMonitor()
        self.workflow = StdWorkflow(
            algorithm=self.algorithm,
            problem=self.problem_instance,
            monitor=self.monitor
        )
        self.workflow_baseline = StdWorkflow(
            algorithm=self.algorithm_baseline,
            problem=self.problem_instance,
            monitor=self.monitor
        )
        self.compiled_step = torch.compile(self.workflow.step)
        self.compiled_step_baseline = torch.compile(self.workflow_baseline.step)

    def _unify_initial_population(self):
        """
        统一主工作流和基准工作流的初始种群及其目标值，
        确保两个算法从相同的起点开始，便于公平比较。
        """
        # 获取主工作流的初始种群和适应度值
        initial_pop = self.workflow.algorithm.pop.clone()
        initial_fit = self.workflow.algorithm.fit.clone()
        
        # 将相同的初始种群和适应度值设置给基准工作流
        self.workflow_baseline.algorithm.pop = initial_pop
        self.workflow_baseline.algorithm.fit = initial_fit

    def reset(self):
        # 初始化问题和算法
        self._setup_algorithm()
        self._setup_problem()
        self._setup_workflow()
        self.workflow.init_step()
        self.workflow_baseline.init_step()
        self._unify_initial_population()
        self.current_gen = 0
        self.indicator_last = -1
        self.reward = 0
        self.rewards = []
        self.front = None
        # 初始化指标
        self.ref_point = torch.tensor([1+0.00001]*self.dim)
        
        # 获取真实Pareto前沿
        self.truepf = self.problem_instance.pf()
        
        return self.get_observation()

    def step(self, action):
        # 将动作传递给算法
        # 注意：MOEAD可能需要添加load_action方法，类似于RVEA
        if hasattr(self.workflow.algorithm, 'load_action'):
            self.workflow.algorithm.load_action(action)

        # 执行一步优化
        self._unify_initial_population()
        self.compiled_step()
        self.compiled_step_baseline()
        self.current_gen += 1

        # 获取当前适应度
        current_f = self.workflow.algorithm.fit
        current_f_baseline = self.workflow_baseline.algorithm.fit
        
        # 计算奖励
        reward = self._compute_reward(current_f, current_f_baseline)

        # 检查是否结束
        done = self.current_gen >= self.n_generations

        # 获取归一化观察
        state2 = self.get_observation()

        self.rewards.append(reward)
        return state2, reward, done, {}

    def _compute_reward(self, current_f, current_f_baseline):
        # 过滤掉NaN值
        # current_f = self._filter_nan_rows(current_f)
        
        if self.reward_mode == 'hv':
            current_hv = hv(current_f, self.ref_point)
            reward = current_hv - self.indicator_last
            self.indicator_last = current_hv
        else:
            # current_hv = hv(current_f, self.ref_point)
            optimum = self.truepf.max(dim=0).values
            
            current_hv = hv_normalized(current_f, optimum)
            current_igd = igd(current_f, self.truepf)
            indicator = current_hv - self.w1 * current_igd
            
            current_hv_baseline = hv_normalized(current_f_baseline, optimum)
            current_igd_baseline = igd(current_f_baseline, self.truepf)
            indicator_baseline = current_hv_baseline - self.w1 * current_igd_baseline
            
            reward = (indicator - indicator_baseline)
            # print('indicator', indicator, 'indicator_baseline', indicator_baseline, 'reward', reward)
            # reward = (indicator - self.indicator_last) / abs(self.indicator_last) if self.indicator_last != 0 else 0
            self.indicator_last = indicator
        if self.reward_mode == 'igdhv':
            pass # no revise reward
        elif self.reward_mode == 'log_smooth':
            sign = np.sign(reward)
            reward = sign * np.log(1 + np.abs(reward))
        else: 
            gen_ratio = self.current_gen / self.n_generations if self.n_generations > 0 else 0.0
            weight = 1.0  # 默认无权重
            if self.reward_mode == 'power':  # 幂函数
                p = self.weight_params.get('p', 2.0)  # 默认指数p=2
                weight = gen_ratio ** p

            elif self.reward_mode == 'log':  # 对数平滑
                # 权重：log(g+1)/log(max_gen+1)，归一化到[0,1]
                pass
                # weight = np.log(self.current_gen + 1) / np.log(self.n_generations + 1)

            elif self.reward_mode == 'sigmoid':  # Sigmoid函数
                k = 5.0  # 曲线陡峭度
                x0 = 0.5 * self.n_generations
                normalized_g = (self.current_gen - x0) / (self.n_generations / 2)  # 归一化到[-1,1]
                weight = 1.0 / (1.0 + np.exp(-k * normalized_g))

            elif self.reward_mode == 'cosine':  # 余弦退火
                weight = 0.5 * (1 - np.cos(np.pi * self.current_gen / self.n_generations))
            else:
                raise ValueError("Unsupported reward mode")
            reward = reward * weight
            
        if self.indicator_last == -1:
            return 0.0
        # 限制奖励范围
        # reward = np.clip(reward, -1, 1)
        return reward

    '''def _filter_nan_rows(self, pop_f: torch.Tensor) -> torch.Tensor:
        """
        过滤掉二维张量中全为 NaN 的行
        Args:
            pop_f: 二维张量
        Returns:
            过滤后的张量（保留非全 NaN 行）
        """
        # 创建掩码：标记非全 NaN 的行
        mask = ~torch.all(torch.isnan(pop_f), dim=1)
        
        # 使用掩码过滤行
        filtered_pop_f = pop_f[mask]
        
        return filtered_pop_f'''

    def get_observation(self):
        front = self.workflow.algorithm.fit
        # 过滤NaN值
        # front = self._filter_nan_rows(front)
        obj_ub = torch.tensor(self.problem_instance.obj_ub)
        self.state = front/obj_ub
        return self.state


# 示例使用方式（需配合SB3使用）
if __name__ == '__main__':
    reg_evox_moead()  # 修改：使用新的注册函数
    import warnings
    import torch
    from datetime import datetime
    torch._dynamo.config.cache_size_limit = 512  # 增大缓存限制
    torch._dynamo.config.suppress_errors = True   # 静默处理错误
    warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor*")
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    date = datetime.now().strftime('%m%d')
    
    # reward_mode = 'log_smooth'
    reward_mode = 'igdhv'

    pop_size = 85
    problem = 'all'
    dim = 5
    n_generations = 10000
    n_steps = 10
    total_timesteps = 4e4

    test_step = 2e3
    N_test = int(total_timesteps // test_step)

    save_path = "moead_" + reward_mode[0:4]+"_"+problem+"_d"+str(dim)+"_g"+to_sci_string(n_generations)+"_ns"+str(n_steps)+'_'+to_sci_string(int(total_timesteps))+'_'+date

    # 创建环境
    env = gym.make('env_moead-v1', problem=problem, dim=dim, n_generations=n_generations, pop_size=pop_size, reward_mode=reward_mode)
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./mlp_tensorboard/", n_steps=n_steps, policy_kwargs=policy_kwargs) 
    
    avg_hvs =[]
    avg_igds = []
    hv_tolence = 0.003
    igd_tolence = 0.003
    for i in range(N_test):
        model.learn(total_timesteps=test_step, tb_log_name="ppo_moead_"+ str(i+1)+save_path, log_interval=1, reset_num_timesteps=False)
        model.save("models/"+"ppo_moead_"+ str(i)+save_path)
        vec_env = model.get_env()
        avg_hv, std_hv, avg_igd, std_igd = test_evox(vec_env, problem, dim, model, n_generations=n_generations, n_runs=5, reward_mode=reward_mode)
        print('avg_hv', avg_hv, 'std_hv', std_hv, 'avg_igd', avg_igd, 'std_igd', std_igd)
        with open("metric_log/metric_ppo_moead.txt", "a") as file:
            file.write(str(i)+save_path)
            file.write(f"avg_hv: {avg_hv}")
            file.write(f"std_hv: {std_hv}")
            file.write(f"avg_igd: {avg_igd}")
            file.write(f"std_igd: {std_igd}\n")