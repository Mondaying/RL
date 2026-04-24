
import random

class GridWorldEnv:
    """
    一个简单的网格世界环境。
    状态空间: 3x3的网格, 编号0到8。
    0 1 2
    3 4 5
    6 7 8
    动作空间: 0:上, 1:下, 2:左, 3:右
    终止状态: 0 (奖励+10), 8 (奖励-10)
    """
    def __init__(self):
        self.nS = 9  # 状态总数
        self.nA = 4  # 动作总数
        self.terminal_states = [0, 8]
        self.start_states = [s for s in range(self.nS) if s not in self.terminal_states]
        self.state = -1

    def step(self, s, a):
        """
        在状态s执行动作a。
        返回 (下一个状态, 奖励)。
        这是一个确定性环境。
        """
        if s in self.terminal_states:
            return s, 0

        # 状态到坐标的映射
        row, col = s // 3, s % 3

        # 根据动作更新坐标
        if a == 0: # 上
            row = max(0, row - 1)
        elif a == 1: # 下
            row = min(2, row + 1)
        elif a == 2: # 左
            col = max(0, col - 1)
        elif a == 3: # 右
            col = min(2, col + 1)

        next_s = row * 3 + col
        
        # 计算奖励
        if next_s == 0:
            reward = 10
        elif next_s == 8:
            reward = -10
        else:
            reward = -1 # 每走一步都有一个小惩罚

        return next_s, reward

def mc_epsilon_greedy(env, num_episodes, gamma=0.9, epsilon=0.1, max_steps_per_episode=100):
    """
    蒙特卡洛 Epsilon-Greedy 控制算法。
    """
    # 1. 初始化
    Q = [[0.0] * env.nA for _ in range(env.nS)]
    returns_sum = [[0.0] * env.nA for _ in range(env.nS)]
    returns_count = [[0.0] * env.nA for _ in range(env.nS)]
    
    # 初始化一个确定性策略 (用于贪婪选择)
    policy = [0] * env.nS

    for i in range(num_episodes):
        # 2. 生成一个回合 (Episode)
        episode = []
        s = random.choice(env.start_states)
        
        for _ in range(max_steps_per_episode):
            # Epsilon-Greedy 动作选择
            if random.random() < epsilon:
                a = random.randrange(env.nA) # 探索：随机选择动作
            else:
                a = policy[s] # 利用：选择当前最优动作
            
            next_s, reward = env.step(s, a)
            episode.append((s, a, reward))
            
            if next_s in env.terminal_states:
                break
            s = next_s

        # 3. 从回合中学习：更新Q值和策略
        G = 0
        visited_sa_pairs = set()
        for t in range(len(episode) - 1, -1, -1):
            s, a, reward = episode[t]
            G = gamma * G + reward
            
            if (s, a) not in visited_sa_pairs:
                visited_sa_pairs.add((s, a))
                returns_sum[s][a] += G
                returns_count[s][a] += 1
                Q[s][a] = returns_sum[s][a] / returns_count[s][a]
                
                # 贪婪地更新策略
                best_action = Q[s].index(max(Q[s]))
                policy[s] = best_action
                        
    return policy, Q

def print_policy(policy, env):
    """
    打印策略。
    """
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    policy_str = ""
    for s in range(env.nS):
        if s in env.terminal_states:
            policy_str += " T "
        else:
            policy_str += f" {action_symbols[policy[s]]} "
        
        if (s + 1) % 3 == 0:
            policy_str += "\n"
    print("计算出的最优策略:")
    print(policy_str)

if __name__ == "__main__":
    env = GridWorldEnv()
    
    # 运行 MC Epsilon-Greedy 算法
    # 使用更多的回合数以获得更稳定的结果
    optimal_policy, Q_table = mc_epsilon_greedy(env, num_episodes=50000, epsilon=0.1)
    
    # 打印结果
    print_policy(optimal_policy, env)
