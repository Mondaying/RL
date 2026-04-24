
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

    def reset(self):
        """ 重置环境并返回一个随机的初始状态 """
        return random.choice(self.start_states)

    def step(self, s, a):
        """
        在状态s执行动作a。
        返回 (下一个状态, 奖励, 是否终止)。
        """
        if s in self.terminal_states:
            return s, 0, True

        row, col = s // 3, s % 3

        if a == 0: row = max(0, row - 1)
        elif a == 1: row = min(2, row + 1)
        elif a == 2: col = max(0, col - 1)
        elif a == 3: col = min(2, col + 1)

        next_s = row * 3 + col
        
        if next_s == 0:
            reward = 10
            done = True
        elif next_s == 8:
            reward = -10
            done = True
        else:
            reward = -1
            done = False

        return next_s, reward, done

def q_learning(env, num_episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-Learning 算法。
    """
    # 1. 初始化Q表
    Q = [[0.0] * env.nA for _ in range(env.nS)]

    for i in range(num_episodes):
        s = env.reset()
        done = False

        while not done:
            # 2. Epsilon-Greedy 动作选择
            if random.random() < epsilon:
                a = random.randrange(env.nA) # 探索
            else:
                # 利用：选择当前状态下Q值最大的动作
                # 如果有多个最大值，随机选择一个以避免偏差
                max_q = max(Q[s])
                actions = [i for i, q in enumerate(Q[s]) if q == max_q]
                a = random.choice(actions)
            
            # 3. 执行动作并观察
            next_s, reward, done = env.step(s, a)

            # 4. 更新Q值 (核心)
            # 找到下一个状态的最大Q值
            next_max_q = max(Q[next_s])
            # 计算TD目标
            td_target = reward + gamma * next_max_q
            # 计算TD误差
            td_error = td_target - Q[s][a]
            # 更新Q值
            Q[s][a] += alpha * td_error

            # 5. 进入下一个状态
            s = next_s
            
    # 从Q表生成最终策略
    policy = [Q[s].index(max(Q[s])) for s in range(env.nS)]
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
    
    # 运行 Q-Learning 算法
    # Q-Learning 通常比MC方法需要更多的回合来收敛
    optimal_policy, Q_table = q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    # 打印结果
    print_policy(optimal_policy, env)
