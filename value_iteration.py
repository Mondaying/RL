import math

def value_iteration(env, gamma=0.9, theta=1e-8):
    """
    值迭代算法

    参数:
    env: OpenAI Gym 风格的环境, 需要包含以下属性:
        - nS: 状态空间中的状态数量
        - nA: 动作空间中的动作数量
        - P: 转移概率, 格式为 P[s][a] = [(prob, next_state, reward, done), ...]
    gamma: 折扣因子
    theta: 收敛阈值, 当值函数的最大变化量小于该值时停止迭代

    返回:
    V: 最优值函数, 一个表示每个状态值的列表
    policy: 最优策略, 一个表示每个状态下最优动作的列表
    """
    # 1. 初始化
    V = [0.0] * env.nS

    while True:
        # 2. 迭代
        delta = 0
        # 遍历所有状态
        for s in range(env.nS):
            v_old = V[s]
            # 计算当前状态下所有可能动作的Q值
            q_values = [sum([p * (r + gamma * V[s_next]) for p, s_next, r, _ in env.P[s][a]]) for a in range(env.nA)]
            # 选择最大的Q值作为当前状态的新值
            V[s] = max(q_values) if q_values else 0.0
            # 更新delta
            delta = max(delta, abs(v_old - V[s]))

        # 3. 检查收敛
        if delta < theta:
            break

    # 4. 提取最优策略
    policy = [0] * env.nS
    for s in range(env.nS):
        # 计算当前状态下所有可能动作的Q值
        q_values = [sum([p * (r + gamma * V[s_next]) for p, s_next, r, _ in env.P[s][a]]) for a in range(env.nA)]
        # 选择使Q值最大化的动作
        if q_values:
            policy[s] = q_values.index(max(q_values))

    return V, policy

class GridWorldEnv:
    """
    一个简单的网格世界环境
    网格布局:
    3x4
    S . . G
    . X . B
    . . . .
    S: 起点 (Start)
    G: 目标 (Goal, +1 reward)
    B: 坏状态 (Bad, -1 reward)
    X: 障碍物 (Obstacle)
    """
    def __init__(self):
        self.rows = 3
        self.cols = 4
        self.nS = self.rows * self.cols
        self.nA = 4  # 0: up, 1: right, 2: down, 3: left

        self.obstacle = (1, 1)
        self.goal_state = (0, 3)
        self.bad_state = (1, 3)

        self.P = self._build_transition_prob()

    def _to_s(self, row, col):
        return row * self.cols + col

    def _to_rc(self, s):
        return s // self.cols, s % self.cols

    def _build_transition_prob(self):
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for row in range(self.rows):
            for col in range(self.cols):
                s = self._to_s(row, col)
                if (row, col) == self.obstacle or (row, col) == self.goal_state or (row, col) == self.bad_state:
                    # 在障碍物、目标或坏状态，任何动作都停在原地，没有奖励
                    for a in range(self.nA):
                        P[s][a] = [(1.0, s, 0, True)]
                    continue

                # 动作: 0: up, 1: right, 2: down, 3: left
                actions_outcomes = {
                    0: (max(0, row - 1), col),
                    1: (row, min(self.cols - 1, col + 1)),
                    2: (min(self.rows - 1, row + 1), col),
                    3: (row, max(0, col - 1)),
                }

                for a, (next_row, next_col) in actions_outcomes.items():
                    # 如果撞到障碍物，则停在原地
                    if (next_row, next_col) == self.obstacle:
                        next_row, next_col = row, col

                    next_s = self._to_s(next_row, next_col)
                    reward = 0
                    done = False
                    if (next_row, next_col) == self.goal_state:
                        reward = 1
                        done = True
                    elif (next_row, next_col) == self.bad_state:
                        reward = -1
                        done = True
                    
                    P[s][a] = [(1.0, next_s, reward, done)]
        return P

if __name__ == '__main__':
    env = GridWorldEnv()
    
    print("环境构建完成，开始值迭代...")
    optimal_V, optimal_policy = value_iteration(env)

    print("\n最优值函数 (Optimal Value Function):")
    # 将V重塑为网格形状以便打印
    for i in range(env.rows):
        row_str = ""
        for j in range(env.cols):
            row_str += f"{optimal_V[i * env.cols + j]:.2f}  "
        print(row_str)

    print("\n最优策略 (Optimal Policy):")
    action_symbols = ['↑', '→', '↓', '←']
    policy_symbols = [action_symbols[i] for i in optimal_policy]
    
    # 在特殊状态上标记
    policy_symbols[env._to_s(*env.goal_state)] = 'G'
    policy_symbols[env._to_s(*env.bad_state)] = 'B'
    policy_symbols[env._to_s(*env.obstacle)] = 'X'

    # 将策略重塑为网格形状以便打印
    for i in range(env.rows):
        row_str = ""
        for j in range(env.cols):
            row_str += f"{policy_symbols[i * env.cols + j]}  "
        print(row_str)
