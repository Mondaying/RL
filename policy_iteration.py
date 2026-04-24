import math

def policy_evaluation(policy, env, gamma=0.9, theta=1e-8):
    """
    评估一个给定的策略。

    参数:
    policy: 要评估的策略, 一个包含每个状态下动作的列表。
    env: OpenAI Gym 风格的环境。
    gamma: 折扣因子。
    theta: 收敛阈值。

    返回:
    V: 策略对应的值函数, 一个列表。
    """
    V = [0.0] * env.nS
    while True:
        delta = 0
        for s in range(env.nS):
            v_old = V[s]
            a = policy[s]
            # 根据贝尔曼期望方程计算新值
            V[s] = sum([p * (r + gamma * V[s_next]) for p, s_next, r, _ in env.P[s][a]])
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration(env, gamma=0.9):
    """
    策略迭代算法。

    参数:
    env: OpenAI Gym 风格的环境。
    gamma: 折扣因子。

    返回:
    V: 最优值函数, 一个列表。
    policy: 最优策略, 一个列表。
    """
    # 1. 初始化一个随机策略
    policy = [0] * env.nS
    
    while True:
        # 2. 策略评估
        V = policy_evaluation(policy, env, gamma)

        # 3. 策略改进
        policy_stable = True
        for s in range(env.nS):
            old_action = policy[s]
            
            # 找到当前状态下最好的动作
            q_values = [sum([p * (r + gamma * V[s_next]) for p, s_next, r, _ in env.P[s][a]]) for a in range(env.nA)]
            if q_values:
                best_action = q_values.index(max(q_values))
            else:
                best_action = old_action

            # 更新策略
            policy[s] = best_action

            # 检查策略是否改变
            if old_action != best_action:
                policy_stable = False

        # 如果策略稳定，则迭代结束
        if policy_stable:
            break
            
    return V, policy

class GridWorldEnv:
    """
    一个简单的网格世界环境 (与之前相同)
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
                    for a in range(self.nA):
                        P[s][a] = [(1.0, s, 0, True)]
                    continue

                actions_outcomes = {
                    0: (max(0, row - 1), col),
                    1: (row, min(self.cols - 1, col + 1)),
                    2: (min(self.rows - 1, row + 1), col),
                    3: (row, max(0, col - 1)),
                }

                for a, (next_row, next_col) in actions_outcomes.items():
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
    
    print("环境构建完成，开始策略迭代...")
    optimal_V, optimal_policy = policy_iteration(env)

    print("\n最优值函数 (Optimal Value Function):")
    for i in range(env.rows):
        row_str = ""
        for j in range(env.cols):
            row_str += f"{optimal_V[i * env.cols + j]:.2f}  "
        print(row_str)

    print("\n最优策略 (Optimal Policy):")
    action_symbols = ['↑', '→', '↓', '←']
    policy_symbols = [action_symbols[i] for i in optimal_policy]
    
    policy_symbols[env._to_s(*env.goal_state)] = 'G'
    policy_symbols[env._to_s(*env.bad_state)] = 'B'
    policy_symbols[env._to_s(*env.obstacle)] = 'X'

    for i in range(env.rows):
        row_str = ""
        for j in range(env.cols):
            row_str += f"{policy_symbols[i * env.cols + j]}  "
        print(row_str)
