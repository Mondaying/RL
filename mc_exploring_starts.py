import random
from collections import defaultdict

class MonteCarloGridWorldEnv:
    """
    一个适用于蒙特卡洛方法的网格世界环境。
    这个环境有一个 step 函数用于交互, 而不是暴露整个转移矩阵P。
    """
    def __init__(self):
        self.rows = 3
        self.cols = 4
        self.nS = self.rows * self.cols
        self.nA = 4  # 0: up, 1: right, 2: down, 3: left

        self.obstacle = (1, 1)
        self.goal_state = self._to_s(0, 3)
        self.bad_state = self._to_s(1, 3)
        self.terminal_states = {self.goal_state, self.bad_state}
        
        self.start_states = list(set(range(self.nS)) - self.terminal_states - {self._to_s(*self.obstacle)})

    def _to_s(self, row, col):
        return row * self.cols + col

    def _to_rc(self, s):
        return s // self.cols, s % self.cols

    def step(self, s, a):
        """
        执行一个动作, 返回 (下一个状态, 奖励)。
        """
        if s in self.terminal_states:
            return s, 0

        row, col = self._to_rc(s)

        # 动作: 0: up, 1: right, 2: down, 3: left
        actions_outcomes = {
            0: (max(0, row - 1), col),
            1: (row, min(self.cols - 1, col + 1)),
            2: (min(self.rows - 1, row + 1), col),
            3: (row, max(0, col - 1)),
        }
        
        next_row, next_col = actions_outcomes[a]

        # 如果撞到障碍物，则停在原地
        if (next_row, next_col) == self.obstacle:
            next_row, next_col = row, col

        next_s = self._to_s(next_row, next_col)
        
        reward = 0
        if next_s == self.goal_state:
            reward = 1
        elif next_s == self.bad_state:
            reward = -1
            
        return next_s, reward

def mc_exploring_starts(env, num_episodes, gamma=0.9, max_steps_per_episode=100):
    """
    蒙特卡洛探索性开端算法。
    """
    # 1. 初始化
    Q = [[0.0] * env.nA for _ in range(env.nS)]
    returns_sum = [[0.0] * env.nA for _ in range(env.nS)]
    returns_count = [[0.0] * env.nA for _ in range(env.nS)]
    
    # 初始化一个确定性策略
    policy = [0] * env.nS

    for i in range(num_episodes):
        # 2. 生成一个回合 (Episode)
        episode = []
        
        # 探索性开端：随机选择一个起始状态和动作
        s = random.choice(env.start_states)
        a = random.randrange(env.nA)
        
        for _ in range(max_steps_per_episode):
            next_s, reward = env.step(s, a)
            episode.append((s, a, reward))
            if next_s in env.terminal_states:
                break
            s = next_s
            # 根据当前策略选择下一个动作 (注意: policy[s] 是整数)
            a = policy[s]

        # 3. 从回合中学习：更新Q值和策略
        G = 0
        visited_sa_pairs = set()
        # 从后往前遍历回合
        for t in range(len(episode) - 1, -1, -1):
            s, a, reward = episode[t]
            G = gamma * G + reward
            
            # 采用首次访问(First-Visit)MC方法
            if (s, a) not in visited_sa_pairs:
                visited_sa_pairs.add((s, a))
                returns_sum[s][a] += G
                returns_count[s][a] += 1
                Q[s][a] = returns_sum[s][a] / returns_count[s][a]
                
                # 贪婪地更新策略
                best_action = Q[s].index(max(Q[s]))
                policy[s] = best_action
                
    return policy, Q

if __name__ == '__main__':
    env = MonteCarloGridWorldEnv()
    num_episodes = 20000
    
    print(f"环境构建完成，开始蒙特卡洛探索性开端（{num_episodes} 回合）...")
    optimal_policy, Q_table = mc_exploring_starts(env, num_episodes)

    print("\n最优策略 (Optimal Policy):")
    action_symbols = ['↑', '→', '↓', '←']
    policy_grid = [[' ' for _ in range(env.cols)] for _ in range(env.rows)]

    for s in range(env.nS):
        row, col = env._to_rc(s)
        if s in env.terminal_states:
            if s == env.goal_state:
                policy_grid[row][col] = 'G'
            else:
                policy_grid[row][col] = 'B'
        elif (row, col) == env.obstacle:
            policy_grid[row][col] = 'X'
        else:
            # policy 是一个列表
            policy_grid[row][col] = action_symbols[optimal_policy[s]]

    for row in policy_grid:
        print("  ".join(row))
