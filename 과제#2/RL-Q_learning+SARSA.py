import random
import math

# 입력 데이터
PROCESSING_TIMES = {
    'A': [5, 5],
    'B': [3, 3, 3],
    'C': [7, 7],
    'D': [4],
    'E': [2],
    'F': [6, 6, 6]
}

DUE_DATES = {
    'A': [10, 10],
    'B': [15, 15, 15],
    'C': [25, 25],
    'D': [12],
    'E': [14],
    'F': [21, 21, 21]
}

# 학습 파라미터
N_EPISODES = 50000
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01

# state, action 정의 파라미터
USE_COUNT_STATE = True
STATE_INTERVAL = 3
USE_JOB_TYPE_ACTION = True
RULE_LIST = ['SPT', 'LPT', 'EDD', 'LDD']

# 알고리즘 선택
ALGORITHMS = ['sarsa', 'q_learning']

class JobScheduler:
    def __init__(self, action_type='job_type', rule_list=None, state_type='count', state_interval=1):
        self.action_type = action_type
        self.rule_list = rule_list if rule_list else RULE_LIST
        self.state_type = state_type
        self.state_interval = state_interval
        self.reset()

    def reset(self):
        self.processing_times = PROCESSING_TIMES
        self.total_jobs = {job_type: len(times) for job_type, times in self.processing_times.items()}
        self.remaining_jobs = self.total_jobs.copy()
        self.time = 0
        self.tardiness = 0
        self.job_sequence = []

        self.due_dates = DUE_DATES

        self.job_list = []
        for job_type, times in self.processing_times.items():
            for i, time in enumerate(times, 1):
                self.job_list.append((job_type, i, time))

        self.job_list.sort(key=lambda x: (x[2], x[1]))

        return self.get_state()

    def step(self, action):
        if self.action_type == 'job_type':
            job_type, job_id = self.get_valid_job_type_action(action)
        else:
            job_type, job_id = self.get_valid_rule_action(action)

        if job_type is None or job_id is None:
            return self.get_state(), 0, True

        job_index = next(i for i, job in enumerate(self.job_list) if job[0] == job_type and job[1] == job_id)

        processing_time = self.job_list[job_index][2]
        due_date = self.due_dates[job_type][job_id - 1]

        self.time += processing_time
        self.job_sequence.append(job_type)

        self.remaining_jobs[job_type] -= 1

        tardiness = max(0, self.time - due_date)
        self.tardiness += tardiness

        self.job_list.pop(job_index)

        done = len(self.job_list) == 0
        reward = -tardiness

        return self.get_state(), reward, done

    def get_valid_job_type_action(self, action):
        job_types = list(self.processing_times.keys())
        for _ in range(len(job_types)):
            job_type = job_types[action % len(job_types)]
            if self.remaining_jobs[job_type] > 0:
                job_id = self.total_jobs[job_type] - self.remaining_jobs[job_type] + 1
                return job_type, job_id
            action = (action + 1) % len(job_types)
        return None, None

    def get_valid_rule_action(self, action):
        for _ in range(len(self.rule_list)):
            rule = self.rule_list[action % len(self.rule_list)]
            job = self.select_job_by_rule(rule)
            if job is not None:
                return job
            action = (action + 1) % len(self.rule_list)
        return None, None

    def select_job_by_rule(self, rule):
        available_jobs = [job for job in self.job_list if self.remaining_jobs[job[0]] > 0]
        if not available_jobs:
            return None

        if rule == 'SPT':
            return min(available_jobs, key=lambda j: j[2])[0:2]
        elif rule == 'LPT':
            return max(available_jobs, key=lambda j: j[2])[0:2]
        elif rule == 'EDD':
            return min(available_jobs, key=lambda j: self.due_dates[j[0]][j[1]-1])[0:2]
        elif rule == 'LDD':
            return max(available_jobs, key=lambda j: self.due_dates[j[0]][j[1]-1])[0:2]

    def get_state(self):
        if self.state_type == 'count':
            return tuple(self.remaining_jobs.values())
        elif self.state_type == 'progress':
            total_jobs = sum(self.total_jobs.values())
            completed_jobs = total_jobs - sum(self.remaining_jobs.values())
            current_state = completed_jobs // self.state_interval
            return (current_state,)

    def get_state_size(self):
        if self.state_type == 'count':
            return len(self.total_jobs)
        elif self.state_type == 'progress':
            total_jobs = sum(self.total_jobs.values())
            return math.ceil(total_jobs / self.state_interval)

class RLAgent:
    def __init__(self, state_size, action_size, algorithm='sarsa'):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.algorithm = algorithm

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        if state not in self.q_table:
            self.q_table[state] = [0] * self.action_size

        return self.q_table[state].index(max(self.q_table[state]))

    def update_q_table(self, state, action, reward, next_state, next_action=None, done=False):
        if state not in self.q_table:
            self.q_table[state] = [0] * self.action_size

        if next_state not in self.q_table:
            self.q_table[next_state] = [0] * self.action_size

        current_q = self.q_table[state][action]

        if self.algorithm == 'sarsa':
            next_q = self.q_table[next_state][next_action] if not done else 0
        else:  # Q-Learning
            next_q = max(self.q_table[next_state]) if not done else 0

        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state][action] = new_q

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train(n_episodes=N_EPISODES, state_type='count', state_interval=STATE_INTERVAL, action_type='job_type', rule_list=None, algorithm='sarsa'):
    env = JobScheduler(action_type=action_type, rule_list=rule_list, state_type=state_type, state_interval=state_interval)
    action_size = len(env.rule_list) if action_type == 'rule' else len(env.processing_times)
    state_size = env.get_state_size()
    agent = RLAgent(state_size=state_size, action_size=action_size, algorithm=algorithm)

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        if algorithm == 'sarsa':
            action = agent.get_action(state)

        while not done:
            if algorithm == 'q_learning':
                action = agent.get_action(state)

            next_state, reward, done = env.step(action)
            
            if algorithm == 'sarsa':
                next_action = agent.get_action(next_state)
                agent.update_q_table(state, action, reward, next_state, next_action, done)
                state, action = next_state, next_action
            else:  # Q-Learning
                agent.update_q_table(state, action, reward, next_state, done=done)
                state = next_state

            total_reward += reward

        if episode % (N_EPISODES//10) == 0:
            print(f"Episode: {episode}, Job Sequence: {env.job_sequence}, Total Tardiness: {env.tardiness}, Epsilon: {agent.epsilon:.2f}")

    return env, agent

def test(env, agent):
    state = env.reset()
    done = False
    action_sequence = []
    job_sequence = []

    while not done:
        action = agent.get_action(state)
        action_sequence.append(action)
        next_state, reward, done = env.step(action)
        job_sequence.append(env.job_sequence[-1] if env.job_sequence else None)
        state = next_state

    print("Final Job Sequence:", job_sequence)
    if env.action_type == 'rule':
        rule_sequence = [env.rule_list[action % len(env.rule_list)] for action in action_sequence]
        print("Rule Sequence:", rule_sequence)
    else:
        action_sequence = [job for job in job_sequence if job]
        print("Action Sequence:", action_sequence)
    print("Total Tardiness:", env.tardiness)

if __name__ == "__main__":
    for algorithm in ALGORITHMS:
        print(f"\n--- Training with {algorithm.upper()} ---")

        # 'count' 상태 정의로 Job Type 액션으로 학습
        print(f"\nTraining with job type actions and 'count' state definition ({algorithm}):")
        env, agent = train(state_type='count', action_type='job_type' if USE_JOB_TYPE_ACTION else 'rule', algorithm=algorithm)
        test(env, agent)

        # 'count' 상태 정의로 Rule 기반 액션으로 학습
        print(f"\nTraining with rule-based actions and 'count' state definition ({algorithm}):")
        env, agent = train(state_type='count', action_type='rule', rule_list=RULE_LIST, algorithm=algorithm)
        test(env, agent)

        # 'progress' 상태 정의로 Job Type 액션 학습 (3개씩 묶음)
        print(f"\nTraining with job type actions and 'progress' state definition (interval 3) ({algorithm}):")
        env, agent = train(state_type='progress', state_interval=STATE_INTERVAL, action_type='job_type' if USE_JOB_TYPE_ACTION else 'rule', algorithm=algorithm)
        test(env, agent)

        # 'progress' 상태 정의로 Rule 기반 액션 학습 (3개씩 묶음)
        print(f"\nTraining with rule-based actions and 'progress' state definition (interval 3) ({algorithm}):")
        env, agent = train(state_type='progress', state_interval=STATE_INTERVAL, action_type='rule', rule_list=RULE_LIST, algorithm=algorithm)
        test(env, agent)
