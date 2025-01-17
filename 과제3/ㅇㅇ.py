import random
import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import accumulate
from collections import deque

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
N_EPISODES = 7000
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_LIMIT = 7000
LEARNING_RATE = 0.001

# state, action 정의 파라미터
USE_COUNT_STATE = True
STATE_INTERVAL = 3
USE_JOB_TYPE_ACTION = True
RULE_LIST = ['SPT', 'LPT', 'EDD', 'LDD', 'Queue', 'Trie']

class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon, valid_actions):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.choice(valid_actions)
        else:
            return valid_actions[out[0, valid_actions].argmax().item()]

class QueueScheduling:
    def __init__(self, processing_times, due_dates):
        self.processing_times = processing_times
        self.due_dates = due_dates
        self.task_queue = deque()
        self.schedule = []
        self.total_tardiness = 0
        self.current_time = 0

    def enqueue(self, task, processing_time, due_date):
        self.task_queue.append((task, processing_time, due_date))

    def dequeue(self):
        return self.task_queue.popleft() if self.task_queue else None

    def create_task_queue(self):
        for task, processing_times in self.processing_times.items():
            for i, processing_time in enumerate(processing_times):
                due_date = self.due_dates[task][i]
                self.enqueue(task, processing_time, due_date)

    def get_next_task(self):
        task_info = self.dequeue()
        if task_info:
            task, _, _ = task_info
            return (task, 1)  # Returning job type and job ID as 1 for simplicity
        return None

class TrieNode:
    def __init__(self):
        self.children = {}
        self.prefix_sum = 0
        self.due_date = None
        self.is_end_of_word = False

class OptimizedTrieScheduler:
    def __init__(self, processing_times, due_dates):
        self.processing_times = processing_times
        self.due_dates = due_dates
        self.trie = TrieNode()
        self.schedule = []
        self.total_tardiness = 0
        self.current_time = 0

    def insert(self, task, prefix_sum, due_date):
        current = self.trie
        for char in task:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.prefix_sum = prefix_sum
        current.due_date = due_date
        current.is_end_of_word = True

    def build_trie(self):
        for task, times in self.processing_times.items():
            due_dates = self.due_dates.get(task, [])
            prefix_sums = list(accumulate(times))
            for i, due_date in enumerate(due_dates):
                if i < len(prefix_sums):
                    self.insert(task + str(i), prefix_sums[i], due_date)

    def get_next_task(self):
        tasks = []
        self._collect_tasks(self.trie, "", tasks)
        if tasks:
            tasks.sort(key=lambda x: x[1])  # Sort by prefix sum
            task, _, _ = tasks[0]
            return (task, 1)  # Returning job type and job ID
        return None

    def _collect_tasks(self, node, path, tasks):
        if node.is_end_of_word:
            tasks.append((path, node.prefix_sum, node.due_date))
        for char, child in node.children.items():
            self._collect_tasks(child, path + char, tasks)

class JobScheduler:
    def __init__(self, action_type='job_type', rule_list=None, state_type='count', state_interval=1):
        self.action_type = action_type
        self.rule_list = rule_list if rule_list else RULE_LIST
        self.state_type = state_type
        self.state_interval = state_interval
        # Queue 및 Trie 스케줄링 인스턴스를 초기화
        self.queue_scheduler = QueueScheduling(PROCESSING_TIMES, DUE_DATES)
        self.queue_scheduler.create_task_queue()
        self.trie_scheduler = OptimizedTrieScheduler(PROCESSING_TIMES, DUE_DATES)
        self.trie_scheduler.build_trie()
        self.reset()

    def reset(self):
        self.processing_times = PROCESSING_TIMES
        self.total_jobs = {job_type: len(times) for job_type, times in self.processing_times.items()}
        self.remaining_jobs = self.total_jobs.copy()
        self.time = 0
        self.tardiness = 0
        self.job_sequence = []
        self.due_dates = DUE_DATES
        self.job_list = [(job_type, i + 1, time) for job_type, times in PROCESSING_TIMES.items() for i, time in enumerate(times)]
        self.job_list.sort(key=lambda x: (x[2], x[1]))
        return self.get_state()

    def step(self, action):
        if self.action_type == 'job_type':
            job_type, job_id = self.get_valid_job_type_action(action)
        else:
            job_type, job_id = self.get_valid_rule_action(action)

        if job_type is None or job_id is None:
            return self.get_state(), 0, True

        job_index = next((i for i, job in enumerate(self.job_list) if job[0] == job_type and job[1] == job_id), None)
        if job_index is None:
            return self.get_state(), 0, True

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
            if job and any(j[0] == job[0] and j[1] == job[1] for j in self.job_list):
                return job
            action = (action + 1) % len(self.rule_list)
        return None, None

    def get_state(self):
        if self.state_type == 'count':
            return torch.tensor(list(self.remaining_jobs.values()), dtype=torch.float32)  # 수정된 부분: 텐서로 변환
        elif self.state_type == 'progress':
            total_jobs = sum(self.total_jobs.values())
            completed_jobs = total_jobs - sum(self.remaining_jobs.values())
            current_state = completed_jobs // self.state_interval
            return torch.tensor([current_state], dtype=torch.float32)  # 수정된 부분: 텐서로 변환

    def get_state_size(self):
        if self.state_type == 'count':
            return len(self.total_jobs)
        elif self.state_type == 'progress':
            total_jobs = sum(self.total_jobs.values())
            return 1  # Always return 1 for progress state representation

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
        elif rule == 'Queue':
            return self.queue_scheduler.get_next_task()
        elif rule == 'Trie':
            return self.trie_scheduler.get_next_task()

    def get_valid_actions(self):
        if self.action_type == 'job_type':
            return [i for i, job_type in enumerate(self.processing_times.keys()) if self.remaining_jobs[job_type] > 0]
        else:
            return list(range(len(self.rule_list)))

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_LIMIT)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        return torch.stack(s_lst), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.stack(s_prime_lst), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

def train_dqn(n_episodes=N_EPISODES, state_type='count', state_interval=STATE_INTERVAL, action_type='job_type', rule_list=None):
    env = JobScheduler(action_type=action_type, rule_list=rule_list, state_type=state_type, state_interval=state_interval)
    action_size = len(env.rule_list) if action_type == 'rule' else len(env.processing_times)
    state_size = env.get_state_size()

    q = Qnet(state_size, action_size)
    q_target = Qnet(state_size, action_size)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    optimizer = optim.Adam(q.parameters(), lr=LEARNING_RATE)

    for episode in range(n_episodes):
        epsilon = max(0.01, 0.08 - 0.01*(episode/200))
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = state.unsqueeze(0)  # 이미 텐서로 변환되었으므로 바로 사용
            valid_actions = env.get_valid_actions()
            action = q.sample_action(state_tensor, epsilon, valid_actions)
            next_state, reward, done = env.step(action)
            done_mask = 0.0 if done else 1.0
            memory.put((state, action, reward / 100.0, next_state, done_mask))
            state = next_state
            total_reward += reward

            if memory.size() > 2000:
                s, a, r, s_prime, done_mask = memory.sample(BATCH_SIZE)
                q_out = q(s)
                q_a = q_out.gather(1, a)

                valid_actions = [env.get_valid_actions() for _ in range(BATCH_SIZE)]
                max_q_prime = torch.zeros(BATCH_SIZE, 1)
                q_prime = q_target(s_prime)
                for i, actions in enumerate(valid_actions):
                    if actions:
                        max_q_prime[i] = q_prime[i, actions].max().unsqueeze(0)

                target = r + GAMMA * max_q_prime * done_mask
                loss = F.smooth_l1_loss(q_a, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 500 == 0 and episode != 0:
            q_target.load_state_dict(q.state_dict())
            print(f"Episode: {episode}, Job Sequence: {env.job_sequence}, Total Tardiness: {env.tardiness}, Epsilon: {epsilon:.2f}")

    return env, q

def test_dqn(env, q):
    state = env.reset()
    done = False
    job_sequence = []

    while not done:
        state_tensor = state.unsqueeze(0)  # 이미 텐서로 변환되었으므로 바로 사용
        valid_actions = env.get_valid_actions()
        action = q.sample_action(state_tensor, 0, valid_actions)
        next_state, reward, done = env.step(action)
        job_sequence.append(env.job_sequence[-1] if env.job_sequence else None)
        state = next_state

    print("Final Job Sequence:", job_sequence)
    print("Total Tardiness:", env.tardiness)

if __name__ == "__main__":
    # 'count' 상태 정의로 Job Type 액션으로 학습
    print("Training with job type actions and 'count' state definition:")
    env, q = train_dqn(state_type='count', action_type='job_type' if USE_JOB_TYPE_ACTION else 'rule')
    test_dqn(env, q)

    # 'count' 상태 정의로 Rule 기반 액션으로 학습
    print("\nTraining with rule-based actions and 'count' state definition:")
    env, q = train_dqn(state_type='count', action_type='rule', rule_list=RULE_LIST)
    test_dqn(env, q)

    # 'progress' 상태 정의로 Job Type 액션 학습
    print("\nTraining with job type actions and 'progress' state definition:")
    env, q = train_dqn(state_type='progress', action_type='job_type' if USE_JOB_TYPE_ACTION else 'rule')
    test_dqn(env, q)

    # 'progress' 상태 정의로 Rule 기반 액션 학습
    print("\nTraining with rule-based actions and 'progress' state definition:")
    env, q = train_dqn(state_type='progress', action_type='rule', rule_list=RULE_LIST)
    test_dqn(env, q)
