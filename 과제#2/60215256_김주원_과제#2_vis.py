from torch.utils.tensorboard import SummaryWriter
import random
import math
from collections import deque
from itertools import accumulate
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TensorBoard 초기화
writer = SummaryWriter(log_dir="runs/JobScheduler")

import matplotlib.pyplot as plt

# 데이터 저장용 리스트 초기화
total_rewards = []
tardiness_values = []
epsilon_values = []
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
N_EPISODES = 75000
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01

# state, action 정의 파라미터
USE_COUNT_STATE = True
STATE_INTERVAL = 3
USE_JOB_TYPE_ACTION = True
RULE_LIST = ['SPT', 'LPT', 'EDD', 'LDD', 'Queue', 'Trie']

# 알고리즘 선택
ALGORITHMS = ['sarsa', 'q_learning']

class JobScheduler:
    def __init__(self, action_type='job_type', rule_list=None, state_type='count', state_interval=1):
        self.action_type = action_type
        self.rule_list = rule_list if rule_list else RULE_LIST
        self.state_type = state_type
        self.state_interval = state_interval
        # Queue 및 Trie 스케줄링 인스턴스를 초기화 #-> GPT 사용
        self.queue_scheduler = QueueScheduling(PROCESSING_TIMES, DUE_DATES) #-> GPT 사용
        self.trie_scheduler = TrieScheduling(PROCESSING_TIMES, DUE_DATES)#-> GPT 사용
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


class QueueScheduling:
    def __init__(self, processing_times, due_dates):
        self.processing_times = processing_times
        self.due_dates = due_dates
        self.task_queue = deque()
        self.schedule = []
        self.optimal_schedule = []
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

    def schedule_tasks(self):
        while self.task_queue:
            task_info = self.dequeue()
            task, processing_time, due_date = task_info
            start_time = self.current_time
            end_time = self.current_time + processing_time
            self.current_time += processing_time
            tardiness = max(0, end_time - due_date)
            self.total_tardiness += tardiness
            self.schedule.append((task, processing_time, start_time, end_time, tardiness))
            self.optimal_schedule.append(task)

    def get_next_task(self):
        task_info = self.dequeue()
        if task_info:
            task, _, _ = task_info # -> GPT 사용
            return (task, 1)  # Returning job type and job ID as 1 for simplicity -> GPT 사용
        return None

class TrieNode:
    def __init__(self):
        self.children = {}
        self.prefix_sum = 0
        self.due_date = None
        self.is_end_of_word = False

class TrieScheduling:
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

    def schedule_tasks(self):
        tasks = []
        self._collect_tasks(self.trie, "", tasks)
        tasks.sort(key=lambda x: x[1])  # Sort by prefix sum
        for task, prefix_sum, due_date in tasks:
            start_time = self.current_time
            end_time = self.current_time + prefix_sum
            self.current_time += prefix_sum
            tardiness = max(0, end_time - due_date)
            self.total_tardiness += tardiness
            self.schedule.append((task, prefix_sum, start_time, end_time, tardiness))

    def _collect_tasks(self, node, path, tasks):
        if node.is_end_of_word:
            tasks.append((path, node.prefix_sum, node.due_date))
        for char, child in node.children.items():
            self._collect_tasks(child, path + char, tasks)

    def get_next_task(self):
        tasks = []
        self._collect_tasks(self.trie, "", tasks)
        if tasks:
            tasks.sort(key=lambda x: x[1])  # Sort by prefix sum
            task, _, _, job_id = tasks[0]
            return (task, job_id)  # Returning job type and job ID
        return None






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

        # TensorBoard에 학습 로그 기록
        writer.add_scalar("Total Tardiness", env.tardiness, episode)
        writer.add_scalar("Total Reward", total_reward, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)
        total_rewards.append(total_reward)
        tardiness_values.append(env.tardiness)
        epsilon_values.append(agent.epsilon)

    

        if episode % (N_EPISODES // 10) == 0:
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

def plot_results(): #-> GPT 사용하여 선언한 함수
    plt.figure(figsize=(12, 6))
    
    # Total Reward Plot
    plt.subplot(3, 1, 1)
    plt.plot(total_rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    # Total Tardiness Plot
    plt.subplot(3, 1, 2)
    plt.plot(tardiness_values, label="Total Tardiness", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Total Tardiness")
    plt.legend()

    # Epsilon Plot
    plt.subplot(3, 1, 3)
    plt.plot(epsilon_values, label="Epsilon", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for algorithm in ALGORITHMS:
        print(f"\n--- Training with {algorithm.upper()} ---")

        # 'count' 상태 정의로 Job Type 액션으로 학습
        print(f"\nTraining with job type actions and 'count' state definition ({algorithm}):")
        total_rewards.clear()
        tardiness_values.clear()
        epsilon_values.clear()
        env, agent = train(state_type='count', action_type='job_type' if USE_JOB_TYPE_ACTION else 'rule', algorithm=algorithm)
        test(env, agent)
        plot_results()

        # 'count' 상태 정의로 Rule 기반 액션으로 학습
        print(f"\nTraining with rule-based actions and 'count' state definition ({algorithm}):")
        total_rewards.clear()
        tardiness_values.clear()
        epsilon_values.clear()
        env, agent = train(state_type='count', action_type='rule', rule_list=RULE_LIST, algorithm=algorithm)
        test(env, agent)
        plot_results()

        # 'progress' 상태 정의로 Job Type 액션 학습 (3개씩 묶음)
        print(f"\nTraining with job type actions and 'progress' state definition (interval 3) ({algorithm}):")
        total_rewards.clear()
        tardiness_values.clear()
        epsilon_values.clear()
        env, agent = train(state_type='progress', state_interval=STATE_INTERVAL, action_type='job_type' if USE_JOB_TYPE_ACTION else 'rule', algorithm=algorithm)
        test(env, agent)
        plot_results()


        # 'progress' 상태 정의로 Rule 기반 액션 학습 (3개씩 묶음)
        print(f"\nTraining with rule-based actions and 'progress' state definition (interval 3) ({algorithm}):")
        total_rewards.clear()
        tardiness_values.clear()
        epsilon_values.clear()
        env, agent = train(state_type='progress', state_interval=STATE_INTERVAL, action_type='rule', rule_list=RULE_LIST, algorithm=algorithm)
        test(env, agent)
        plot_results()

writer.close()