import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import math
from collections import deque

# 환경 설정
class SchedulingEnvironment:
    def __init__(self, processing_times, due_dates):
        self.processing_times = processing_times
        self.due_dates = due_dates
        self.num_jobs = len(processing_times)
        self.reset()

    def reset(self):
        self.remaining_jobs = list(range(self.num_jobs))
        self.current_time = 0
        self.total_tardiness = 0
        self.job_sequence = []
        return self._get_state()

    def _get_state(self):
        # 상태를 고정된 크기의 벡터로 반환하도록 수정
        state = np.zeros(self.num_jobs + 1, dtype=np.float32)
        state[0] = self.current_time
        state[1:1 + len(self.remaining_jobs)] = self.remaining_jobs
        return state

    def step(self, job_index):
        if job_index >= len(self.remaining_jobs):
            job_index = len(self.remaining_jobs) - 1

        job = self.remaining_jobs.pop(job_index)
        self.job_sequence.append(job)
        processing_time = self.processing_times[job]
        due_date = self.due_dates[job]

        self.current_time += processing_time
        tardiness = max(0, self.current_time - due_date)
        self.total_tardiness += tardiness

        done = len(self.remaining_jobs) == 0
        return self._get_state(), -tardiness, done, {}

# DQN 설정
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# GA로 DQN 가중치 최적화 -> GPT를 사용하여 만든 함수
class GeneticAlgorithm:
    def __init__(self, population_size, input_dim, output_dim, mutation_rate=0.1, generations=10):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = [DQN(input_dim, output_dim) for _ in range(population_size)]
        self.input_dim = input_dim
        self.output_dim = output_dim

    def evaluate_fitness(self, env):
        fitness_scores = []
        for individual in self.population:
            state = env.reset()
            total_reward = 0
            while True:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = individual(state_tensor)
                    action = torch.argmax(q_values).item()
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
                if done:
                    break
            fitness_scores.append(total_reward)
        return fitness_scores

    def select_parents(self, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)
        return [self.population[i] for i in sorted_indices[-2:]]  # 상위 2개 선택

    def crossover(self, parent1, parent2):
        child = DQN(self.input_dim, self.output_dim)
        for child_param, p1_param, p2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            mask = torch.randint(0, 2, p1_param.shape).float()
            child_param.data.copy_(mask * p1_param + (1 - mask) * p2_param)
        return child

    def mutate(self, individual):
        for param in individual.parameters():
            if random.random() < self.mutation_rate:
                param.data += torch.randn_like(param) * 0.1

    def evolve(self, env):
        for generation in range(self.generations):
            fitness_scores = self.evaluate_fitness(env)
            parents = self.select_parents(fitness_scores)
            new_population = [self.crossover(parents[0], parents[1]) for _ in range(self.population_size - 2)]
            new_population += parents
            for individual in new_population:
                self.mutate(individual)
            self.population = new_population
            print(f"Generation {generation + 1}: Best Fitness = {max(fitness_scores)}")

        # 최적의 에이전트를 반환
        best_index = np.argmax(self.evaluate_fitness(env))
        return self.population[best_index]


# GA로 하이퍼파라미터 최적화 -> GPT를 사용하여 만든 함수
class GAForHyperparameters:
    def __init__(self, population_size, generations, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = [
            {
                "learning_rate": 10 ** random.uniform(-4, -2),
                "gamma": random.uniform(0.8, 0.99),
                "epsilon_decay": random.uniform(0.95, 0.999),
            }
            for _ in range(population_size)
        ]

    def evaluate_fitness(self, env, hyperparameters):
        # 주어진 하이퍼파라미터로 DQN 학습 후 성능 평가
        policy_net = DQN(input_dim, output_dim)
        target_net = DQN(input_dim, output_dim)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=hyperparameters["learning_rate"])
        replay_buffer = ReplayBuffer(10000)

        epsilon = EPSILON_START
        total_reward = 0

        for _ in range(100):  # 테스트 학습 에피소드 수
            state = env.reset()
            while True:
                if random.random() < epsilon:
                    action = random.randint(0, len(env.remaining_jobs) - 1)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = policy_net(state_tensor)
                        action = torch.argmax(q_values).item()

                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state

                if done:
                    break

                if len(replay_buffer) >= BATCH_SIZE:
                    batch = replay_buffer.sample(BATCH_SIZE)
                    states, actions, rewards, next_states, dones = batch

                    state_batch = torch.FloatTensor(states)
                    action_batch = torch.LongTensor(actions)
                    reward_batch = torch.FloatTensor(rewards)
                    next_state_batch = torch.FloatTensor(next_states)
                    done_batch = torch.FloatTensor(dones)

                    current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        max_next_q_values = target_net(next_state_batch).max(1)[0]
                        target_q_values = reward_batch + (1 - done_batch) * hyperparameters["gamma"] * max_next_q_values

                    loss = F.mse_loss(current_q_values, target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epsilon = max(EPSILON_END, epsilon * hyperparameters["epsilon_decay"])

        return total_reward

    def select_parents(self, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)
        return [self.population[i] for i in sorted_indices[-2:]]  # 상위 2개 선택

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1:
            child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def mutate(self, individual):
        for key in individual:
            if random.random() < self.mutation_rate:
                if key == "learning_rate":
                    individual[key] = 10 ** random.uniform(-4, -2)
                elif key == "gamma":
                    individual[key] = random.uniform(0.8, 0.99)
                elif key == "epsilon_decay":
                    individual[key] = random.uniform(0.95, 0.999)

    def evolve(self, env):
        for generation in range(self.generations):
            fitness_scores = [self.evaluate_fitness(env, individual) for individual in self.population]
            parents = self.select_parents(fitness_scores)
            new_population = [self.crossover(parents[0], parents[1]) for _ in range(self.population_size - 2)]
            new_population += parents
            for individual in new_population:
                self.mutate(individual)
            self.population = new_population
            print(f"Generation {generation + 1}: Best Fitness = {max(fitness_scores)}")

        # 최적의 하이퍼파라미터 반환 -> GPT를 사용하여 만듬
        best_index = np.argmax([self.evaluate_fitness(env, individual) for individual in self.population])
        return self.population[best_index]
# 하이퍼파라미터 설정
NUM_EPISODES = 1000
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
DISCOUNT_FACTOR = 0.95

# 경험 리플레이 버퍼
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        return (states, np.array(actions), np.array(rewards), next_states, np.array(dones))

    def __len__(self):
        return len(self.buffer)
### 나만의 규칙 1 : 자료구조 Queue를 활하여 FIFO 스케줄링 ###
from collections import deque
class Queue:
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
        if self.task_queue:
            return self.task_queue.popleft()
        return None

    def create_task_queue(self):
        # 작업 큐를 생성하여 (작업 이름, 처리 시간)으로 큐에 삽입
        for task, processing_times in self.processing_times.items(): #GPT 도움
            for i, processing_time in enumerate(processing_times): #GPT 도움
                due_date = self.due_dates[task][i]
                self.enqueue(task, processing_time, due_date)  # enqueue 호출

    def schedule_tasks(self):
        # 큐에서 작업을 꺼내서 스케줄링
        while self.task_queue:
            task_info = self.dequeue()  # dequeue 호출
            if task_info is None:
                break

            task, processing_time, due_date = task_info

            start_time = self.current_time
            end_time = self.current_time + processing_time
            self.current_time += processing_time

            tardiness = max(0, end_time - due_date)
            self.total_tardiness += tardiness

            self.schedule.append((task, processing_time, start_time, end_time, tardiness))
            self.optimal_schedule.append(task)



### 나만의 규칙 2: Trie 구조를 활용해 스케줄링 ###
from itertools import accumulate

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
        for char in task: #GPT 도움
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.prefix_sum = prefix_sum 
        current.due_date = due_date
        current.is_end_of_word = True

    def build_trie(self):
        for task, times in self.processing_times.items():#GPT 도움
            due_dates = self.due_dates.get(task, [])
            prefix_sums = list(accumulate(times))
            for i, due_date in enumerate(due_dates):
                if i < len(prefix_sums):
                    self.insert(task + str(i), prefix_sums[i], due_date)
    
    def schedule_tasks(self):
        tasks = []
        self._collect_tasks(self.trie, "", tasks)
        tasks.sort(key=lambda x: x[1])  # Sort by prefix sum #GPT 도움
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
        for char, child in node.children.items(): #GPT 도움
            self._collect_tasks(child, path + char, tasks)
    

# 문제 데이터 정의
problems = [
    {
        "PROCESSING_TIME": [
            73, 12, 45, 89, 34, 67, 21, 95, 3, 56,
            82, 15, 98, 41, 27, 63, 9, 51, 87, 32,
            76, 18, 94, 37, 69, 5, 83, 24, 91, 48,
            13, 57, 85, 31, 96, 42, 78, 16, 64, 29,
            72, 8, 53, 97, 25, 61, 43, 88, 19, 75,
            35, 92, 14, 58, 81, 47, 26, 93, 39, 70,
            7, 54, 86, 22, 65, 99, 33, 79, 11, 50,
            84, 28, 62, 4, 77, 40, 95, 17, 68, 36,
            90, 23, 55, 80, 44, 2, 74, 38, 100, 20,
            60, 87, 30, 66, 6, 52, 96, 41, 71, 10
        ],
        "DUE_DATE": [
            516, 2841, 1273, 4562, 892, 3214, 967, 4891, 145, 2738,
            3967, 782, 4123, 1856, 947, 3582, 256, 2194, 4731, 1523,
            3841, 692, 4271, 1834, 3156, 427, 2913, 1245, 4682, 2147,
            896, 3471, 1723, 4285, 2614, 937, 3842, 516, 2731, 1428,
            4156, 873, 2941, 4567, 1324, 3756, 947, 4231, 1685, 3247,
            916, 4523, 1847, 3261, 742, 2914, 4371, 1526, 3842, 967,
            2314, 4681, 1247, 3561, 826, 4127, 1934, 3672, 591, 2847,
            4156, 1273, 3614, 892, 2471, 4123, 1756, 3241, 967, 2514,
            4731, 1426, 3267, 841, 2156, 4527, 1823, 3461, 756, 2914,
            4237, 1623, 3841, 967, 2514, 4127, 1834, 3567, 926, 2471
        ]
    },
    {
        "PROCESSING_TIME": [
            65, 28, 93, 17, 82, 44, 31, 99, 12, 76,
            39, 88, 25, 71, 14, 57, 33, 96, 22, 84,
            47, 15, 69, 41, 78, 26, 92, 18, 63, 35,
            87, 23, 55, 29, 74, 11, 66, 38, 95, 21,
            58, 32, 81, 16, 73, 45, 89, 27, 61, 43,
            94, 19, 85, 36, 67, 13, 91, 48, 24, 77,
            42, 86, 53, 97, 34, 68, 15, 83, 49, 20,
            75, 37, 90, 52, 28, 64, 46, 100, 30, 72,
            40, 88, 59, 25, 70, 16, 98, 51, 80, 33,
            62, 10, 79, 45, 94, 22, 87, 54, 26, 56
        ],
        "DUE_DATE": [
            389, 2567, 934, 4245, 1678, 3834, 756, 4123, 1789, 3456,
            1789, 4567, 923, 3345, 1678, 4890, 234, 2567, 4845, 1234,
            3567, 890, 4345, 1456, 3789, 678, 4123, 1345, 3678, 890,
            2234, 4567, 845, 3234, 1567, 4789, 345, 2456, 4789, 1678,
            3923, 845, 4678, 1890, 3234, 567, 4845, 1234, 3567, 890,
            2345, 4567, 789, 3123, 1456, 4789, 234, 2567, 4890, 1234,
            3567, 789, 4234, 1567, 3890, 678, 4123, 1345, 3678, 890,
            2234, 4567, 845, 3234, 1567, 4789, 345, 2456, 4789, 1678,
            3923, 845, 4678, 1890, 3234, 567, 4845, 1234, 3567, 890,
            2345, 4567, 789, 3123, 1456, 4789, 234, 2567, 4890, 1234
        ]
    },
    
    {
        "PROCESSING_TIME": [
            70, 13, 88, 42, 95, 27, 61, 33, 79, 16,
            84, 39, 67, 22, 91, 45, 73, 18, 56, 30,
            82, 25, 93, 48, 64, 11, 77, 35, 89, 20,
            59, 43, 87, 14, 71, 38, 96, 28, 52, 15,
            68, 31, 83, 46, 92, 19, 75, 23, 57, 34,
            97, 41, 66, 12, 85, 49, 74, 26, 60, 17,
            86, 44, 69, 21, 90, 36, 63, 29, 81, 47,
            72, 24, 98, 40, 65, 32, 78, 15, 54, 37,    
            94, 27, 58, 43, 88, 16, 76, 35, 100, 22,
            80, 45, 62, 19, 84, 50, 99, 34, 53, 10
        ],
        "DUE_DATE": [
            485, 2734, 1156, 4823, 967, 3541, 789, 4267, 1523, 3896,
            2145, 4678, 934, 3256, 1789, 4512, 678, 2945, 4156, 1834,
            3567, 912, 4289, 1645, 3912, 756, 4134, 1567, 3845, 923,
            2678, 4156, 1834, 3567, 745, 4123, 1956, 3478, 912, 2645,
            4189, 1567, 3823, 956, 4512, 1734, 3956, 845, 2567, 4123,
            1789, 3456, 967, 4234, 1567, 3845, 712, 2956, 4189, 1534,
            3867, 945, 4256, 1789, 3534, 867, 4123, 1567, 3845, 912,
            2678, 4156, 1834, 3567, 745, 4123, 1956, 3478, 912, 2645,
            4189, 1567, 3823, 956, 4512, 1734, 3956, 845, 2567, 4123,
            1789, 3456, 967, 4234, 1567, 3845, 712, 2956, 4189, 1534
        ]
    }
]

# Q-Learning과 SARSA 관련 학습 함수 정의
def train_q_learning(env, num_episodes, problem_idx):
    q_table = np.zeros((env.num_jobs, env.num_jobs))
    epsilon = EPSILON_START

    for episode in range(num_episodes):
        state = env.reset()
        current_state = 0
        total_tardiness = 0
        job_sequence = []

        while True:
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, len(env.remaining_jobs) - 1)
            else:
                action = np.argmax(q_table[current_state])

            next_state, reward, done, _ = env.step(action)
            next_state_index = action
            job_sequence.append(action)
            total_tardiness += -reward

            if not done:
                best_next_action = np.argmax(q_table[next_state_index])
                q_table[current_state, action] = (1 - LEARNING_RATE) * q_table[current_state, action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * q_table[next_state_index, best_next_action])
            else:
                q_table[current_state, action] = (1 - LEARNING_RATE) * q_table[current_state, action] + LEARNING_RATE * reward
                break

            current_state = next_state_index

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # 매 100 에피소드마다 타디니스와 작업 시퀀스 출력
        if episode % 1000 == 999:
            print(f"Problem {problem_idx + 1}, Q-Learning: Episode {episode + 1}, Total Tardiness: {total_tardiness}, Job Sequence: {job_sequence}")

    return q_table

def train_sarsa(env, num_episodes, problem_idx):
    sarsa_table = np.zeros((env.num_jobs, env.num_jobs))
    epsilon = EPSILON_START

    for episode in range(num_episodes):
        state = env.reset()
        current_state = 0
        total_tardiness = 0
        job_sequence = []

        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, len(env.remaining_jobs) - 1)
        else:
            action = np.argmax(sarsa_table[current_state])

        while True:
            next_state, reward, done, _ = env.step(action)
            next_state_index = action
            job_sequence.append(action)
            total_tardiness += -reward

            if not done:
                if random.uniform(0, 1) < epsilon:
                    next_action = random.randint(0, len(env.remaining_jobs) - 1)
                else:
                    next_action = np.argmax(sarsa_table[next_state_index])

                sarsa_table[current_state, action] = (1 - LEARNING_RATE) * sarsa_table[current_state, action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * sarsa_table[next_state_index, next_action])
                current_state, action = next_state_index, next_action
            else:
                sarsa_table[current_state, action] = (1 - LEARNING_RATE) * sarsa_table[current_state, action] + LEARNING_RATE * reward
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # 매 100 에피소드마다 타디니스와 작업 시퀀스 출력
        if episode % 1000 == 999:
            print(f"Problem {problem_idx + 1}, SARSA: Episode {episode + 1}, Total Tardiness: {total_tardiness}, Job Sequence: {job_sequence}")

    return sarsa_table

for idx, problem in enumerate(problems): #-> GPT를 사용하여 만든 for문
    print(f"\nProblem {idx + 1}")
    processing_time = problem["PROCESSING_TIME"]
    due_date = problem["DUE_DATE"]
    env = SchedulingEnvironment(processing_time, due_date)
    input_dim = len(env._get_state())
    output_dim = len(processing_time)

    # Step 1: GA로 초기 가중치 최적화
    print("Optimizing initial weights with GA...")
    ga = GeneticAlgorithm(population_size=10, input_dim=input_dim, output_dim=output_dim, generations=5)
    optimal_agent = ga.evolve(env)
    print("GA optimized weights applied.")

    # Step 2: GA로 하이퍼파라미터 탐색
    print("Optimizing hyperparameters with GA...")
    ga_hyper = GAForHyperparameters(population_size=10, generations=5)
    best_hyperparameters = ga_hyper.evolve(env)
    print(f"Best Hyperparameters for Problem {idx + 1}: {best_hyperparameters}")

    # Step 3: GA+DQN 학습
    print("\nTraining GA+DQN with optimized weights and hyperparameters...")
    ga_dqn_policy_net = optimal_agent
    ga_dqn_target_net = DQN(input_dim, output_dim)
    ga_dqn_target_net.load_state_dict(ga_dqn_policy_net.state_dict())
    ga_dqn_target_net.eval()

    optimizer_ga_dqn = optim.Adam(ga_dqn_policy_net.parameters(), lr=best_hyperparameters["learning_rate"])
    replay_buffer_ga_dqn = ReplayBuffer(10000)
    epsilon_ga_dqn = EPSILON_START

    for episode in range(NUM_EPISODES): #GPT 활용하여 만든 for문
        state = env.reset()
        total_reward_ga_dqn = 0

        while True:
            if random.random() < epsilon_ga_dqn:
                action = random.randint(0, len(env.remaining_jobs) - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = ga_dqn_policy_net(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total_reward_ga_dqn += reward
            replay_buffer_ga_dqn.push(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

            if len(replay_buffer_ga_dqn) >= BATCH_SIZE:
                batch = replay_buffer_ga_dqn.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = batch

                state_batch = torch.FloatTensor(states)
                action_batch = torch.LongTensor(actions)
                reward_batch = torch.FloatTensor(rewards)
                next_state_batch = torch.FloatTensor(next_states)
                done_batch = torch.FloatTensor(dones)

                current_q_values = ga_dqn_policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = ga_dqn_target_net(next_state_batch).max(1)[0]
                    target_q_values = reward_batch + (1 - done_batch) * best_hyperparameters["gamma"] * max_next_q_values

                loss = F.mse_loss(current_q_values, target_q_values)
                optimizer_ga_dqn.zero_grad()
                loss.backward()
                optimizer_ga_dqn.step()

        epsilon_ga_dqn = max(EPSILON_END, epsilon_ga_dqn * best_hyperparameters["epsilon_decay"])

        # 타겟 네트워크 업데이트
        if episode % 10 == 0:
            ga_dqn_target_net.load_state_dict(ga_dqn_policy_net.state_dict())

        # 매 1000 에피소드마다 결과 출력
        if episode % 1000 == 999:
            print(f"GA+DQN: Episode {episode + 1}, Total Tardiness: {env.total_tardiness}, Job Sequence: {env.job_sequence}")

    # Step 4: DQN Only 학습 (기존 하이퍼파라미터 사용)
    print("\nTraining DQN Only with fixed hyperparameters...")
    dqn_policy_net = DQN(input_dim, output_dim)
    dqn_target_net = DQN(input_dim, output_dim)
    dqn_target_net.load_state_dict(dqn_policy_net.state_dict())
    dqn_target_net.eval()

    optimizer_dqn = optim.Adam(dqn_policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer_dqn = ReplayBuffer(10000)
    epsilon_dqn = EPSILON_START

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward_dqn = 0

        while True:
            if random.random() < epsilon_dqn:
                action = random.randint(0, len(env.remaining_jobs) - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = dqn_policy_net(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total_reward_dqn += reward
            replay_buffer_dqn.push(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

            if len(replay_buffer_dqn) >= BATCH_SIZE:
                batch = replay_buffer_dqn.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = batch

                state_batch = torch.FloatTensor(states)
                action_batch = torch.LongTensor(actions)
                reward_batch = torch.FloatTensor(rewards)
                next_state_batch = torch.FloatTensor(next_states)
                done_batch = torch.FloatTensor(dones)

                current_q_values = dqn_policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = dqn_target_net(next_state_batch).max(1)[0]
                    target_q_values = reward_batch + (1 - done_batch) * GAMMA * max_next_q_values

                loss = F.mse_loss(current_q_values, target_q_values)
                optimizer_dqn.zero_grad()
                loss.backward()
                optimizer_dqn.step()

        epsilon_dqn = max(EPSILON_END, epsilon_dqn * EPSILON_DECAY)

        # 타겟 네트워크 업데이트
        if episode % 10 == 0:
            dqn_target_net.load_state_dict(dqn_policy_net.state_dict())

        # 매 1000 에피소드마다 결과 출력
        if episode % 1000 == 999:
            print(f"DQN Only: Episode {episode + 1}, Total Tardiness: {env.total_tardiness}, Job Sequence: {env.job_sequence}")




    # 기존 스케줄링 기법 및 사용자 정의 규칙을 활용한 전체 코드 -> GPT 활용하여 코딩
    for idx, problem in enumerate(problems):
        print(f"\nProblem {idx + 1}")
        processing_time = problem["PROCESSING_TIME"]
        due_date = problem["DUE_DATE"]
        env = SchedulingEnvironment(processing_time, due_date)

        # Rule-based Scheduling Methods (including Trie and Queue)
        print("\nRule-based Scheduling Methods...")
        rule_based_methods = [
            {"name": "SPT", "rule": lambda: sorted(range(len(processing_time)), key=lambda x: processing_time[x])},
            {"name": "LPT", "rule": lambda: sorted(range(len(processing_time)), key=lambda x: -processing_time[x])},
            {"name": "EDD", "rule": lambda: sorted(range(len(due_date)), key=lambda x: due_date[x])},
            {"name": "LDD", "rule": lambda: sorted(range(len(due_date)), key=lambda x: -due_date[x])},
            {"name": "Queue", "rule": lambda: Queue(
                processing_times={str(i): [processing_time[i]] for i in range(len(processing_time))},
                due_dates={str(i): [due_date[i]] for i in range(len(due_date))}
            )},
            {"name": "Trie", "rule": lambda: OptimizedTrieScheduler(
                processing_times={str(i): [processing_time[i]] for i in range(len(processing_time))},
                due_dates={str(i): [due_date[i]] for i in range(len(due_date))}
            )},
        ]

        for method in rule_based_methods:
            if method["name"] in ["SPT", "LPT", "EDD", "LDD"]:
                # 기존 SPT, LPT, EDD, LDD 스케줄링
                sequence = method["rule"]()
                total_tardiness = 0
                current_time = 0
                for job in sequence:
                    current_time += processing_time[job]
                    tardiness = max(0, current_time - due_date[job])
                    total_tardiness += tardiness
                print(f"{method['name']}: Total Tardiness = {total_tardiness}, Job Sequence = {sequence}")

            elif method["name"] == "Queue":
                # Queue 기반 스케줄링
                queue_scheduler = method["rule"]()
                queue_scheduler.create_task_queue()
                queue_scheduler.schedule_tasks()
                print(f"{method['name']}: Total Tardiness = {queue_scheduler.total_tardiness}, Job Sequence = {queue_scheduler.optimal_schedule}")

            elif method["name"] == "Trie":
                # Trie 기반 스케줄링
                trie_scheduler = method["rule"]()
                trie_scheduler.build_trie()
                trie_scheduler.schedule_tasks()
                print(f"{method['name']}: Total Tardiness = {trie_scheduler.total_tardiness}, Job Sequence = {[task for task, _, _, _, _ in trie_scheduler.schedule]}")
        


         

        

        print("\n--- End of Rule-based Scheduling Methods ---\n")
    for idx, problem in enumerate(problems):
        print(f"\nProblem {idx + 1}")
        processing_time = problem["PROCESSING_TIME"]
        due_date = problem["DUE_DATE"]
        env = SchedulingEnvironment(processing_time, due_date)

        # Q-Learning
        print("Training Q-Learning...")
        q_table = train_q_learning(env, num_episodes=1000, problem_idx=idx)
        print(f"Q-Learning: Problem {idx + 1}")
        

        # SARSA
        print("Training SARSA...")
        sarsa_table = train_sarsa(env, num_episodes=1000, problem_idx=idx)
        print(f"SARSA: Problem {idx + 1}")
        
