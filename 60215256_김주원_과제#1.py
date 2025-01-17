PROCESSING_TIMES = {
    'A': [6],
    'B': [4],
    'C': [7, 8, 7],
    'D': [3, 3],
    'E': [2],
    'F': [4, 4, 4, 4]
}

DUE_DATES = {
    'A': [10],
    'B': [30],
    'C': [25, 5, 25],
    'D': [15, 17],
    'E': [14],
    'F': [20, 20, 30, 20]
}

class JobScheduler:
    def __init__(self, rule):
        self.rule = rule
        self.processing_times = PROCESSING_TIMES
        self.due_dates = DUE_DATES
        self.time = 0
        self.tardiness = 0
        self.job_sequence = []
        self.job_list = []
        
        for job_type, times in self.processing_times.items():
            for i, time in enumerate(times, 1):
                self.job_list.append((job_type, i, time))

    def schedule_jobs(self):
        while self.job_list:
            job = self.select_job_by_rule()
            if job is None:
                break
            
            job_type, job_id, processing_time = job
            due_date = self.due_dates[job_type][job_id - 1]

            self.time += processing_time
            self.job_sequence.append(job_type)

            tardiness = max(0, self.time - due_date)
            self.tardiness += tardiness

            self.job_list.remove(job)

    def select_job_by_rule(self):
        if not self.job_list:
            return None

        if self.rule == 'SPT':
            return min(self.job_list, key=lambda j: j[2])
        elif self.rule == 'LPT':
            return max(self.job_list, key=lambda j: j[2])
        elif self.rule == 'EDD':
            return min(self.job_list, key=lambda j: self.due_dates[j[0]][j[1]-1])
        elif self.rule == 'LDD':
            return max(self.job_list, key=lambda j: self.due_dates[j[0]][j[1]-1])

def scheduling(rule):
    scheduler = JobScheduler(rule)
    scheduler.schedule_jobs()
    print(f"Rule: {rule}")
    print("Job Sequence:", scheduler.job_sequence)
    print("Total Tardiness:", scheduler.tardiness)
    print()

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

    def print_schedule(self):
        # 최적 스케줄 및 최소 타디니스 출력
        print(f"Rule: Queue")
        print("Job Sequence:", self.optimal_schedule)
        print("Total Tardiness:", self.total_tardiness)

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
    
    def print_schedule(self):
        print("")
        print(f"Rule: Trie")
        print("Job Sequence:", [task for task, _, _, _, _ in self.schedule])
        print("Total Tardiness:", self.total_tardiness)




       



if __name__ == "__main__":

    # SPT rule로 스케줄링
    scheduling('SPT')

    # LPT rule로 스케줄링
    scheduling('LPT')

    # EDD rule로 스케줄링
    scheduling('EDD')

    # LDD rule로 스케줄링
    scheduling('LDD')
    
    # Queue rule로 스케줄링
    Queue_scheduler = Queue(PROCESSING_TIMES, DUE_DATES)
    Queue_scheduler.create_task_queue()
    Queue_scheduler.schedule_tasks()
    Queue_scheduler.print_schedule()

    # Trie rule로 스케줄링
    scheduler = OptimizedTrieScheduler(PROCESSING_TIMES, DUE_DATES)
    scheduler.build_trie()
    scheduler.schedule_tasks()
    scheduler.print_schedule()
