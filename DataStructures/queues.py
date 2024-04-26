from collections import deque

my_queue = deque()

my_queue.append(0)
my_queue.append(3)
my_queue.append(6)

item = my_queue.popleft()

print(item)