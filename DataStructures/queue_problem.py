from queue import Queue
from typing import Any

def reverse_queue(queue: Queue) -> None:
    if queue.empty() or queue.qsize() == 1:
        return
    
    stack: list[Any] = []
    while not queue.empty():
        stack.append(queue.get())

    while stack:
        queue.put(stack.pop())

# Example usage:
if __name__ == "__main__":
    queue = Queue()
    queue.put(1)
    queue.put(2)
    queue.put(3)
    queue.put(4)
    queue.put(5)

    print("Original:")
    original_elements = []
    while not queue.empty():
        element = queue.get()
        original_elements.append(element)
        print(element, end=' <- ')
    print()

    reverse_queue(queue)

    print("\nReversed:")
    for element in original_elements[::-1]:
        print(element, end=' <- ')
