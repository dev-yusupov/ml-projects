class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def print_linked_list(head: ListNode):
    current_node = head

    while current_node:
        print(current_node.val, end=" -> ")
        current_node = current_node.next
    
    print(None)


node1 = ListNode(1)
node2 = ListNode(3)
node3 = ListNode(4)
node4 = ListNode(5)
node5 = ListNode(6)

node1.next = node2
node2.next = node3
node3.next = node4

print_linked_list(node1)