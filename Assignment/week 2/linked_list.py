from typing import Optional

class Node:
    """
    A class to represent a single node in a singly linked list.

    Attributes:
        data (int): The data stored in the node.
        next (Optional[Node]): Reference to the next node in the list.
    """
    def __init__(self, data: int):
        self.data: int = data
        self.next: Optional['Node'] = None


class LinkedList:
    """
    A class to represent a singly linked list.

    Methods:
        add_node(data): Adds a node with the given data to the end of the list.
        print_list(): Prints the elements in the list.
        del_nth_node(n): Deletes the nth node in the list (1-based index).
    """
    def __init__(self):
        self.head: Optional[Node] = None

    def add_node(self, data: int) -> None:
        """Adds a node to the end of the linked list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return

        current: Node = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self) -> None:
        """Prints all nodes in the linked list."""
        if not self.head:
            print("List is empty.")
            return

        current: Optional[Node] = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def del_nth_node(self, n: int) -> None:
        """Deletes the nth node (1-based index) from the list."""
        if not self.head:
            raise Exception("Cannot delete from an empty list.")

        if n <= 0:
            raise IndexError("Index must be a positive integer.")

        if n == 1:
            print(f"Deleting node at position {n} with value {self.head.data}")
            self.head = self.head.next
            return

        current: Optional[Node] = self.head
        for i in range(n - 2):
            if current is None or current.next is None:
                raise IndexError(f"Index {n} out of range.")
            current = current.next

        if current.next is None:
            raise IndexError(f"Index {n} out of range.")

        print(f"\n Deleting node at position {n} with value {current.next.data}")
        current.next = current.next.next



if __name__ == "__main__":
    ll = LinkedList()
    
    # Adding nodes
    ll.add_node(10)
    ll.add_node(20)
    ll.add_node(30)
    ll.add_node(40)
    
    print("Initial Linked List: ")
    ll.print_list()
    
    # Deleting 2nd node
    ll.del_nth_node(2)
    print("\nAfter deleting 2nd node:")
    ll.print_list()
    
    # Trying to delete node at an invalid position
    try:
        ll.del_nth_node(10)
    except Exception as e:
        print("\nError:", e)
