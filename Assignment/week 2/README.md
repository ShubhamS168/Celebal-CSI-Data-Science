# Singly Linked List in Python (OOP Style)

## Overview
This project demonstrates a simple implementation of a singly linked list using object-oriented programming principles in Python.

## Files
- **linked_list.py**: Contains the implementation of the singly linked list data structure.

## Features
The implementation includes:

- A `Node` class to represent individual nodes in the linked list.
- A `LinkedList` class with the following methods:
  - `add_node(data)`: Adds a new node with the specified integer data to the end of the list.
  - `print_list()`: Prints all elements in the linked list in sequence.
  - `del_nth_node(n)`: Deletes the nth node from the list (1-based indexing).
- Exception handling for edge cases:
  - Attempting to delete a node from an empty list.
  - Attempting to delete a node with an invalid index (negative, zero, or out of range).

## Diagrams
### Basic Structure of a Singly Linked List
```
head
 │
 ▼
┌───┬───┐    ┌───┬───┐    ┌───┬───┐    ┌───┬───┐
│ 10│ ●─┼───>│ 20│ ●─┼───>│ 30│ ●─┼───>│ 40│ / │
└───┴───┘    └───┴───┘    └───┴───┘    └───┴───┘
  Node 1       Node 2       Node 3       Node 4
```

### Append Operation (Adding a New Node)
Before adding node with value 40:
```
head
 │
 ▼
┌───┬───┐    ┌───┬───┐    ┌───┬───┐
│ 10│ ●─┼───>│ 20│ ●─┼───>│ 30│ / │
└───┴───┘    └───┴───┘    └───┴───┘
```

After adding node with value 40:
```
head
 │
 ▼
┌───┬───┐    ┌───┬───┐    ┌───┬───┐    ┌───┬───┐
│ 10│ ●─┼───>│ 20│ ●─┼───>│ 30│ ●─┼───>│ 40│ / │
└───┴───┘    └───┴───┘    └───┴───┘    └───┴───┘
```

### Delete Operation (Removing the 2nd Node)
Before deletion:
```
head
 │
 ▼
┌───┬───┐    ┌───┬───┐    ┌───┬───┐    ┌───┬───┐
│ 10│ ●─┼───>│ 20│ ●─┼───>│ 30│ ●─┼───>│ 40│ / │
└───┴───┘    └───┴───┘    └───┴───┘    └───┴───┘
```

After deleting the 2nd node (value 20):
```
head
 │
 ▼
┌───┬───┐    ┌───┬───┐    ┌───┬───┐
│ 10│ ●─┼───>│ 30│ ●─┼───>│ 40│ / │
└───┴───┘    └───┴───┘    └───┴───┘
                  │                    ▲
                  │    ┌───┬───┐       │
                  └────┤ 20│ ●─┼───────┘
                       └───┴───┘
                      (disconnected)
```

## How to Run
To run the program, use Python 3:
```bash
python linked_list.py
```

### Example Output
When you run the program, the output will be similar to:
```
Initial Linked List:
10 -> 20 -> 30 -> 40 -> None
After deleting 2nd node:
Deleting node at position 2 with value 20
10 -> 30 -> 40 -> None
Error: Index 10 out of range
```

## Implementation Details
### Node Class
Each node in the linked list contains:
- **data**: An integer value stored in the node.
- **next**: A reference to the next node in the list (or `None` if it is the last node).
```
┌─────────────┐
│    Node     │
├─────────────┤
│ data: int   │  <- Integer value stored in the node
│ next: Node  │  <- Reference to the next node (or None)
└─────────────┘
```

### LinkedList Class
The linked list maintains a reference to the head node and provides methods for common operations:
- **add_node(data)**: Adds a new node with the given integer data to the end of the list.
- **print_list()**: Prints all elements in the list, ending with `None` to indicate the list's end.
- **del_nth_node(n)**: Deletes the nth node (1-based indexing) and prints the position and value of the deleted node.
```
┌────────────────┐
│   LinkedList   │
├────────────────┤
│ head: Node     │  <- Reference to the first node
├────────────────┤
│ add_node()     │  <- Add a node at the end
│ print_list()   │  <- Display all elements
│ del_nth_node() │  <- Delete the nth node
└────────────────┘
```

The implementation uses type hints from the `typing` module for clarity and includes exception handling:
- Raises an `Exception` when attempting to delete from an empty list.
- Raises an `IndexError` for invalid indices (negative, zero, or out of range).
