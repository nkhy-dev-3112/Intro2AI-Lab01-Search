import numpy as np
from queue import PriorityQueue
from collections import deque

def BFS(matrix, start, end):
    """
    BFS algorithm:
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO: 
    queue = deque()  # Create a queue using deque (double-ended queue)
    visited = {}  # Create an empty dictionary to store visited nodes
    path = []  # Create an empty list to store the path

    queue.append(start)  # Enqueue the starting node
    visited[start] = None  # Mark the starting node as visited

    while queue:
        node = queue.popleft()  # Dequeue a node from the front of the queue

        if node == end:
            # Reconstruct the path from end to start
            current = end
            while current is not None:
                path.insert(0, current)  # Insert the current node at the beginning of the path list
                current = visited[current]  # Move to the previous node in the path
            break  # Exit the loop if the end node is reached

        neighbors = np.nonzero(matrix[node])[0]  # Find the neighbors of the current node
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append(neighbor)  # Enqueue the unvisited neighbor
                visited[neighbor] = node  # Mark the neighbor as visited and store the current node as its previous node

    return visited, path  # Return the visited dictionary and the path list


def DFS(matrix, start, end):
    """
    DFS algorithm
     Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited 
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """

    # TODO: 
    visited = {}  # Dictionary to keep track of visited nodes
    path = []  # List to store the path from start to end

    # Recursive DFS function
    def dfs(node, matrix, end, visited):
        if node == end: 
            return  # Exit the function if the end node is reached

        for adjacent_node in range(len(matrix[node])):
            if matrix[node][adjacent_node] != 0 and adjacent_node not in visited:
                visited[adjacent_node] = node  # Mark the adjacent node as visited and store the current node as its previous node
                dfs(adjacent_node, matrix, end, visited)  # Recursively call the DFS function on the adjacent node

    # Call the DFS function to start the search
    visited[start] = None  # Mark the starting node as visited
    dfs(start, matrix, end, visited)  # Start the DFS search

    # Build the path from start to end using the visited dictionary
    if end in visited:
        current = end
        while current != start:
            path.insert(0, current)  # Insert the current node at the beginning of the path list
            current = visited[current]  # Move to the previous node in the path
        path.insert(0, start)  # Insert the start node at the beginning of the path list

    return visited, path  # Return the visited dictionary and the path list


def UCS(matrix, start, end):
    """
    Uniform Cost Search algorithm
     Parameters:visited
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO:  
    pq = PriorityQueue()  # Create a priority queue to prioritize nodes based on their costs
    visited = {}  # Dictionary to keep track of visited nodes
    path = []  # List to store the path from start to end

    pq.put((0, start))  # Enqueue the starting node with a cost of 0
    visited[start] = (0, None)  # Mark the starting node as visited with a cost of 0 and no previous node

    while not pq.empty():
        cost, node = pq.get()  # Dequeue the node with the lowest cost from the priority queue

        if node == end:
            # Reconstruct the path from end to start
            current = end
            while current is not None:
                path.insert(0, current)  # Insert the current node at the beginning of the path list
                current = visited[current][1]  # Move to the previous node in the path
            break  # Exit the loop if the end node is reached

        neighbors = np.nonzero(matrix[node])[0]  # Find the neighbors of the current node
        for neighbor in neighbors:
            new_cost = cost + matrix[node, neighbor]  # Calculate the cost to reach the neighbor
            if neighbor not in visited or new_cost < visited[neighbor][0]:
                visited[neighbor] = (new_cost, node)  # Update the cost and previous node for the neighbor
                pq.put((new_cost, neighbor))  # Enqueue the neighbor with the updated cost

    return visited, path  # Return the visited dictionary and the path list


def GBFS(matrix, start, end):
    """
    Greedy Best First Search algorithm 
    heuristic : edge weights
     Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
   
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO: 
    visited = {}  # Dictionary to keep track of visited nodes
    path = []  # List to store the path from start to end

    priority_queue = PriorityQueue()  # Priority queue for selecting nodes based on the heuristic value
    priority_queue.put((0, start))  # Initialize the priority queue with the start node and heuristic value 0
    visited[start] = None  # Mark the start node as visited
    cost = {}  # Dictionary to store the smallest cost of the path of each node with its parent node
    cost[start] = 0  # Cost of the start node is 0

    while not priority_queue.empty():
        _, current_node = priority_queue.get()  # Get the node with the lowest heuristic value
        if current_node == end:
            break  # End node found, exit the loop

        for adjacent_node in range(len(matrix[current_node])):
            if matrix[current_node][adjacent_node] != 0 and adjacent_node not in visited and (
                    adjacent_node not in cost or matrix[current_node][adjacent_node] < cost[adjacent_node]):
                priority_queue.put((matrix[current_node][adjacent_node], adjacent_node))
                cost[adjacent_node] = matrix[current_node][adjacent_node]
                visited[adjacent_node] = current_node

    # Build the path from start to end using the visited dictionary
    if end in visited:
        current = end
        while current != start:
            path.insert(0, current)
            current = visited[current]
        path.insert(0, start)

    return visited, path

def Astar(matrix, start, end, pos):
    """
    A* Search algorithm
    heuristic: eclid distance based positions parameter
     Parameters:
    ---------------------------
    matrix: np array UCS
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    pos: dictionary. keys are nodes, values are positions
        positions of graph nodes
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO: 
    pq = PriorityQueue()  # Create a priority queue to prioritize nodes based on their costs
    visited = {}  # Dictionary to keep track of visited nodes
    path = []  # List to store the path from start to end

    def heuristic(node):
        # Calculate the Euclidean distance between the given node and the end node
        return np.linalg.norm(np.array(pos[end]) - np.array(pos[node]))

    pq.put((0, start))  # Enqueue the starting node with a cost of 0
    visited[start] = (0, None)  # Mark the starting node as visited with a cost of 0 and no previous node

    while not pq.empty():
        cost, node = pq.get()  # Dequeue the node with the lowest cost from the priority queue

        if node == end:
            # Reconstruct the path from end to start
            current = end
            while current is not None:
                path.insert(0, current)  # Insert the current node at the beginning of the path list
                current = visited[current][1]  # Move to the previous node in the path
            break  # Exit the loop if the end node is reached

        neighbors = np.nonzero(matrix[node])[0]  # Find the neighbors of the current node
        for neighbor in neighbors:
            new_cost = visited[node][0] + matrix[node, neighbor]  # Calculate the cost to reach the neighbor
            if neighbor not in visited or new_cost < visited[neighbor][0]:
                visited[neighbor] = (new_cost, node)  # Update the cost and previous node for the neighbor
                priority = new_cost + heuristic(neighbor)  # Calculate the priority using the cost and heuristic
                pq.put((priority, neighbor))  # Enqueue the neighbor with the updated priority

    return visited, path  # Return the visited dictionary and the path list
