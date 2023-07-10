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
    queue = deque()
    visited = {}
    path = []

    queue.append(start)
    visited[start] = None

    while queue:
        node = queue.popleft()

        if node == end:
            # Reconstruct the path from end to start
            current = end
            while current is not None:
                path.insert(0, current)
                current = visited[current]
            break

        neighbors = np.nonzero(matrix[node])[0]
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append(neighbor)
                visited[neighbor] = node

    return visited, path


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
    visited = {}
    path = []

    def explore(node):
        if node == end:
            # Reconstruct the path from end to start
            current = end
            while current is not None:
                path.insert(0, current)
                current = visited[current]
            return True

        visited[node] = None

        neighbors = np.nonzero(matrix[node])[0]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited[neighbor] = node
                if explore(neighbor):
                    return True

        return False

    explore(start)

    return visited, path


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
    pq = PriorityQueue()
    visited = {}
    path = []

    pq.put((0, start))
    visited[start] = (0, None)

    while not pq.empty():
        cost, node = pq.get()

        if node == end:
            # Reconstruct the path from end to start
            current = end
            while current is not None:
                path.insert(0, current)
                current = visited[current][1]
            break

        neighbors = np.nonzero(matrix[node])[0]
        for neighbor in neighbors:
            new_cost = cost + matrix[node, neighbor]
            if neighbor not in visited or new_cost < visited[neighbor][0]:
                visited[neighbor] = (new_cost, node)
                pq.put((new_cost, neighbor))

    return visited, path



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
    pq = PriorityQueue()
    visited = {}
    path = []

    def heuristic(node):
        return matrix[node, end]

    pq.put((0, start))
    visited[start] = None

    while not pq.empty():
        _, node = pq.get()

        if node == end:
            # Reconstruct the path from end to start
            current = end
            while current is not None:
                path.insert(0, current)
                current = visited[current]
            break

        neighbors = np.nonzero(matrix[node])[0]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited[neighbor] = node
                priority = heuristic(neighbor)
                pq.put((priority, neighbor))

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

    pq = PriorityQueue()
    visited = {}
    path = []

    def heuristic(node):
        return np.linalg.norm(np.array(pos[end]) - np.array(pos[node]))

    pq.put((0, start))
    visited[start] = (0, None)

    while not pq.empty():
        cost, node = pq.get()

        if node == end:
            # Reconstruct the path from end to start
            current = end
            while current is not None:
                path.insert(0, current)
                current = visited[current][1]
            break

        neighbors = np.nonzero(matrix[node])[0]
        for neighbor in neighbors:
            new_cost = visited[node][0] + matrix[node, neighbor]
            if neighbor not in visited or new_cost < visited[neighbor][0]:
                visited[neighbor] = (new_cost, node)
                priority = new_cost + heuristic(neighbor)
                pq.put((priority, neighbor))

    return visited, path

