## EXTRA CREDIT ATTEMPTED ##

from slidingBricks import *
import time
import heapq

def hashMatrix(matrix, nCols, nRows) -> int:
    mod = 10**9 + 7 
    base = 31           # prime  multiplier to minimize collisions
    
    hashValue = 0

    for row in range(nRows):
        for col in range(nCols):
            value = int(matrix[row][col])  
            hashValue = (hashValue * base + value) % mod
    
    return hashValue

class HashTable():
    def __init__(self):
        # {key: [matrix]}
        self.hashTable = {}

    def addMatrix(self, matrix, nCols, nRows):
        normState = State(nRows, nCols, matrix)
        normState = normalizeState(normState)

        index = hashMatrix(normState.matrix, nCols, nRows)
        
        if index in self.hashTable:
            self.hashTable[index].append(normState.matrix)

        else:
            self.hashTable[index] = [normState.matrix]
    
    def isDuplicate(self, matrix, nCols, nRows):
        normState = State(nRows, nCols, matrix)
        normState = normalizeState(normState)

        index = hashMatrix(normState.matrix, nCols, nRows)
        
        if index not in self.hashTable:
            return False
         
        else:
            sublist = self.hashTable[index]
            for m in sublist:
                if compareMatrices(m, normState.matrix, len(m[0]), nCols, len(m), nRows):
                    return True
            
            return False

class Node():
    def __init__(self, matrix, path: list = [], next=None):
        self.value = matrix
        self.next = next
        self.path = path

class Hnode():
    def __init__(self, matrix, path: list = [], next=None):
        self.value = matrix
        self.next = next
        self.path = path
        self.numRows = len(matrix)
        self.numCols = len(matrix[0])
        self.h = 0
    
    def heuristic(self):
        x_exit = 0
        y_exit = 0
        numExitEntries = 0
        x_master = 0
        y_master = 0
        numMasterEntries = 0

        for i in range(self.numRows):
            for j in range(self.numCols):
                if self.value[i][j] == '2':
                    x_master += j
                    y_master += i
                    numMasterEntries += 1

                if self.value[i][j] == '-1':
                    x_exit += j
                    y_exit += i
                    numExitEntries += 1

        if numExitEntries == 0:
            # no -1 entries left, distance to exit = 0
            self.h = 0

        else:
            self.h = abs(x_master/numMasterEntries - x_exit/numExitEntries) + abs(y_master/numMasterEntries - y_exit/numExitEntries)

class Heap():
    def __init__(self):
        self.nodesH = []
        self.nodes = {}        # {h: Queue(Hnode)} - nodes with the same heuristic are enqueued at the same place in dictionary

    def insert(self, node: Hnode):
        node.heuristic()
        heapq.heappush(self.nodesH, node.h)

        if node.h not in self.nodes:
            self.nodes[node.h] = Queue(node)
        else:
            self.nodes[node.h].enqueue(node)

    def extractMin(self) -> Hnode:
        key = heapq.heappop(self.nodesH)

        # return the node enqueued first with the minimum h
        node = self.nodes[key].dequeue()
        return node

    def isEmpty(self):
        return len(self.nodesH) == 0
    

class Queue():
    def __init__(self, head: Node =None):
        self.head = head
        self.tail = head

        if self.head is not None:
            while (self.tail.next is not None):
                self.tail = self.tail.next

    def enqueue(self, node: Node):
        if self.head is None:
            self.head = node
            self.tail = node

        else:
            self.tail.next = node  

            while node.next is not None:
                node = node.next

            self.tail = node


    def dequeue(self):
        if self.head is None:
            return None 
                
        dequeued = self.head
        self.head = self.head.next

        if self.head is None:
            self.tail = None  

        return dequeued
        
    
    def isEmpty(self):
        return self.head is None

class Stack():
    def __init__(self, top: Node = None):
        self.top = top
    
    def getTop(self):
        return self.top

    def pop(self):
        popped = self.getTop()

        self.top = self.top.next

        return popped
    
    def isEmpty(self):
        return self.top is None
    
    def push(self, node: Node):
        if self.isEmpty():
            self.top = node
        
        else:
            node.next = self.top
            self.top = node


def printPath(path:list):
    for move in path:
        print(f'({move[0]},{move[1]})')
    print()


def BFS(root: State):
    nCols, nRows = root.numCols, root.numRows

    initialMatrix = clone(root.matrix)
    Q = Queue(Node(initialMatrix))
    numNodesExplored = 0

    duplicates = HashTable()

    d_moves = {}
    start_time = time.time()

    while not Q.isEmpty():
        currNode = Q.dequeue()
        currMatrix = currNode.value
        numNodesExplored += 1
        goalReached = isGoal(currMatrix)

        if goalReached:
            # print("Goal state reached")
            printPath(currNode.path)
            State(nRows, nCols, currMatrix).printState()
            break

        currState = State(nRows, nCols, currMatrix)

        d_moves = currState.availableMoves()

        # apply all available moves and enqueue state matrices after advancing
        for key in d_moves.keys():
            if len(d_moves[key]) > 0:
                # print("checking brick #" + key)

                for move in d_moves[key]:
                    newMatrix = clone(currState.matrix)
                    newState = State(nRows, nCols, newMatrix)

                    # print("applying move:", move)
                    newState.applyMove(key, move)

                    # if state is not a dup, clone the matrix and enqueue
                    if not duplicates.isDuplicate(newState.matrix, nCols, nRows):
                        # newState.printState()
                        # print(f"({key}, {move})")
                        
                        duplicates.addMatrix(clone(newState.matrix), nCols, nRows)

                        # record the path to current new state
                        newPath = copy.deepcopy(currNode.path)
                        newPath.append((key, move))

                        newNode = Node(newState.matrix, newPath)
                        Q.enqueue(newNode)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(numNodesExplored)
    print(f"{execution_time:.2f}")
    print(len(currNode.path))
            
def DFS(root: State):
    nCols, nRows = root.numCols, root.numRows
    initialMatrix = clone(root.matrix)
    stack = Stack(Node(initialMatrix))
    numNodesExplored = 0

    duplicates = HashTable()
    duplicates.addMatrix(initialMatrix, nCols, nRows)

    start_time = time.time()

    while not stack.isEmpty():
        currNode = stack.pop()
        currMatrix = currNode.value
        goalReached = isGoal(currMatrix)
        duplicates.addMatrix(currMatrix, nCols, nRows)

        numNodesExplored += 1

        if goalReached:
            printPath(currNode.path)
            State(nRows, nCols, currMatrix).printState()
            break

        # currState = normalizeState(State(nRows, nCols, currMatrix))
        currState = State(nRows, nCols, currMatrix)
        d_moves = currState.availableMoves()

        # for (b, l_moves) in sorted(l_moves, reverse=True):
        #     d_moves[b] = l_moves
        keys = sorted(d_moves.keys(), reverse=True)
        # keys = d_moves.keys()

        for key in keys:
            if len(d_moves[key]) > 0:
                # print("checking brick #" + key)

                for move in d_moves[key]:
                    

                    # print((key, move))
                    newMatrix = clone(currState.matrix)
                    newState = State(nRows, nCols, newMatrix)

                    newState.applyMove(key, move)
                    
                    # newState = normalizeState(newState)

                    # if state is not a dup, clone the matrix and push to stack
                    if not duplicates.isDuplicate(newState.matrix, nCols, nRows):
                        # newState.printState()
                        # print(f"({key}, {move})")

                        # record the path to current new state
                        newPath = copy.deepcopy(currNode.path)
                        newPath.append((key, move))

                        newNode = Node(newState.matrix, newPath)
                        stack.push(newNode) 

    end_time = time.time()
    execution_time = end_time - start_time
    print(numNodesExplored)
    print(f"{execution_time:.2f}")
    print(len(currNode.path))

def IDS(root: State):
    start_time = time.time()
    numNodesExplored = 0

    depth_limit = 0  

    while True:  
        stack = [Node(root.matrix)]  # reset stack for each new depth limit
        duplicates = HashTable()  # reset duplicates tracking
        duplicates.addMatrix(root.matrix, root.numCols, root.numRows)

        while stack:
            currNode = stack.pop()
            currMatrix = currNode.value
            numNodesExplored += 1

            if len(currNode.path) > depth_limit:
                continue  # skip nodes that exceed the current depth limit

            if isGoal(currMatrix):  
                printPath(currNode.path)
                State(root.numRows, root.numCols, currMatrix).printState()
                end_time = time.time()
                print(numNodesExplored)
                print(f"{(end_time - start_time):.2f}")
                print(len(currNode.path))
                return

            currState = State(root.numRows, root.numCols, currMatrix)
            d_moves = currState.availableMoves()

            for key in d_moves.keys():
                for move in d_moves[key]:
                    newMatrix = clone(currState.matrix)
                    newState = State(root.numRows, root.numCols, newMatrix)
                    newState.applyMove(key, move)

                    if not duplicates.isDuplicate(newState.matrix, root.numCols, root.numRows):
                        newPath = copy.deepcopy(currNode.path)
                        newPath.append((key, move))
                        newNode = Node(newState.matrix, newPath)
                        stack.append(newNode)
                        duplicates.addMatrix(newState.matrix, root.numCols, root.numRows)

        # increase depth limit and retry search
        depth_limit += 1


def Astar(root: State):
    initialMatrix = clone(root.matrix)
    numNodesExplored = 0

    start_time = time.time()

    currNode = Hnode(initialMatrix)
    nCols, nRows = currNode.numCols, currNode.numRows

    heapQ = Heap()
    heapQ.insert(currNode)

    duplicates = HashTable()
    duplicates.addMatrix(initialMatrix, nCols, nRows)

    while not heapQ.isEmpty():    
        currNode = heapQ.extractMin()

        currMatrix = currNode.value
        numNodesExplored += 1

        duplicates.addMatrix(currMatrix, nCols, nRows)

        goalReached = isGoal(currMatrix)
        if goalReached:
            printPath(currNode.path)
            State(root.numRows, root.numCols, currMatrix).printState()
            break

        currState = currState = State(nRows, nCols, currMatrix)
        d_moves = currState.availableMoves()

        # apply all available moves and enqueue state matrices after advancing
        for key in d_moves.keys():
            if len(d_moves[key]) > 0:

                for move in d_moves[key]:
                    newMatrix = clone(currState.matrix)
                    newState = State(nRows, nCols, newMatrix)

                    newState.applyMove(key, move)

                    # if state is not a dup, clone the matrix and insert to heap
                    if not duplicates.isDuplicate(newState.matrix, nCols, nRows):
                        duplicates.addMatrix(clone(newState.matrix), nCols, nRows)

                        # record the path to current new state
                        newPath = copy.deepcopy(currNode.path)
                        newPath.append((key, move))

                        newNode = Hnode(newState.matrix, newPath)
                        heapQ.insert(newNode)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(numNodesExplored)
    print(f"{execution_time:.2f}")
    print(len(currNode.path))
            
def processCommand(searchAlg, filename):
    filename = 'SBP-levels/' + filename
        
    nCols, nRows, matrix = readFromFile(filename)
    InitialState = State(nRows, nCols, matrix)

    if searchAlg == 'bfs':
        BFS(InitialState)
    
    if searchAlg == 'dfs':
        DFS(InitialState)
    
    if searchAlg == 'ids':
        IDS(InitialState)

    if searchAlg == 'astar' or searchAlg == 'competition':
        Astar(InitialState)



def main():
    # nCols, nRows, matrix =  readFromFile("SBP-levels/SBP-level0.txt")
    nCols, nRows, matrix =  readFromFile("SBP-levels/SBP-bricks-level1.txt")

    InitialState = State(nRows, nCols, matrix)
    # BFS(InitialState)
    # print()
    # DFS(InitialState)
    # IDS(InitialState)
    # Astar(InitialState)

# main()

                        
