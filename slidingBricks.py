import copy
import sys

import random

class State:
    def __init__(self, numRows, numCols, matrix):
        self.numRows = numRows;
        self.numCols = numCols;
        self.matrix = matrix;

    def printState(self):
        print(f'{self.numCols}, {self.numRows},')
        for row in self.matrix:
            line = ''
            for entry in row:
                line += ' ' * (2 - len(str(entry))) + str(entry) + ','
            
            print(line) 
        print()

    def availableMoves(self):
        # dictionary to keep all pair of {brickNumber: brickMoves}
        visitedBricks = {} 
        state = self.matrix

        moves = []

        # find all possible moves of all brick using pieceMoves
        for i in range(self.numRows):
            for j in range(self.numCols):
                if state[i][j] != '1' and state[i][j] != '-1' and state[i][j] != '0':
                    if state[i][j] not in visitedBricks:
                        brickMoves = pieceMoves(self, state[i][j])
                        moves.append((state[i][j], brickMoves))
                        visitedBricks[state[i][j]] = brickMoves

        # print(visitedBricks)
        return visitedBricks
    
    def applyMove(self, piece, direction):
        '''
        Modify the state matrix after move applied
        '''
        self.matrix = copy.deepcopy(applyMoveNew(self, piece, direction))

    def randomWalk(self, numMoves):
        for i in range(numMoves):
            d_possibleMoves = self.availableMoves()
            keys = list(d_possibleMoves.keys())
            piece = random.choice(keys)

            while (len(d_possibleMoves[piece]) == 0):
                keys.remove(piece)
                piece = random.choice(keys)

            direction = random.choice(d_possibleMoves[piece])
            print(f'({piece}, {direction})')

            self.applyMove(piece, direction)
            self.printState()

def readFromFile(fname):
    '''
    Read the file and return the number of columns, number of rows and a matrix as state representation
    '''
    f = open(fname, "r")
    lines = f.readlines()

    matrix = []

    rowCol = lines[0].split(',')
    numCols = int(rowCol[0])
    numRows = int(rowCol[1])
    # print("Number of rows:", numRows)
    # print("Number of cols:", numCols)

    for line in lines[1:]:
        row = line.split(',')
        # print(row[:numCols:])
        matrix.append(row[:numCols:])

    f.close()

    return numCols, numRows, matrix

def clone(matrix):
    cloneMatrix = copy.deepcopy(matrix)
    return cloneMatrix

def isGoal(matrix):
    for row in matrix:
        for entry in row:
            if entry == '-1':
                return False
    
    return True

def pieceMoves(state:State, piece):

    # i: row number
    # j: col number

    # print("Checking brick number", piece)

    # set flags to avoid removing moves more than once
    up = 1
    down = 1
    left = 1
    right = 1

    moves = ['left', 'right', 'up',  'down']
    # moves = ['up', 'right', 'down', 'left']
    # moves = ['up', 'right', 'down', 'left']

    masterBrickMove = ['0', '-1']

    matrix = state.matrix

    for i in range(state.numRows):
        for j in range(state.numCols):
            if matrix[i][j] == piece:

                # checking N
                if i > 0:
                    if up == 1:
                        if piece == '2':
                            if matrix[i-1][j] != piece and matrix[i-1][j] not in masterBrickMove:
                                moves.remove('up')
                                up = 0
                        
                        else:
                            if matrix[i-1][j] != piece and matrix[i-1][j] != '0':
                                moves.remove('up')
                                up = 0
                        
                # checking S
                if i < state.numRows - 1:
                    if down == 1:
                        if piece == '2':
                            if matrix[i+1][j] != piece and matrix[i+1][j] not in masterBrickMove:
                                moves.remove('down')
                                down = 0
                        else:
                            if matrix[i+1][j] != piece and matrix[i+1][j] != '0':
                                moves.remove('down')
                                down = 0

                # checking W
                if j > 0:
                    if left == 1:
                        if piece == '2':
                            if matrix[i][j-1] != piece and matrix[i][j-1] not in masterBrickMove:
                                moves.remove('left')
                                left = 0

                        else:
                            if matrix[i][j-1] != piece and matrix[i][j-1] != '0':
                                moves.remove('left')
                                left = 0  

                # checking E
                if j < state.numCols - 1:
                    if right == 1:
                        if piece == '2':
                            if matrix[i][j+1] != piece and matrix[i][j+1] not in masterBrickMove:
                                moves.remove('right')
                                right = 0 
                        
                        else:
                            if matrix[i][j+1] != piece and matrix[i][j+1] != '0':
                                moves.remove('right')
                                right = 0   

    return moves    

def applyMoveNew(state: State, piece, direction):
    '''
    clone the state, apply move and return the new (cloned) state
    '''
    moves = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    i_move, j_move = moves[direction]

    clonedMatrix = clone(state.matrix)
    
    # set order of entries update
    if direction in ['up', 'left']:  
        row_range = range(state.numRows) 
        col_range = range(state.numCols)
    else:  
        row_range = range(state.numRows - 1, -1, -1)  
        col_range = range(state.numCols - 1, -1, -1)

    for i in row_range:
        for j in col_range:
            if state.matrix[i][j] == piece:
                new_i, new_j = i + i_move, j + j_move
                
                if 0 <= new_i < state.numRows and 0 <= new_j < state.numCols:
                    if piece == '2':
                        if clonedMatrix[new_i][new_j] == '0' or clonedMatrix[new_i][new_j] == '-1':
                            clonedMatrix[new_i][new_j] = piece
                            clonedMatrix[i][j] = '0'
                    else: 
                        if clonedMatrix[new_i][new_j] == '0':
                            clonedMatrix[new_i][new_j] = piece
                            clonedMatrix[i][j] = '0'

    return clonedMatrix

def normalizeState(state: State):
    # keep track of bricks by their starting indices
    brick_positions = {}

    for i in range(state.numRows):
        for j in range(state.numCols):
            cell = state.matrix[i][j]

            if cell !='0' and cell != '1' and cell != '-1' and cell != '2':
                index = i * state.numCols + j
                if cell not in brick_positions:
                    brick_positions[cell] = index
                else:
                    brick_positions[cell] = min(brick_positions[cell], index)
    
    # sort bricks by position
    sorted_bricks = sorted(brick_positions.items(), key=lambda x: x[1])
    
    # keep pairs of brick's old-new numbers in a dictionary
    new_number = 3
    d_bricksNum = {}
    for old_number, _ in sorted_bricks:
        d_bricksNum[old_number] = str(new_number)
        new_number += 1
    
    normalized_matrix = []
    for i in range(state.numRows):
        row = []
        for j in range(state.numCols):
            cell = state.matrix[i][j]
            if cell in d_bricksNum:
                row.append(d_bricksNum[cell])
            else:
                row.append(cell)
        normalized_matrix.append(row)
    
    return State(state.numRows, state.numCols, normalized_matrix)


## FUNCTIONS TO HANDLE BASH COMMANDS ##

def handlePrint():
    filename = 'SBP-levels/' + sys.argv[1]
    numCols, numRows, matrix = readFromFile(filename)
    state = State(numRows, numCols, matrix)
    state.printState()

def handleDone():
    filename = 'SBP-levels/' + sys.argv[1]
    numCols, numRows, matrix = readFromFile(filename)
    print(isGoal(matrix))

def handleAvailableMoves():
    filename = 'SBP-levels/' + sys.argv[1]
    numCols, numRows, matrix = readFromFile(filename)
    state = State(numRows, numCols, matrix)
    d_moves = state.availableMoves()

    for key in d_moves.keys():
        if len(d_moves[key]) > 0:
            # print(visitedBricks[key])
            for move in d_moves[key]:
                print(f'({key}, {move})')

def handleApplyMove():
    filename = 'SBP-levels/' + sys.argv[1]
    move = sys.argv[2]  # (piece, direction)

    piece = move[1:len(move)-1].split(',')[0]
    direction = move[1:len(move)-1].split(',')[1]
    # print(f'piece: {piece}, direction: {direction}')


    numCols, numRows, matrix = readFromFile(filename)
    state = State(numRows, numCols, matrix)
    state.applyMove(piece, direction)
    state.printState()

def compareMatrices(matrix1, matrix2, numCols1, numCols2, numRows1, numRows2):
    if numCols1 != numCols2 or numRows1 != numRows2:
        print(False)
        return False
    
    for i in range(numRows1):
        for j in range(numCols1):
            if matrix1[i][j] != matrix2[i][j]:
                # print(False)
                return False
    # print(True)
    return True

def handleCompare():
    filename1 = 'SBP-levels/' + sys.argv[1]
    numCols1, numRows1, matrix1 = readFromFile(filename1)

    filename2 = 'SBP-levels/' + sys.argv[2]
    numCols2, numRows2, matrix2 = readFromFile(filename2)

    # if numCols1 != numCols2 or numRows1 != numRows2:
    #     print(False)
    #     return False
    
    # for i in range(numRows1):
    #     for j in range(numCols1):
    #         if matrix1[i][j] != matrix2[i][j]:
    #             print(False)
    #             return False
    # print(True)
    print(compareMatrices(matrix1, matrix2, numCols1, numCols2, numRows1, numRows2))

def handleNorm():
    filename = 'SBP-levels/' + sys.argv[1]
    numCols, numRows, matrix = readFromFile(filename)
    state = State(numRows, numCols, matrix)
    normalized_state = normalizeState(state)
    normalized_state.printState()

def handleRandom():
    filename = 'SBP-levels/' + sys.argv[1]
    numCols, numRows, matrix = readFromFile(filename)
    state = State(numRows, numCols, matrix)
    state.printState()

    state.randomWalk(int(sys.argv[2]))







    