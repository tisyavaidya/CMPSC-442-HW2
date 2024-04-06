############################################################
# CMPSC 442: Informed Search
############################################################

student_name = "Tisya Vaidya"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

import random
from queue import PriorityQueue
import queue
import math
import sys




############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    board = [[cols * i + j + 1 for j in range(cols)] for i in range(rows)]
    board[-1][-1] = 0  
    return TilePuzzle(board)

class TilePuzzle(object):
    
    # Required
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.empty_tile = self.find_empty_tile()  

    def find_empty_tile(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 0:
                    return (i, j)
        raise ValueError("Empty tile not found in the board.")
        
    def get_board(self):
        return self.board

    def perform_move(self, direction):
        empty_row, empty_col = self.empty_tile
        if direction == "up" and empty_row > 0:
            self.board[empty_row][empty_col], self.board[empty_row-1][empty_col] = self.board[empty_row-1][empty_col], self.board[empty_row][empty_col]
            self.empty_tile = (empty_row-1, empty_col)
            
        elif direction == "down" and empty_row < self.rows - 1:
            self.board[empty_row][empty_col], self.board[empty_row+1][empty_col] = self.board[empty_row+1][empty_col], self.board[empty_row][empty_col]
            self.empty_tile = (empty_row+1, empty_col)
            
        elif direction == "left" and empty_col > 0:
            self.board[empty_row][empty_col], self.board[empty_row][empty_col-1] = self.board[empty_row][empty_col-1], self.board[empty_row][empty_col]
            self.empty_tile = (empty_row, empty_col-1)
            
        elif direction == "right" and empty_col < self.cols - 1:
            self.board[empty_row][empty_col], self.board[empty_row][empty_col+1] = self.board[empty_row][empty_col+1], self.board[empty_row][empty_col]
            self.empty_tile = (empty_row, empty_col+1)
            

    def scramble(self, num_moves):
        directions = ["up", "down", "left", "right"]
        for _ in range(num_moves):
            direction = random.choice(directions)
            self.perform_move(direction)

    def is_solved(self):
        solved_board = [[self.cols * i + j + 1 for j in range(self.cols)] for i in range(self.rows)]
        solved_board[-1][-1] = 0  
        return self.board == solved_board
    
    def copy(self):
        return TilePuzzle([row[:] for row in self.board])

    def successors(self):
        empty_row, empty_col = self.empty_tile
        for d_row, d_col, direction in [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]:
            new_row, new_col = empty_row + d_row, empty_col + d_col
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                new_puzzle = self.copy()
                new_puzzle.perform_move(direction)
                yield direction, new_puzzle

    # Required
    def find_solutions_iddfs(self):
        def iddfs_helper(node, path, depth_limit):
            if depth_limit == 0 and node.is_solved():
                yield path, True
                return
            if depth_limit > 0:
                for move, successor in node.successors():
                    yield from iddfs_helper(successor, path + [move], depth_limit - 1)

        depth_limit = 0
        solved = False
        while not solved:
            solutions = list(iddfs_helper(self, [], depth_limit))
            if len(solutions) >= 1:
                solved = True
                for solution, _ in solutions:
                    yield solution
            depth_limit += 1

    def manhattan_distance(self):
        distance = 0
        for i in range(self.rows):
            for j in range(self.cols):
                tile = self.board[i][j]
                if tile != 0:
                    target_row = tile // self.cols
                    target_col = tile % self.cols
                    distance += abs(i - target_row) + abs(j - target_col)
        return distance
    
    #Required
    def find_solution_a_star(self):
        priority_queue = PriorityQueue()
        start_distance = self.manhattan_distance()
        priority_queue.put((start_distance, [], self))  
        visited = set()

        while not priority_queue.empty():
            _, path, current_node = priority_queue.get()  
            if current_node.is_solved():
                return path
            visited.add(tuple(map(tuple, current_node.board)))

            for move, successor in current_node.successors():
                if tuple(map(tuple, successor.board)) not in visited:
                    priority = len(path) + 1 + successor.manhattan_distance()
                    priority_queue.put((priority, path + [move], successor))  
                    

############################################################
# Section 2: Grid Navigation
############################################################

def find_path(start, goal, scene):
    if scene[start[0]][start[1]] or scene[goal[0]][goal[1]]:
        return None  

    rows, cols = len(scene), len(scene[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def euclidean_distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def is_valid(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols and not scene[point[0]][point[1]]

    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    open_set = set()  
    open_set.add(start)  
    came_from = {}  
    g_score = {start: 0}  

    while open_set:
        current = min(open_set, key=lambda x: g_score[x] + euclidean_distance(x, goal))  
        open_set.remove(current)
        
        if current == goal:
            return reconstruct_path(came_from, current)

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if not is_valid(neighbor):
                continue

            tentative_g_score = g_score[current] + euclidean_distance(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                open_set.add(neighbor)

    return None

############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

def perform_move(grid, i, steps):
    length = len(grid)
    if 0 <= i + steps < length:
        if grid[i] != -1:
            if grid[i + steps] == -1:
                grid[i + steps], grid[i] = grid[i], -1

def is_solved(grid, n):
    for i in range(len(grid)):
        if i >= len(grid) - n:
            if grid[i] != len(grid) - i - 1:
                return False
        else:
            if grid[i] != -1:
                return False
    return True

def successors(grid, length, n):
    for i in range(length):
        for steps in [-2, -1, 1, 2]:
            if 0 <= i + steps < length and grid[i] != -1 and grid[i + steps] == -1:
                new_grid = grid[:]
                perform_move(new_grid, i, steps)
                yield (i, i + steps), new_grid

def heuristic(grid):
    heuristic_value = 0
    length = len(grid)
    for i in range(length):
        if grid[i] != -1:
            heuristic_value += abs(length - 1 - grid[i] - i)
    return heuristic_value

def find_solution_a_star(length, n):
    initial_grid = list(range(n)) + [-1] * (length - n)
    pq = queue.PriorityQueue()
    pq.put((heuristic(initial_grid), 0, [], initial_grid))
    trace = set()
    while not pq.empty():
        _, moves, path, current_grid = pq.get()
        if tuple(current_grid) in trace:
            continue
        trace.add(tuple(current_grid))
        if is_solved(current_grid, n):
            return path
        for (move, new_grid) in successors(current_grid, length, n):
            if tuple(new_grid) not in trace:
                pq.put((moves + 1 + heuristic(new_grid), moves + 1, path + [move], new_grid))
    return None 

def solve_distinct_disks(length, n):
    return find_solution_a_star(length, n)


############################################################
# Section 4: Dominoes Game
############################################################

class DominoesGame:
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

    def get_board(self):
        return self.board

    @staticmethod
    def create_dominoes_game(rows, cols):
        grid = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                row.append(False)
            grid.append(row)
        return DominoesGame(grid)

    def reset(self):
        self.board = []
        for _ in range(self.rows):
            row = []
            for _ in range(self.cols):
                row.append(False)
            self.board.append(row)

    def is_legal_move(self, row, col, vertical):
        if vertical:
            if row + 1 < self.rows and not self.board[row][col] and not self.board[row + 1][col]:
                return True
        else:
            if col + 1 < self.cols and not self.board[row][col] and not self.board[row][col + 1]:
                return True
        return False
    

    def legal_moves(self, vertical):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_legal_move(r, c, vertical):
                    yield (r, c)

    def perform_move(self, row, col, vertical):
        if vertical:
            self.board[row][col] = True
            self.board[row + 1][col] = True
        else:
            self.board[row][col] = True
            self.board[row][col + 1] = True

    def game_over(self, vertical):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_legal_move(r, c, vertical):
                    return False
        return True
    
    def copy(self):
        new_board = [row[:] for row in self.board]
        return DominoesGame(new_board)

    def successors(self, vertical):
        for move in self.legal_moves(vertical):
            new_game = self.copy()
            new_game.perform_move(move[0], move[1], vertical)
            yield (move, new_game)

    def get_best_move(self, vertical, limit):
        def alpha_beta_search(game, depth, alpha, beta, maximizing_player):
            if depth == 0 or game.game_over(vertical):
                return None, game.evaluate(), 1

            if maximizing_player:
                value = float('-inf')
                best_move = None
                leaf_nodes_visited = 0
                for move, new_game in game.successors(vertical):
                    _, new_value, nodes_visited = alpha_beta_search(new_game, depth - 1, alpha, beta, False)
                    leaf_nodes_visited += nodes_visited
                    if new_value > value:
                        value = new_value
                        best_move = move
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return best_move, value, leaf_nodes_visited
            else:
                value = float('inf')
                leaf_nodes_visited = 0
                for _, new_game in game.successors(not vertical):
                    _, new_value, nodes_visited = alpha_beta_search(new_game, depth - 1, alpha, beta, True)
                    leaf_nodes_visited += nodes_visited
                    value = min(value, new_value)
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return None, value, leaf_nodes_visited
        return alpha_beta_search(self, limit, float('-inf'), float('inf'), True)
    

    def evaluate(self):
        current_player_moves = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_legal_move(r, c, True):
                    current_player_moves += 1
        opponent_moves = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_legal_move(r, c, False):
                    opponent_moves += 1
        return current_player_moves - opponent_moves
