import random

CELL_UNKNOWN = 0
CELL_MISS = -1
CELL_HIT = 1
CELL_SUNK = 2

class BattleshipGame:
    def __init__(self):
        self.size = 10
        self.fleet_sizes = [5, 4, 3, 3, 2]
        self.total_pegs = sum(self.fleet_sizes)
        self.pegs_hit = 0
        
        self.truth_board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.tracking_board = [[CELL_UNKNOWN for _ in range(self.size)] for _ in range(self.size)]
        
        self.ship_pegs = {}
        self.ship_health = {}
        self.move_history = []
        self.shot_number = 0

    def can_place(self, x, y, length, is_horizontal):
        for i in range(length):
            cx = x + i if is_horizontal else x
            cy = y if is_horizontal else y + i
            
            if cx < 0 or cx >= self.size or cy < 0 or cy >= self.size:
                return False
            if self.truth_board[cx][cy] != 0:
                return False
        return True

    def place_ship_manual(self, x, y, length, is_horizontal, ship_id):
        if not self.can_place(x, y, length, is_horizontal):
            return False
            
        self.ship_pegs[ship_id] = []
        self.ship_health[ship_id] = length
        
        for i in range(length):
            cx = x + i if is_horizontal else x
            cy = y if is_horizontal else y + i
            self.truth_board[cx][cy] = ship_id
            self.ship_pegs[ship_id].append((cx, cy))
            
        return True

    def place_ships_randomly(self):
        for ship_id, length in enumerate(self.fleet_sizes, start=1):
            placed = False
            while not placed:
                is_horizontal = random.choice([True, False])
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)
                placed = self.place_ship_manual(x, y, length, is_horizontal, ship_id)

    def fire_shot(self, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return "INVALID"
            
        if self.tracking_board[x][y] != CELL_UNKNOWN:
            return "INVALID"

        self.shot_number += 1
        truth_cell = self.truth_board[x][y]
        
        is_sunk = False
        result_str = ""

        if truth_cell == 0:
            self.tracking_board[x][y] = CELL_MISS
            result_str = "MISS"
        else:
            ship_id = truth_cell
            self.ship_health[ship_id] -= 1
            self.pegs_hit += 1
            
            if self.ship_health[ship_id] == 0:
                self.tracking_board[x][y] = CELL_SUNK
                is_sunk = True
                result_str = "SUNK"
            else:
                self.tracking_board[x][y] = CELL_HIT
                result_str = "HIT"

        self.move_history.append({
            "shotNumber": self.shot_number,
            "x": x,
            "y": y,
            "result": result_str,
            "isSunk": is_sunk
        })

        return result_str

    def undo_last_shot(self):
        """Reverses the last action in the move history ledger."""
        if not self.move_history:
            return False
            
        last_move = self.move_history.pop()
        x, y = last_move["x"], last_move["y"]
        result = last_move["result"]
        
        self.shot_number -= 1
        self.tracking_board[x][y] = CELL_UNKNOWN
        
        # If it was a hit or sunk, we must give the ship its health back
        if result in ["HIT", "SUNK"]:
            ship_id = self.truth_board[x][y]
            self.ship_health[ship_id] += 1
            self.pegs_hit -= 1
            
        return True

    def is_game_over(self):
        return self.pegs_hit == self.total_pegs