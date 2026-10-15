import pygame
import sys
import csv
import json
import glob
import os
import argparse
from core.game_engine import BattleshipGame, CELL_UNKNOWN, CELL_MISS, CELL_HIT, CELL_SUNK

# ==========================================
# PYGAME SHARED SETUP & CONSTANTS
# ==========================================
pygame.init()
CELL_SIZE = 50
MARGIN = 80 
GRID_SIZE = 10

WINDOW_WIDTH = (MARGIN * 3) + (CELL_SIZE * GRID_SIZE * 2)
WINDOW_HEIGHT = CELL_SIZE * GRID_SIZE + (MARGIN * 3) 

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)
label_font = pygame.font.SysFont(None, 28)

BG_COLOR = (25, 25, 30)
GRID_COLOR = (100, 100, 100)
WATER_COLOR = (10, 60, 120)
SHIP_COLOR = (60, 70, 60) 
GHOST_VALID = (0, 255, 0, 120)
GHOST_INVALID = (255, 0, 0, 120)
LABEL_COLOR = (180, 180, 180)
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER = (100, 100, 100)
TELEMETRY_COLOR = (255, 200, 100) 

PEG_MISS_COLOR = (220, 220, 220)
PEG_HIT_COLOR = (255, 50, 50)
PEG_SUNK_COLOR = (50, 150, 255)

DIAG_CLAIMED = (150, 150, 150)   
DIAG_ACTIVE = (0, 229, 255)      

HEAT_COLORS = [
    (10, 60, 120),  (50, 120, 160), (80, 160, 120), 
    (180, 180, 60), (220, 120, 40), (255, 50, 50)   
]

# ==========================================
# SHARED RENDERING FUNCTIONS
# ==========================================
def draw_grid_labels(offset_x):
    for i in range(GRID_SIZE):
        col_text = label_font.render(str(i + 1), True, LABEL_COLOR)
        col_x = offset_x + (i * CELL_SIZE) + (CELL_SIZE // 2) - (col_text.get_width() // 2)
        screen.blit(col_text, (col_x, MARGIN - 30))
        
        row_text = label_font.render(chr(65 + i), True, LABEL_COLOR)
        row_y = MARGIN + (i * CELL_SIZE) + (CELL_SIZE // 2) - (row_text.get_height() // 2)
        screen.blit(row_text, (offset_x - 30 - (row_text.get_width() // 2), row_y))

def draw_pegs_and_diagnostics(board, diagnostics, offset_x):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = pygame.Rect(offset_x + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            inner_rect = rect.inflate(-8, -8)

            track_val = board[row][col]
            if track_val == CELL_MISS:
                pygame.draw.circle(screen, PEG_MISS_COLOR, rect.center, CELL_SIZE // 6)
            elif track_val == CELL_HIT:
                pad = 12
                pygame.draw.line(screen, PEG_HIT_COLOR, (rect.left + pad, rect.top + pad), (rect.right - pad, rect.bottom - pad), 4)
                pygame.draw.line(screen, PEG_HIT_COLOR, (rect.right - pad, rect.top + pad), (rect.left + pad, rect.bottom - pad), 4)
            elif track_val == CELL_SUNK:
                pad = 10
                pygame.draw.line(screen, PEG_SUNK_COLOR, (rect.left + pad, rect.top + pad), (rect.right - pad, rect.bottom - pad), 6)
                pygame.draw.line(screen, PEG_SUNK_COLOR, (rect.right - pad, rect.top + pad), (rect.left + pad, rect.bottom - pad), 6)
            
            if diagnostics and (row, col) in diagnostics.get("active", []):
                pygame.draw.rect(screen, DIAG_ACTIVE, inner_rect, 4, border_radius=4)
                pygame.draw.circle(screen, DIAG_ACTIVE, rect.center, CELL_SIZE * 0.25, 3)

def draw_heatmap_grid(board, ai_bot, dead_sizes, offset_x):
    draw_grid_labels(offset_x)
    
    title = font.render(f"{ai_bot.__class__.__name__} Heatmap", True, (200, 200, 200))
    screen.blit(title, (offset_x, MARGIN - 65))

    heat_map = ai_bot.get_heatmap(board, dead_sizes)
    max_heat = max((max(row) for row in heat_map)) if heat_map else 0

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = pygame.Rect(offset_x + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            cell_heat = heat_map[row][col]
            if max_heat == 0 or cell_heat == 0:
                color_index = 0
            else:
                ratio = cell_heat / max_heat
                color_index = max(1, int(ratio * (len(HEAT_COLORS) - 1))) 

            pygame.draw.rect(screen, HEAT_COLORS[color_index], rect)
            if cell_heat > 0:
                heat_text = small_font.render(str(cell_heat), True, (255, 255, 255))
                screen.blit(heat_text, heat_text.get_rect(center=rect.center))
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

def get_bot_instance(bot_choice, weights_path=None):
    b = bot_choice.lower()
    if b in ['neural', 'moriarty', 'neural_luki', 'neural_rl', 'rl']:
        from bots.neural_bot import NeuralBot
        if b == 'neural_luki' and not weights_path:
            weights_path = os.path.join(os.path.dirname(__file__), 'data', 'moriarty_luki_best.pth')
        elif b in ['neural_rl', 'rl'] and not weights_path:
            weights_path = os.path.join(os.path.dirname(__file__), 'data', 'moriarty_rl_best.pth')
        return NeuralBot(weights_path=weights_path)
    elif b == 'entropy':
        from bots.entropy_bot import EntropyBot
        return EntropyBot()
    elif b == 'mycroft':
        from bots.mycroft import MycroftBot
        return MycroftBot()
    elif b in ['zdd', 'exact', 'population']:
        from bots.zdd_bot import ZDDBot
        return ZDDBot()
    elif b in ['porlock', 'porlock_bot']:
        from bots.porlock_bot import PorlockBot
        return PorlockBot()
    elif b in ['apex', 'sherrinford_apex']:
        from bots.sherrinford_apex import SherrinfordApexBot
        return SherrinfordApexBot()
    elif b in ['parity', 'sherrinford_parity']:
        from bots.sherrinford_parity import SherrinfordParityBot
        return SherrinfordParityBot()
    elif b in ['deduction', 'sherrinford_deduction']:
        from bots.sherrinford_deduction import SherrinfordDeductionBot
        return SherrinfordDeductionBot()
    elif b in ['sherrinford', 'sherrinford_bot']:
        from bots.sherrinford_bot import SherrinfordBot
        return SherrinfordBot()
    else:
        from bots.sherlock import SherlockBot
        return SherlockBot()

# ==========================================
# MODE 1: LIVE INTERACTIVE PLAY
# ==========================================
def run_live_mode(bot_choice, weights_path=None):
    pygame.display.set_caption(f"{bot_choice.capitalize()}Bot - Live Play & Diagnostics")
    
    def reset_state():
        g = BattleshipGame()
        bot = get_bot_instance(bot_choice, weights_path=weights_path)
        return g, bot, "SETUP", g.fleet_sizes.copy(), 1, True, -1, None, "", [], {}, True

    game, ai_bot, phase, ships_to_place, current_ship_id, is_horizontal, last_shot_count, ai_target, strategy, dead_sizes, diagnostics, show_ships = reset_state()
    
    left_grid_offset = MARGIN
    right_grid_offset = (MARGIN * 2) + (CELL_SIZE * GRID_SIZE)
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        hover_col = (mouse_pos[0] - left_grid_offset) // CELL_SIZE
        hover_row = (mouse_pos[1] - MARGIN) // CELL_SIZE
        on_left_grid = (0 <= hover_col < GRID_SIZE) and (0 <= hover_row < GRID_SIZE)

        if phase == "PLAY" and not game.is_game_over():
            if game.shot_number != last_shot_count:
                ai_target, strategy, dead_sizes, diagnostics = ai_bot.get_best_move(game.tracking_board, game.move_history)
                last_shot_count = game.shot_number
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u and phase == "PLAY": game.undo_last_shot()
                elif event.key == pygame.K_h and phase == "PLAY": show_ships = not show_ships
                elif event.key == pygame.K_SPACE and phase == "PLAY" and ai_target: game.fire_shot(ai_target[0], ai_target[1])
                elif event.key == pygame.K_r and phase == "SETUP": game.place_ships_randomly(); phase = "PLAY"; ships_to_place.clear()
                elif event.key == pygame.K_n: game, ai_bot, phase, ships_to_place, current_ship_id, is_horizontal, last_shot_count, ai_target, strategy, dead_sizes, diagnostics, show_ships = reset_state()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if phase == "SETUP":
                    if event.button == 3: is_horizontal = not is_horizontal
                    elif event.button == 1 and on_left_grid and ships_to_place:
                        if game.place_ship_manual(hover_row, hover_col, ships_to_place[0], not is_horizontal, current_ship_id):
                            ships_to_place.pop(0); current_ship_id += 1
                            if not ships_to_place: phase = "PLAY"
                elif phase == "PLAY" and event.button == 1 and on_left_grid:
                    game.fire_shot(hover_row, hover_col)

        screen.fill(BG_COLOR)
        draw_grid_labels(left_grid_offset)
        title = font.render("Play & Diagnostics Board", True, (200, 200, 200))
        screen.blit(title, (left_grid_offset, MARGIN - 65))

        # Draw Left Grid Background
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                pygame.draw.rect(screen, WATER_COLOR, pygame.Rect(left_grid_offset + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, GRID_COLOR, pygame.Rect(left_grid_offset + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        # Draw Ships
        if phase == "SETUP" or show_ships:
            for ship_id, pegs in game.ship_pegs.items():
                if pegs:
                    min_row, max_row = min(p[0] for p in pegs), max(p[0] for p in pegs)
                    min_col, max_col = min(p[1] for p in pegs), max(p[1] for p in pegs)
                    ship_rect = pygame.Rect(left_grid_offset + min_col * CELL_SIZE, MARGIN + min_row * CELL_SIZE, (max_col - min_col + 1) * CELL_SIZE, (max_row - min_row + 1) * CELL_SIZE).inflate(-8, -8)
                    pygame.draw.rect(screen, SHIP_COLOR, ship_rect, border_radius=15)

        draw_pegs_and_diagnostics(game.tracking_board, diagnostics, left_grid_offset)

        if phase == "SETUP":
            if on_left_grid and ships_to_place:
                ship_width = (ships_to_place[0] * CELL_SIZE) if is_horizontal else CELL_SIZE
                ship_height = CELL_SIZE if is_horizontal else (ships_to_place[0] * CELL_SIZE)
                ghost_rect = pygame.Rect(left_grid_offset + hover_col * CELL_SIZE, MARGIN + hover_row * CELL_SIZE, ship_width, ship_height).inflate(-8, -8)
                color = GHOST_VALID if game.can_place(hover_row, hover_col, ships_to_place[0], not is_horizontal) else GHOST_INVALID
                pygame.draw.rect(screen, color, ghost_rect, border_radius=15)
            screen.blit(small_font.render("Left Click: Place | Right Click: Rotate | R: Randomize", True, LABEL_COLOR), (MARGIN, WINDOW_HEIGHT - 50))
        
        elif phase == "PLAY":
            draw_heatmap_grid(game.tracking_board, ai_bot, dead_sizes, right_grid_offset)
            if not game.is_game_over():
                screen.blit(small_font.render(f"Strategy: {strategy}   |   [SPACEBAR] Auto-Fire   |   [U] Undo   |   [H] Toggle Ships", True, TELEMETRY_COLOR), (MARGIN, WINDOW_HEIGHT - 50))
            else:
                screen.blit(font.render(f"Game Over! Cleared in {game.shot_number} shots. Press [N] to restart.", True, (0, 255, 0)), (MARGIN, WINDOW_HEIGHT - 60))

        pygame.display.flip()
        pygame.time.Clock().tick(30) 

# ==========================================
# MODE 2: FORENSIC REPLAY
# ==========================================
def run_replay_mode(bot_choice):
    bot_prefix = "mycroft" if bot_choice.lower() == 'mycroft' else "sherlock"
    pygame.display.set_caption(f"{bot_choice.capitalize()}Bot - Forensic Replay Viewer")
    
    csv_files = glob.glob(f"data/{bot_prefix}_outliers_*.csv")
    if not csv_files:
        print(f"No outlier CSV files found in data/ for {bot_prefix}!")
        sys.exit()
    
    latest_file = max(csv_files, key=os.path.getctime)
    all_games = []
    with open(latest_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            all_games.append({
                "game_id": row["Game_ID"],
                "total_shots": int(row["Total_Shots"]),
                "ships": json.loads(row["Ship_Layout"]),
                "history": json.loads(row["Shot_Sequence"])
            })

    ai_bot = get_bot_instance(bot_choice)
    min_shots_filter = 0
    filtered_games = all_games
    current_game_index = 0
    current_shot_index = 0
    
    left_grid_offset = MARGIN
    right_grid_offset = (MARGIN * 2) + (CELL_SIZE * GRID_SIZE)
    
    running = True
    while running:
        if filtered_games:
            game_data = filtered_games[current_game_index]
            total_shots = game_data["total_shots"]
            history = game_data["history"]
            
            # Reconstruct State
            board = [[CELL_UNKNOWN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            move_history = []
            for i in range(current_shot_index):
                x, y, res = history[i]
                board[x][y] = CELL_MISS if res == "MISS" else CELL_HIT if res == "HIT" else CELL_SUNK
                move_history.append({"x": x, "y": y, "result": res, "shotNumber": i + 1})
            
            # Sync Parity & Get Diagnostics
            if history:
                first_x, first_y, _ = history[0]
                ai_bot.parity_offset = (first_x + first_y) % 2
                if hasattr(ai_bot, 'sherlock'): ai_bot.sherlock.parity_offset = ai_bot.parity_offset
            
            deduction_engine = ai_bot.sherlock if hasattr(ai_bot, 'sherlock') else ai_bot
            active_hits, dead_ships, _, dead_sizes = deduction_engine._analyze_board_state(board, move_history)
            
            if getattr(ai_bot, 'last_calculated_shot', -1) != current_shot_index:
                ai_bot.get_best_move(board, move_history) 
                ai_bot.last_calculated_shot = current_shot_index

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    min_shots_filter += 5; filtered_games = [g for g in all_games if g["total_shots"] >= min_shots_filter]; current_game_index = current_shot_index = 0
                elif event.key == pygame.K_s:
                    min_shots_filter = max(0, min_shots_filter - 5); filtered_games = [g for g in all_games if g["total_shots"] >= min_shots_filter]; current_game_index = current_shot_index = 0
                if filtered_games:
                    if event.key == pygame.K_RIGHT: current_shot_index = min(current_shot_index + 1, total_shots)
                    elif event.key == pygame.K_LEFT: current_shot_index = max(current_shot_index - 1, 0)
                    elif event.key == pygame.K_UP: current_game_index = max(current_game_index - 1, 0); current_shot_index = 0
                    elif event.key == pygame.K_DOWN: current_game_index = min(current_game_index + 1, len(filtered_games) - 1); current_shot_index = 0

        screen.fill(BG_COLOR)
        
        if filtered_games:
            screen.blit(font.render(f"[FORENSIC REPLAY] Game {current_game_index + 1}/{len(filtered_games)}", True, TELEMETRY_COLOR), (MARGIN, MARGIN - 65))
            screen.blit(font.render(f"Frame: {current_shot_index} / {total_shots} Shots", True, LABEL_COLOR), (right_grid_offset, MARGIN - 65))
            
            draw_grid_labels(left_grid_offset)
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    pygame.draw.rect(screen, WATER_COLOR, pygame.Rect(left_grid_offset + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, GRID_COLOR, pygame.Rect(left_grid_offset + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

            # Draw Ships
            for ship_id, pegs in game_data["ships"].items():
                if pegs:
                    min_row, max_row = min(p[0] for p in pegs), max(p[0] for p in pegs)
                    min_col, max_col = min(p[1] for p in pegs), max(p[1] for p in pegs)
                    ship_rect = pygame.Rect(left_grid_offset + min_col * CELL_SIZE, MARGIN + min_row * CELL_SIZE, (max_col - min_col + 1) * CELL_SIZE, (max_row - min_row + 1) * CELL_SIZE).inflate(-8, -8)
                    pygame.draw.rect(screen, SHIP_COLOR, ship_rect, border_radius=15)

            draw_pegs_and_diagnostics(board, {"active": active_hits, "dead_ships": dead_ships}, left_grid_offset)
            draw_heatmap_grid(board, ai_bot, dead_sizes, right_grid_offset)
        else:
            screen.blit(font.render(f"No games found with >= {min_shots_filter} shots.", True, LABEL_COLOR), (MARGIN, MARGIN))
        
        screen.blit(font.render("UP/DOWN: Change Game | LEFT/RIGHT: Scrub Timeline | W/S: Adjust Filter", True, (200, 200, 200)), (MARGIN, WINDOW_HEIGHT - 50))
        pygame.display.flip()
        pygame.time.Clock().tick(30) 

# ==========================================
# BOOT ROUTER
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Battleship Visualizer Suite")
    parser.add_argument('--bot', type=str, choices=['porlock', 'apex', 'sherrinford', 'parity', 'deduction', 'sherlock', 'mycroft', 'neural', 'moriarty', 'neural_luki', 'neural_rl', 'rl', 'entropy'], default='porlock', help='Choose bot to evaluate')
    parser.add_argument('--mode', type=str, choices=['live', 'replay'], default='live', help='Choose Visualizer Mode')
    parser.add_argument('--weights', type=str, default=None, help='Custom PyTorch weights file for NeuralBot')
    args = parser.parse_args()
    
    if args.mode == 'live':
        run_live_mode(args.bot, weights_path=args.weights)
    else:
        run_replay_mode(args.bot)