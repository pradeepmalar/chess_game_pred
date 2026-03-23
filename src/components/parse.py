import os
import glob
import chess.pgn
import pandas as pd
from tqdm import tqdm
import pyarrow  

class ChessDataParser:
    def __init__(self, pgn_folder):
        self.pgn_folder = pgn_folder
        self.pgn_data = []

    def convert_result(self, result_str):
        if result_str == "1-0":
            return 1
        elif result_str == "0-1":
            return 0
        elif result_str == "1/2-1/2":
            return 0.5
        return None

    def load_pgn_data(self, max_games=None):
        print("♟️ Loading PGN files from folder...")
        pgn_files = glob.glob(os.path.join(self.pgn_folder, "*.pgn"))

        total_loaded = 0
        for file_path in pgn_files:
            print(f"   Parsing: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as pgn:
                game_counter = 0
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break

                    result = self.convert_result(game.headers.get("Result", "*"))
                    if result is None:
                        continue

                    try:
                        white_elo = int(game.headers.get("WhiteElo", 0))
                        black_elo = int(game.headers.get("BlackElo", 0))
                    except ValueError:
                        continue

                    game_length = len(list(game.mainline_moves()))
                    eco = game.headers.get("ECO", "UNKNOWN")
                    opening = game.headers.get("Opening", "UNKNOWN")

                    self.pgn_data.append({
                        'white_elo': white_elo,
                        'black_elo': black_elo,
                        'result': result,
                        'game_length': game_length,
                        'eco': eco,
                        'opening': opening,
                        'source': os.path.basename(file_path)
                    })

                    game_counter += 1
                    total_loaded += 1
                    if max_games is not None and total_loaded >= max_games:
                        break
            if max_games is not None and total_loaded >= max_games:
                break

        print(f"Loaded {total_loaded} games from {len(pgn_files)} PGN files")

    def get_dataframe(self):
        return pd.DataFrame(self.pgn_data)

def save_features_to_disk(dataframe, output_path, method="feather"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if method == "feather":
        dataframe.reset_index(drop=True).to_feather(output_path)
        print(f"Features saved to: {output_path} (format: feather)")
    elif method == "pkl":
        dataframe.to_pickle(output_path)
        print(f"Features saved to: {output_path} (format: pickle)")
    else:
        raise ValueError("Supported formats: 'feather' or 'pkl'")

# ==== Run the parser ====

pgn_folder = "C:/Users/chall/Documents/chess_game_pred/data/professional"
output_file = "C:/Users/chall/Documents/chess_game_pred/data/parsed/chess_games.feather"

parser = ChessDataParser(pgn_folder)
parser.load_pgn_data()  # Add max_games=50000 if testing
df = parser.get_dataframe()
save_features_to_disk(df, output_file, method="feather")