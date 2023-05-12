from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import chess.pgn
import os
import time

load_dotenv()
games_dir = os.getenv("GAMES_DIR")

pgn_file_path = f"{games_dir}"

with open(pgn_file_path, "r") as pgn_file:
    games = []

    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        games.append(game)

def board_to_feature_vector(board):
    piece_map = board.piece_map()
    feature_vector = np.zeros((8, 8, 12), dtype=np.uint8)

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        layer = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}[piece.symbol()]
        feature_vector[row, col, layer] = 1

    return feature_vector

def create_cnn(input_shape=(8, 8, 12)):
    model = tf.keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1, activation="tanh")
    ])

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])
    return model

cnn_model = create_cnn()

def data_stream(pgn_file_path, batch_size, max_moves_per_game=50):
    with open(pgn_file_path, "r") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            X = []
            y = []

            board_state = game.board().fen()  # Store the board state as a string
            result = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[game.headers["Result"]]

            for i, move in enumerate(game.mainline_moves()):
                if i >= max_moves_per_game:
                    break

                board = chess.Board(board_state)
                board.push(move)
                board_state = board.fen()

                X.append(board_to_feature_vector(board))
                y.append(result)

                if len(X) == batch_size:
                    yield np.array(X), np.array(y)
                    X.clear()
                    y.clear()




batch_size = 16
num_games = 1

# Measure the start time
start_time = time.time()

for X_batch, y_batch in data_stream(pgn_file_path, batch_size, max_moves_per_game=30):
    cnn_model.train_on_batch(X_batch, y_batch)



# Measure the end time
end_time = time.time()

# Calculate the elapsed time
training_time = end_time - start_time
print("Training time:", training_time, "seconds")

cnn_model.save("chess_cnn_model.h5")
loaded_model = tf.keras.models.load_model("chess_cnn_model.h5")

def evaluate_position(model, board):
    feature_vector = board_to_feature_vector(board)
    input_data = np.expand_dims(feature_vector, axis=0)
    return model.predict(input_data)[0][0]

def alpha_beta_search(model, board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_position(model, board), None

    best_move = None

    if maximizing_player:
        max_eval = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = alpha_beta_search(model, board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = alpha_beta_search(model, board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def find_best_move(model, board, depth=3):
    _, best_move = alpha_beta_search(model, board, depth, -float("inf"), float("inf"), board.turn)
    return best_move
