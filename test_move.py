import chess
import numpy as np
import tensorflow as tf

def board_to_feature_vector(board):
    piece_map = board.piece_map()
    feature_vector = np.zeros((8, 8, 12), dtype=np.uint8)

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        layer = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}[piece.symbol()]
        feature_vector[row, col, layer] = 1

    return feature_vector

def predict_best_move(fen_string):
    # Load the trained model
    model = tf.keras.models.load_model('trained_model.h5')

    # Create a chess board from the FEN string
    board = chess.Board(fen_string)

    # Initialize best value and best move
    best_value = -np.inf
    best_move = None

    # Loop over all legal moves
    for move in board.legal_moves:
        # Apply the move to a copy of the board
        board_copy = board.copy()
        board_copy.push(move)

        # Convert the board to a feature vector
        feature_vector = board_to_feature_vector(board_copy)

        # Add an extra dimension because the model expects a batch
        feature_vector = np.expand_dims(feature_vector, axis=0)

        # Use the model to predict the outcome
        value = model.predict(feature_vector)

        # If this move has a higher value than the current best, update best_value and best_move
        if value > best_value:
            best_value = value
            best_move = move

    return best_move

if __name__ == "__main__":
    fen_string = "r2qkb1r/1pp2ppp/p1n1b3/3pp1P1/B2N3P/2Pn4/PP1P1P2/RNBQ1K1R w kq - 3 13"  # insert your FEN string here
    best_move = predict_best_move(fen_string)
    print("Predicted best move:", best_move)