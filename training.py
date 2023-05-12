#!/usr/bin/env python3
from tensorflow.keras import layers
import os
import numpy as np
import chess.pgn
import time
import tensorflow as tf
import chess


def get_dataset(num_samples=None, total_samples_limit=None):
    X, Y = [], []
    gn = 0
    total_samples = 0 
    values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    with open(r'C:/Users/ataka/Desktop/games.pgn') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            res = game.headers['Result']
            if res not in values:
                continue
            value = values[res]
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                if total_samples_limit is not None and total_samples >= total_samples_limit: 
                    return
                board.push(move)
                ser = board_to_feature_vector(board)
                X.append(ser)
                Y.append(value)
                total_samples += 1  
                if num_samples is not None and len(X) >= num_samples:
                    yield np.array(X), np.array(Y)
                    X, Y = [], [] 
            print("parsing game %d, got %d examples" % (gn, len(X)))
            gn += 1



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

def train_model(model, X, Y):
    model.fit(X, Y, epochs=10, batch_size=64, validation_split=0.2)


if __name__ == "__main__":
    cnn_model = create_cnn()
    start_time = time.time()
    batch_rounds = 0  
    total_examples = 0  

    try:
        for X, Y in get_dataset(10000, 2000000):  # generate batches of 10000, stop after 2000000 total samples
            cnn_model.fit(X, Y, epochs=10)
            batch_rounds += 1 
            total_examples += len(X) 
            elapsed_time = time.time() - start_time  
            print(f"Finished batch round {batch_rounds}. Total examples: {total_examples}. Elapsed time: {elapsed_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("Interrupted, saving model...")
    finally:
        elapsed_time = time.time() - start_time 
        print(f"Total examples processed: {total_examples}. Total elapsed time: {elapsed_time:.2f} seconds.")
        cnn_model.save('trained_model.h5')
