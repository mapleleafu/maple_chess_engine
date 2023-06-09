import os
import chess
import time
import traceback
import signal
import sys
from state import State
import numpy as np
import tensorflow as tf
import base64
from flask import Flask, Response, request

MAXVAL = 10000

model = tf.keras.models.load_model('trained_model.h5')

moves = []
move_times = []

class ClassicValuator(object):
    values = {chess.PAWN: 1,
              chess.KNIGHT: 3.2,
              chess.BISHOP: 3.3,
              chess.ROOK: 5,
              chess.QUEEN: 9,
              chess.KING: 0}

    def __init__(self):
        self.reset()
        self.memo = {}

    def reset(self):
        self.count = 0

    def __call__(self, s):
        self.count += 1
        key = s.key()
        if key not in self.memo:
            self.memo[key] = self.value(s)
        return self.memo[key]

    def value(self, s):
        b = s.board
        # game over values
        if b.is_game_over():
            if b.result() == "1-0":
                return MAXVAL
            elif b.result() == "0-1":
                return -MAXVAL
            else:
                return 0

        val = 0.0
        # piece values
        pm = s.board.piece_map()
        for x in pm:
            tval = self.values[pm[x].piece_type]
            if pm[x].color == chess.WHITE:
                val += tval
            else:
                val -= tval

        # add a number of legal moves term
        bak = b.turn
        b.turn = chess.WHITE
        val += 0.1 * b.legal_moves.count()
        b.turn = chess.BLACK
        val -= 0.1 * b.legal_moves.count()
        b.turn = bak

        return val


class Valuator(object):
    def __init__(self):
        self.memo = {}
        self.count = 0
        self.values = [0, 1, 3, 3, 5, 9, 0]  # From pawn to king

    def reset(self):
        self.count = 0

    def __call__(self, s):
        input_data = s.serialize()
        output = model.predict(input_data.reshape(1, 5, 8, 8))
        return output

    def value(self, s):
        b = s.board
        # game over values
        if b.is_game_over():
            if b.result() == "1-0":
                return MAXVAL
            elif b.result() == "0-1":
                return -MAXVAL
            else:
                return 0

        val = 0.0
        # piece values
        pm = s.board.piece_map()
        for x in pm:
            tval = self.values[pm[x].piece_type]
            if pm[x].color == chess.WHITE:
                val += tval
            else:
                val -= tval

        # add a number of legal moves term
        bak = b.turn
        b.turn = chess.WHITE
        val += 0.1 * b.legal_moves.count()
        b.turn = chess.BLACK
        val -= 0.1 * b.legal_moves.count()
        b.turn = bak

        return val




def computer_minimax(s, v, depth, a, b, big=False):
    #! Change the depth, plays good with 5 depth 
    if depth >= 5 or s.board.is_game_over(): 
        return v(s)
    # white is maximizing player
    turn = s.board.turn
    if turn == chess.WHITE:
        ret = -MAXVAL
    else:
        ret = MAXVAL
    if big:
        bret = []

    checks, captures, others = order_moves(s.board)
    ordered_moves = checks + captures + others

    # can prune here with beam search
    isort = []
    for e in ordered_moves:
        s.board.push(e)
        isort.append((v(s), e))
        s.board.pop()
    move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)

    # beam search beyond depth 3
    if depth >= 3:
        move = move[:10]

    for e in [x[1] for x in move]:
        s.board.push(e)
        tval = computer_minimax(s, v, depth+1, a, b)
        s.board.pop()
        if big:
            bret.append((tval, e))
        if turn == chess.WHITE:
            ret = max(ret, tval)
            a = max(a, ret)
            if a >= b:
                break  # b cut-off
        else:
            ret = min(ret, tval)
            b = min(b, ret)
            if a >= b:
                break  # a cut-off
    if big:
        return ret, bret
    else:
        return ret

def print_statistics():
    avg_time = sum(move_times) / len(move_times) if move_times else 0
    print("Average time per move: {:.2f} seconds".format(avg_time))
    print("Total moves: {}".format(len(moves)//2))
    print("Moves: {}".format(' '.join(moves)))
    print("Lichess analysis: https://lichess.org/analysis/{}".format(s.board.fen()))


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    print_statistics()
    sys.exit(0)

if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    signal.signal(signal.SIGINT, signal_handler)


def order_moves(board):
    checks = []
    captures = []
    others = []
    for move in board.legal_moves:
        if board.is_check():
            checks.append(move)
        elif board.is_capture(move):
            captures.append(move)
        else:
            others.append(move)
    return checks, captures, others


def explore_leaves(s, v):
    ret = []
    start = time.time()
    v.reset()
    bval = v(s)
    cval, ret = computer_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
    eta = time.time() - start
    print(
        "%.2f -> %.2f: explored %d nodes in %.3f seconds %d/sec"
        % (bval, cval, v.count, eta, int(v.count / eta))
    )
    return ret


s = State()
v = ClassicValuator()


def to_svg(s):
    return base64.b64encode(chess.svg.board(board=s.board).encode("utf-8")).decode("utf-8")


app = Flask(__name__)


@app.route("/")
def hello():
    ret = open("index.html").read()
    return ret.replace("start", s.board.fen())


@app.route("/move_coordinates")
def move_coordinates():
    if not s.board.is_game_over():
        source = int(request.args.get("from", default=""))
        target = int(request.args.get("to", default=""))
        promotion = (
            True if request.args.get("promotion", default="") == "true" else False
        )
        move = s.board.san(
            chess.Move(
                source, target, promotion=chess.QUEEN if promotion else None
            )
        )
        if move is not None and move != "":
            try:
                start_time = time.time()
                s.board.push_san(move)
                move_times.append(time.time() - start_time)
                moves.append(move)  # Append player move to moves list
                computer_move(s, v)
            except Exception:
                traceback.print_exc()
                return app.response_class(response="Illegal move", status=400)
        return app.response_class(response=s.board.fen(), status=200)
    else:
        print("GAME IS OVER")
        print_statistics()  # Print statistics at end of game
        return app.response_class(response="game over", status=200)
    

@app.route("/newgame")
def newgame():
    s.board.reset()
    response = app.response_class(response=s.board.fen(), status=200)
    return response


def computer_move(s, v):
    # computer move
    move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
    if len(move) == 0:
        return
    print("top 3:")
    for i, m in enumerate(move[0:3]):
        print("  ", m)
    print(("White" if s.board.turn else "Black"), "moving", move[0][1])
    san_move = s.board.san(move[0][1])  # Convert move to SAN before pushing
    s.board.push(move[0][1])
    moves.append(san_move)  # Append AI move to moves list


@app.route("/selfplay")
def selfplay():
    while not s.board.is_game_over():
        computer_move(s, v)
    return "Game over"


if __name__ == "__main__":
    app.run(debug=True)
