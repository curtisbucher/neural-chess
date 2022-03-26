from tensorflow import keras
from numpy import array
import chess

import sys
sys.path.append('../../neural-chess')
import data.util as util

MODEL = keras.models.load_model("model/DeepChess")
MAX_POS = 0
MIN_POS = 1


def compare_positions(pos1: chess.Board, pos2: chess.Board):
    """
    Compare two positions.
    """
    if (pos1 == MAX_POS) or (pos2 == MIN_POS):
        return 1, 0
    elif (pos1 == MIN_POS) or (pos2 == MAX_POS):
        return 0, 1

    board1 = util.bitifyFEN(
        util.beautifyFEN(
            pos1.fen()))

    board2 = util.bitifyFEN(
            util.beautifyFEN(
                pos2.fen()))

    a = array([board1])
    b = array([board2])

    out = MODEL.predict([a, b])[0]

    # print(out)
    return out[0], out[1]