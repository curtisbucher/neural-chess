# For reletive imports
import sys
sys.path.append('../../neural-chess')
import data.util as util
import keras
import numpy as np

from tensorflow import Tensor

from keras.utils.vis_utils import plot_model

fen1 = util.bitifyFEN(
        util.beautifyFEN(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))

fen2 = util.bitifyFEN(
        util.beautifyFEN(
            "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"))

DeepChess = keras.models.load_model("../model/DeepChess")
print(DeepChess.summary())

input = [np.array(fen1), np.array(fen2)]
a = DeepChess(input)

print(a)