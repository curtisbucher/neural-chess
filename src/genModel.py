# For reletive imports
import sys
sys.path.append('../../neural-chess')

# Various imports
import numpy as np
from tensorflow import keras
from keras import layers

from data.loader import *

# Setup plotting
import matplotlib.pyplot as plt

# Set Matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

## Defining constants
TRAIN_AUTOENCODER = 0
TRAIN_NET = 1

TOTAL_AE = 250000
TOTAL_MLP = 750000

BS_AE = 20
BS_MLP = 50
EPOCHS_AE = 50
EPOCHS_MLP = 201
RATE_AE = 0.005
DECAY_AE = 0.98
RATE_MLP = 0.005
DECAY_MLP = 0.98

BIAS = 0.15

N_INPUT = 769
ENCODING_1 = 600
ENCODING_2 = 400
ENCODING_3 = 200
ENCODING_4 = 100

HIDDEN_1 = 200
HIDDEN_2 = 400
HIDDEN_3 = 200
HIDDEN_4 = 100
N_OUT = 2

VOLUME_SIZE = 25000

export_path = '../model'

#Get the data from the game files
validation_test, validation_test_l = getTest(N_INPUT, 40, 44)
whiteWins, blackWins = getTrain(N_INPUT, TOTAL_MLP, VOLUME_SIZE)

"""
CREATING SIAMESE AUTOENCODERS
"""
sample_size = 10000
# Defining a decaying learning rate
p2v_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=RATE_AE,
    decay_steps= sample_size / 200,
    decay_rate=DECAY_AE,
    name="p2v_schedule")
p2v_decay_optimizer = keras.optimizers.SGD(learning_rate=p2v_schedule)

# Defining a short circuiting early stopping function
early_stopping = keras.callbacks.EarlyStopping(
    patience=500,
    min_delta=0.0001,
    restore_best_weights=True,
)

print("Generating and training siamese autoencoders")
# Layer 1
Pos2Vec = keras.Sequential([
    layers.Dense(ENCODING_1, activation='relu', input_shape=(N_INPUT,)),
    layers.Dense(N_INPUT, activation='sigmoid'),
])

# Compiling model
Pos2Vec.compile(
    optimizer=p2v_decay_optimizer,
    loss='mean_squared_error',
)


# whiteWins, blackWins
print("Generating random train")
sample_size = 10000#1000000
test_size= 1000#1000
white_sample = whiteWins[np.random.randint(whiteWins.shape[0], size=sample_size), :]
black_sample = blackWins[np.random.randint(blackWins.shape[0], size=sample_size), :]

white_test = whiteWins[np.random.randint(whiteWins.shape[0], size=test_size), :]
black_test = blackWins[np.random.randint(blackWins.shape[0], size=test_size), :]

autoencoder_train = np.concatenate([white_sample, black_sample])
autoencoder_test = np.concatenate([white_test, black_test])

print("Training")
history = Pos2Vec.fit(
    autoencoder_train, autoencoder_train,
    validation_data=(autoencoder_test, autoencoder_test),
    batch_size=None,
    epochs=10, #200
    callbacks=[early_stopping],
    verbose=1,
)


# Training the 400 node layer
layer_1 = Pos2Vec.layers[0] # 769 - 600nodes
layer_2 = Pos2Vec.layers[1] # 600 - 769nodes

layer_1.trainable = False
layer_2.trainable = False

Pos2Vec_2 = keras.Sequential([
    # layers.Dense(769, activation='relu', input_shape=input_shape),
    layer_1,
    layers.Dense(ENCODING_2, activation='relu'),
    layers.Dense(ENCODING_1, activation='relu'),
    layer_2,
])

# Compiling model
Pos2Vec_2.compile(
    optimizer=p2v_decay_optimizer,
    loss='mean_squared_error',
)


print("Training")
history = Pos2Vec_2.fit(
    autoencoder_train, autoencoder_train,
    validation_data=(autoencoder_test, autoencoder_test),
    batch_size=None,
    epochs=10, #200
    callbacks=[early_stopping],
    verbose=1,
)


# Training the 400 node layer
layer_3 = Pos2Vec_2.layers[1] # 600 - 400nodes
layer_4 = Pos2Vec_2.layers[2] # 400 - 600nodes

layer_3.trainable = False
layer_4.trainable = False

Pos2Vec_3 = keras.Sequential([
    # layers.Dense(769, activation='relu', input_shape=input_shape),
    layer_1,
    layer_3,
    layers.Dense(ENCODING_3, activation='relu'),
    layers.Dense(ENCODING_2, activation='relu'),
    layer_4,
    layer_2,
])


# Compiling model
Pos2Vec_3.compile(
    optimizer=p2v_decay_optimizer,
    loss='mean_squared_error',
)


print("Training")
history = Pos2Vec_3.fit(
    autoencoder_train, autoencoder_train,
    validation_data=(autoencoder_test, autoencoder_test),
    batch_size=None,
    epochs=10, #200
    callbacks=[early_stopping],
    verbose=1,
)


# Training the 400 node layer
layer_5 = Pos2Vec_3.layers[2] # 400 - 200nodes
layer_6 = Pos2Vec_3.layers[3] # 200 - 400nodes

layer_5.trainable = False
layer_6.trainable = False

Pos2Vec_4 = keras.Sequential([
    # layers.Dense(769, activation='relu', input_shape=input_shape),
    layer_1,
    layer_3,
    layer_5,
    layers.Dense(ENCODING_4, activation='relu'),
    layers.Dense(ENCODING_3, activation='relu'),
    layer_6,
    layer_4,
    layer_2,
])

# Compiling model
Pos2Vec_4.compile(
    optimizer=p2v_decay_optimizer,
    loss='mean_squared_error',
)

print("Training")
history = Pos2Vec_4.fit(
    autoencoder_train, autoencoder_train,
    validation_data=(autoencoder_test, autoencoder_test),
    batch_size=None,
    epochs=10, #200
    callbacks=[early_stopping],
    verbose=1,
)

# Training the 400 node layer
layer_7 = Pos2Vec_4.layers[3] # 400 - 200nodes
layer_8 = Pos2Vec_4.layers[4] # 200 - 400nodes

layer_7.trainable = False
layer_8.trainable = False

Pos2Vec_A = keras.Sequential([
    # layers.Dense(769, activation='relu', input_shape=input_shape),
    layer_1,
    layer_3,
    layer_5,
    layer_7,
])


import pydotplus
from keras.utils.vis_utils import plot_model
from keras.layers import concatenate

pydot = pydotplus

# Duplicating siamese Pos2Vec with tied weights
la0, la1, la2, la3 = Pos2Vec_A.layers
Pos2Vec_B_in = keras.layers.Input(shape=(769,))
lc0 = la0(Pos2Vec_B_in)
lc1 = la1(lc0)
lc2 = la2(lc1)
lc2 = la3(lc2)

Pos2Vec_B = keras.Model(inputs=Pos2Vec_B_in, outputs=lc2)

# Creating DeepChess layers to compare pos2vec
twin_p2v_in = concatenate([Pos2Vec_A.output, Pos2Vec_B.output])
l0 = layers.Dense(HIDDEN_2, activation="relu")(twin_p2v_in)
l1 = layers.Dense(HIDDEN_3, activation="relu")(l0)
l2 = layers.Dense(HIDDEN_4, activation="relu")(l1)
deepchess_out = layers.Dense(N_OUT, activation="relu")(l2)

DeepChess = keras.Model(
    inputs=[Pos2Vec_A.input, Pos2Vec_B.input],
    outputs=[deepchess_out])

plot_model(DeepChess,
            show_shapes=True)


# whiteWins, blackWins
print("Generating random sample")
train_size = 10000#1000000
test_size= 1000#100000

## Generating training data
# sampling white wins and losses (black wins)
white_w_train = whiteWins[np.random.randint(whiteWins.shape[0], size=train_size), :]
white_l_train = blackWins[np.random.randint(blackWins.shape[0], size=train_size), :]

# Creating (W, L) or (L, W) pairs
DeepChess_in_A = np.concatenate((white_w_train[:train_size // 2], white_l_train[:train_size // 2]))
DeepChess_in_B = np.concatenate((white_l_train[train_size // 2:], white_w_train[train_size // 2:]))

# Creating (1, 0) or (0, 1) pairs corresponding to input
DeepChess_out = np.array([(1,0)] * (train_size // 2) +
                            [(0,1)] * (train_size // 2))


## Generating testing data
# test white wins and losses (black wins)
white_w_test= whiteWins[np.random.randint(whiteWins.shape[0], size=test_size), :]
white_l_test= blackWins[np.random.randint(blackWins.shape[0], size=test_size), :]

# Creating (W, L) or (L, W) pairs
DeepChess_test_in_A = np.concatenate((white_w_test[:test_size // 2], white_l_test[:test_size // 2]))
DeepChess_test_in_B = np.concatenate((white_l_test[test_size // 2:], white_w_test[test_size // 2:]))

# Creating (1, 0) or (0, 1) pairs corresponding to input
DeepChess_test_out = np.array([(1,0)] * (test_size // 2) +
                            [(0,1)] * (test_size // 2))

dc_schedule = keras.optimizers.schedules.ExponentialDecay(
    0.01,
    decay_steps=100000,
    decay_rate=0.99)

dc_decay_optimizer = keras.optimizers.SGD(learning_rate=dc_schedule)

# Compiling model
DeepChess.compile(
    optimizer=dc_decay_optimizer,
    loss='mean_squared_error',
)

print("Training")
history = DeepChess.fit(
    x=[DeepChess_in_A, DeepChess_in_B], y=DeepChess_out,
    validation_split=0.1,
    batch_size=None,
    epochs=100, #1000
    callbacks=[early_stopping],
    use_multiprocessing=True,
    verbose=1,
)


