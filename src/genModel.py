# Various imports
from matplotlib import blocking_input
import numpy as np
from tensorflow import keras
from keras import layers
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint

# For reletive imports
import sys
sys.path.append('../../neural-chess')
from data.loader import *

# Setup plotting
import matplotlib.pyplot as plt

import typing
from typing import List, Tuple
import numpy.typing as npt

import pydotplus
from keras.utils.vis_utils import plot_model
pydot = pydotplus

import argparse

# Set Matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

## Defining constants
TRAIN_AUTOENCODER = 1
TRAIN_NET = 1

TOTAL_AE = 250000   #20
TOTAL_MLP = 750000
TRAIN_TEST_RATIO = 0.1

BS_AE = 20
BS_MLP = 50
EPOCHS_AE = 20 #50
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

SEED = 0

DC_EXPORT_PATH = "../model/DeepChess"
DC_AUTOSAVE_PATH = "../model/DeepChess_Autosave"
AE_EXPORT_PATH = "../model/P2V"

def gen_pos_to_vec( training_data: Tuple[npt.NDArray, npt.NDArray],
                    train_size: int = TOTAL_AE,
                    validation_split: float = TRAIN_TEST_RATIO,
                    batch_size: int = BS_AE,
                    epochs: int= EPOCHS_AE,
                    opt_init_rate: float = RATE_AE,
                    opt_decay_rate: float = DECAY_AE,
                    structure: typing.List[int]=[ENCODING_1, ENCODING_2, ENCODING_3, ENCODING_4],
                    ) -> keras.Model:
    '''
        Generates board autoencoders for siamese input into neural-chess network
        Parameters:
            training_data : A tuple of training data (whiteWins, Blackwins)
            train_size : The number of training examples to use
            validation_split : The percentage of the training data to use for validation
            batch_size : The number of examples to use per batch
            epochs : The number of epochs to train for
            opt_init_rate : The initial learning rate for the optimizer
            opt_decay_rate : The decay rate for the learning rate
            structure : The structure of the autoencoder

        Returns:
            model : The autoencoder model
    '''

    whiteWins, blackWins = training_data

    # Defining a decaying learning rate
    p2v_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=opt_init_rate,
        decay_steps= train_size / 200,
        decay_rate=opt_decay_rate,
        name="p2v_schedule")
    p2v_decay_optimizer = keras.optimizers.SGD(learning_rate=p2v_schedule)
    optimizer = "adam"

    # Defining a short circuiting early stopping function
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
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
        optimizer=optimizer,
        loss='mean_squared_error',
    )

    # whiteWins, blackWins
    print("Generating random train")
    white_sample = whiteWins[np.random.randint(whiteWins.shape[0], size=train_size), :]
    black_sample = blackWins[np.random.randint(blackWins.shape[0], size=train_size), :]

    autoencoder_train = np.concatenate([white_sample, black_sample])

    del white_sample, black_sample

    print("Training")
    history = Pos2Vec.fit(
        autoencoder_train, autoencoder_train,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs, #200
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
        layers.Dense(structure[1], activation='relu'),
        layers.Dense(structure[0], activation='relu'),
        layer_2,
    ])

    # Compiling model
    Pos2Vec_2.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
    )

    print("Training")
    history = Pos2Vec_2.fit(
        autoencoder_train, autoencoder_train,
        validation_split=validation_split,
        batch_size=None,
        epochs=epochs, #200
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
        layers.Dense(structure[2], activation='relu'),
        layers.Dense(structure[1], activation='relu'),
        layer_4,
        layer_2,
    ])


    # Compiling model
    Pos2Vec_3.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
    )


    print("Training")
    history = Pos2Vec_3.fit(
        autoencoder_train, autoencoder_train,
        validation_split=validation_split,
        batch_size=None,
        epochs=epochs, #200
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
        layers.Dense(structure[3], activation='relu'),
        layers.Dense(structure[2], activation='relu'),
        layer_6,
        layer_4,
        layer_2,
    ])

    # Compiling model
    Pos2Vec_4.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
    )

    print("Training")
    history = Pos2Vec_4.fit(
        autoencoder_train, autoencoder_train,
        validation_split=validation_split,
        batch_size=None,
        epochs=epochs, #200
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

    layer_1.trainable = True
    layer_3.trainable = True
    layer_5.trainable = True
    layer_7.trainable = True

    # Pos2Vec_A.compile()

    return Pos2Vec_A

def gen_neural_chess(structure: typing.List[int]=[HIDDEN_1, HIDDEN_2, HIDDEN_3, HIDDEN_4],
                    ) -> keras.Model:
    '''
        Generates the deepchess network
        Parameters:
            epochs: int - The number of epochs
            opt_init_rate: float - The initial learning rate
            opt_decay_rate: float - The decay rate
            structure: List[int] - The structure of the network
        Returns:
            keras.Model - The trained deepchess network
    '''

    Pos2Vec: keras.Model = keras.models.load_model("../model/Pos2Vec")

    A_in = keras.layers.Input(shape=(769,))
    B_in = keras.layers.Input(shape=(769,))

    l0 = layers.Dense(600, activation='relu')
    l1 = layers.Dense(400, activation='relu')
    l2 = layers.Dense(200, activation='relu')
    l3 = layers.Dense(100, activation='relu')

    A = l3(l2(l1(l0(A_in))))
    B = l3(l2(l1(l0(B_in))))

    Pos2Vec_A = keras.models.Model(inputs=A_in, outputs=A)
    Pos2Vec_A.set_weights(Pos2Vec.get_weights())

    Pos2Vec_B = keras.models.Model(inputs=B_in, outputs=B)
    Pos2Vec_B.set_weights(Pos2Vec.get_weights())

    # Creating DeepChess layers to compare pos2vec
    twin_p2v_in = concatenate([Pos2Vec_A.output, Pos2Vec_B.output])
    l0 = layers.Dense(structure[1], activation="relu")(twin_p2v_in)
    l1 = layers.Dense(structure[2], activation="relu")(l0)
    l2 = layers.Dense(structure[3], activation="relu")(l1)
    l3 = layers.Dense(N_OUT)(l2)
    deepchess_out = layers.Softmax()(l3)

    DeepChess = keras.Model(
        inputs=[Pos2Vec_A.input, Pos2Vec_B.input],
        outputs=[deepchess_out])

    return DeepChess

def train_deepchess(training_data: Tuple[npt.NDArray, npt.NDArray],
                    filepath: str = DC_AUTOSAVE_PATH,
                    load_last: bool = True,
                    opt_init_rate: float = RATE_MLP,
                    opt_decay_rate: float = DECAY_MLP,
                    train_size: int = TOTAL_MLP,
                    validation_split: float = TRAIN_TEST_RATIO,
                    batch_size: int = BS_MLP,
                    epochs: int= EPOCHS_MLP
                    ) -> Tuple[keras.Model, keras.callbacks.History]:
    '''
        Trains the deepchess network
        Parameters:
            training_data:      Tuple[npt.NDArray, npt.NDArray] - The training data
            filepath:           str - The path to the file to load the data from
            load_last:          bool - Whether to load the last saved model
            train_size:         int - The size of the training data
            validation_split:   float - The percentage of the training data to be used for validation
            batch_size:         int - The batch size
            epochs:             int - The number of epochs

        Returns:
            keras.Model - The trained deepchess network
            history: keras.callbacks.History - The history of the training
    '''
    if load_last:
        DeepChess = keras.models.load_model(filepath)
    else:
        DeepChess = gen_neural_chess(training_data)

    # Defining a short circuiting early stopping function
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        min_delta=0.0001,
        restore_best_weights=True,
    )

    # Callback for saving model after each epoch
    checkpoint = ModelCheckpoint(DC_AUTOSAVE_PATH, monitor='val_accuracy', verbose=1,
                                    save_best_only=True, mode='auto', period=1)

    # whiteWins, blackWins
    print("Generating random sample")

    whiteWins, blackWins = training_data
    ## Generating training data
    # sampling white wins and losses (black wins)
    white_w_train = whiteWins[np.random.randint(whiteWins.shape[0], size=train_size), :]
    white_l_train = blackWins[np.random.randint(blackWins.shape[0], size=train_size), :]

    # Creating (W, L) or (L, W) pairs
    DeepChess_in_A = np.concatenate((white_w_train[:train_size // 2], white_l_train[:train_size // 2]))

    DeepChess_in_B = np.concatenate((white_l_train[train_size // 2:], white_w_train[train_size // 2:]))

    del white_w_train, white_l_train

    # Creating (1, 0) or (0, 1) pairs corresponding to input
    DeepChess_out = np.array([(1,0)] * (train_size // 2) +
                                [(0,1)] * (train_size // 2))

    dc_schedule = keras.optimizers.schedules.ExponentialDecay(
        opt_init_rate,
        decay_steps=epochs, ## TODO: WHAT?
        decay_rate=opt_decay_rate)
    dc_decay_optimizer = keras.optimizers.SGD(learning_rate=dc_schedule)
    optimizer= "adam"

    # Compiling model
    DeepChess.compile(
        optimizer=optimizer,
        loss= "categorical_crossentropy",
        metrics=['accuracy']
    )

    print("Training")
    history = DeepChess.fit(
        x=[DeepChess_in_A, DeepChess_in_B], y=DeepChess_out,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[early_stopping, checkpoint],
        use_multiprocessing=True,
        verbose=1,
    )

    return DeepChess, history

def plot_loss(history: keras.callbacks.History):
    '''
        Plots the loss of the deepchess network
        Parameters:
            history: keras.callbacks.History - The history of the training
    '''
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title('Model loss and accuracy')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    TRAIN_AE = 0
    TRAIN_NC = 1

    # TODO
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--depth", default=DEFAULT_DEPTH, help="provide an integer (default: 3)")
    # args = parser.parse_args()
    # return max([1, int(args.depth)])

    #Get the data from the game files
    print("Loading training data")
    # validation_test, validation_test_l = getTest(N_INPUT, 40, 44)
    training_data = getTrain(N_INPUT, TOTAL_MLP, VOLUME_SIZE)

    if TRAIN_AE:
        print("Generate Autoencoders")
        Pos2Vec = gen_pos_to_vec(training_data)
        Pos2Vec.save(AE_EXPORT_PATH)

    if TRAIN_NC:
        print("Training Model")
        DeepChess, history = train_deepchess(training_data)
        plot_loss(history)
        DeepChess.save(DC_EXPORT_PATH)

    print("Done!")