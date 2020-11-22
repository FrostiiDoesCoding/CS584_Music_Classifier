import string, math, random, re, time
import sklearn, scipy
import numpy as np
import pandas as pd
from numpy import array
from scipy.special import softmax
from tensorflow import keras
from tensorflow.keras import layers
def test():
    data = np.array(
    [[7.18,	5.23,	0.249,	1.35,1.48,	0.531],
    [1.89,	0.761,	0.345,	2.3,	1.65,	0.0676],
    [0.528,	-0.0777,	-0.28,	0.686,	1.94,	0.881]]
    )
    data = data.reshape(3, 6, 1)
    data_y = np.array([0, 1, 2])

    model = keras.Sequential()

    model.add(layers.LSTM(64, activation= 'relu', input_shape=(6, 1)))
    #model.add(layers.LSTM(32, activation = 'relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    epochs = 100
    batch_size = 32

    history = model.fit(data, data_y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    test = np.array([[[7.17],	[5.22],	[0.241],	[1.35],	[1.23],	[0.511]],
                    [[1.89],	[0.761],	[0.345],	[2.3],	[1.65],	[0.0676]]])

    pred = model.predict(test)

    #test2 = np.array([[[1.89],	[0.761],	[0.345],	[2.3],	[1.65],	[0.0676]]])

    #pred2 = model.predict(test2)
    x = 10

def example():
    X = list()
    Y = list()
    X = [x+1 for x in range(20)]
    Y = np.array([y * 15 for y in X])

    print(X)
    print(Y)
    X = array(X).reshape(20, 1, 1)
    model = keras.Sequential()
    model.add(layers.LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    model.fit(X, Y, epochs=2000, validation_split=0.2, batch_size=5)

    test_input = array([30])
    test_input = test_input.reshape((1, 1, 1))
    test_output = model.predict(test_input, verbose=0)
    print(test_output)

if __name__ == "__main__":
    in_file = "echo_features_mfcc_mean.csv"

    data = pd.read_csv(in_file, header = [0, 1, 2])

    test()