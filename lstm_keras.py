import string, math, random, re, time
import sklearn, scipy
import numpy as np
import pandas as pd
from numpy import array
from scipy.special import softmax
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def test(data, data_y, test, test_y):
    data = data.reshape(len(data), len(data[0]), 1)
    test = test.reshape(len(test), len(test[0]), 1)
    
    test_y = keras.utils.to_categorical(test_y, max(data_y) + 1).astype('int32')
    data_y = keras.utils.to_categorical(data_y).astype('int32')


    model = keras.Sequential()
    model.add(layers.LSTM(128, input_shape=(20, 1), return_sequences=True))
    model.add(layers.LSTM(128))
    #model.add(layers.LSTM(32, activation = 'relu'))
    model.add(keras.layers.Dense(units=len(data_y[0]), activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])
    epochs = 100
    batch_size = 2000

    model.summary()
    genre_model = model.fit(data, data_y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    #test = np.array([[[7.17],	[5.22],	[0.241],	[1.35],	[1.23],	[0.511]],
    #                [[1.89],	[0.761],	[0.345],	[2.3],	[1.65],	[0.0676]],
    #                [[0],	[0],	[0],	[0],	[0],	[0]]])

    pred = model.predict(test)
    
    #test2 = np.array([[[1.89],	[0.761],	[0.345],	[2.3],	[1.65],	[0.0676]]])

    #pred2 = model.predict(test2)
    x = 10
    return

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
    in_file = "mfcc_mean_1genre.csv"
    in_genre = "V5/v5_genreLabels.csv"#"fma_metadata/tracks_genres.csv"
    in_genre_remapped = "V5/v6_genreLabels.csv"
    data_in = pd.read_csv(in_file)

    #genre_in = pd.read_csv(in_genre)
    #genre_in = genre_in.loc[:, :'genres']
    genre_in_test = pd.read_csv(in_genre_remapped)
    genre_in = genre_in_test.drop('genres', axis = 1)
    #genre_in = genre_in.drop('reassigned', axis = 1)

    data = np.array(
    [[7.18,	5.23,	0.249,	1.35,1.48,	0.531],
    [1.89,	0.761,	0.345,	2.3,	1.65,	0.0676],
    [0.528,	-0.0777,	-0.28,	0.686,	1.94,	0.881],
    [0,	0,	0,	0,	0,	0]]
    )
    
    tracks = genre_in['track_id'].tolist()
    #data_in = data_in.set_index('track_id')
    #track_id = data_in[0]
    data_in = data_in[data_in['track_id'].isin(tracks)]
    #genre_num = len(set(genre_in['genres'].tolist()))
    data_in = data_in.merge(genre_in, on=['track_id'])
    genre_in = data_in['reassigned'].to_numpy()
    data_in = data_in.drop('reassigned' , axis=1)
    data_in = data_in.set_index('track_id').to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(data_in, genre_in, test_size = .3) 
    #data_in = data_in.drop(0, axis = 1)
    #data_y = np.array([0, 1, 2, 1])
    test(x_train, y_train, x_test, y_test)