import string, math, random, re, time
import sklearn, scipy
import numpy as np
import pandas as pd
from scipy.special import softmax
from tensorflow import keras
from tensorflow.keras import layers


data = np.array(
[[[7.18,	5.23,	0.249,	1.35,	1.48,	0.531]],
[[1.89,	0.761,	0.345,	2.3,	1.65,	0.0676]],
[[0.528,	-0.0777,	-0.28,	0.686,	1.94,	0.881]]]
)

data_y = np.array([0, 1, 2])

model = keras.Sequential()

model.add(layers.LSTM(50, input_shape=(1, 6)))
model.add(keras.layers.Dense(3, activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
epochs = 5
batch_size = 32

history = model.fit(data, data_y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

test = np.array([[[7.17,	5.22,	0.241,	1.35,	1.23,	0.511]]])

pred = model.predict(test)

x = 10
