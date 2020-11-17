import string, math, random, re, time
import sklearn, scipy
import numpy as np
import pandas as pd
from scipy.special import softmax

#Custom RNN algorithm based on tutorials. 
def rnn_cell(xt, ht, parameters):
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    #hn defines what will be passed onto the next iteration/rnn cell
    hn = np.tanh(np.dot(Wax, xt) + np.dot(Waa, ht) + ba)
    yt_pred = softmax(np.dot(Wya, hn) + by) #Softmax is an activation function that represents probably distributions of list of potential outcomes. This computes losses when training.
    
    cache = (hn, ht, parameters)
    return hn, yt_pred, cache

#Passing to the next iteration/cell
def pass_to_next(xt, ht, parameters):
    history = []

    #Initializing states comprising of zeros.
    _, m, T_x = xt.shape
    n_y, n_a = parameters["Wya"].shape
    y_pred = np.zeroes((n_y, m, T_x))
    a = np.zeros((n_a, m, T_x))

    a_next = ht
    #Start looping over the sequence
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell(xt[:, :, t],  a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
    history.append(cache)
    history = (history, xt)
    return a, y_pred, history

def main():

    return