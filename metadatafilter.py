import string, math, random, re, time
import sklearn, scipy
import numpy as np
import pandas as pd
import tensorflow

def main():
    input_file = "fma_metadata/features_2.csv"
    input_echo= "fma_metadata/echonest_trim.csv"

    echo =  pd.read_csv(input_echo, encoding='utf-8', sep=',', header=None)[0].tolist()

    features = pd.read_csv(input_file, encoding='utf-8', sep=',', header=None, chunksize= 10000)
    
    for chunk in features:
        try:
            data_filter = data_filter.append(chunk[chunk[0].isin(echo)])
        except NameError:
            data_filter = chunk[chunk[0].isin(echo)]
    
    data_filter.to_csv('echo_featers.csv', index = False, header = False)

    return

if __name__== '__main__':
    main()