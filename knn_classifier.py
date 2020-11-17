from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity

tracks_genres = pd.read_csv('./fma_metadata/tracks_genres.csv')
echonest_edit = pd.read_csv('./fma_metadata/echonest_edit.csv')

echonest_edit2 = pd.read_csv('./fma_metadata/echonest_edit2.csv')

echonest_edit2 = echonest_edit2.drop('track_id', axis=1)

# print(tracks_genres.head)
# print(echonest_edit.head)

print("lol")
trackIDs = tracks_genres.loc[:, 'track_id']
trackIDs_index = trackIDs.values
trackIDs_list = trackIDs_index.tolist()

genres = tracks_genres.loc[:, 'genres']
genres_index = genres.values
print(trackIDs_index)
print(len(trackIDs_index))
print(len(genres_index))

# 30689 - 30723
print("lol2")
trainIDs = echonest_edit.loc[:, 'track_id']
trainIDs_index = trainIDs.values
trainIDs_list = trainIDs_index.tolist()
print(len(trainIDs_index))

g = []

for i in range(0, len(trainIDs_index)):
    x = trainIDs_index[i]
    index = trackIDs_list.index(str(x))
    y = genres[index]
    # print(y)
    g.append(y)

print(len(g))

# f = open("genres2.txt", "w")
# for i in range(len(g)):
#     f.write(str(trainIDs_index[i]) + "\t" + str(g[i]) + "\n")
# f.close()
#
# print("DONE")

t = []
for i in range(0, len(g)):
    x = g[i]
    str1 = x.replace(']', '').replace('[', '')
    y = str1.replace('"', '').split(",")
    t.append(y[0])
    # print(y[0])


X = echonest_edit2
y = t

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)  # split
KNN = KNeighborsClassifier(n_neighbors=50, metric='euclidean')
KNN.fit(X_train, y_train)  # KNN

knn_predict = KNN.predict(X_test)
cm = confusion_matrix(y_test, knn_predict)
accuracy = KNN.score(X_test, y_test)  # test accuracy
print(accuracy)

f = open("test_labels.txt", "w")
for i in range(len(knn_predict)):
    f.write(str(trainIDs_index[i]) + "\t" + str(knn_predict[i]) + "\n")
f.close()

print("DONE")