import operator
import os
import pickle
import random
import sys
from collections import defaultdict
from random import shuffle
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
import classic as cl

directory = "../Music-Genre-Classification/Data/genres_original"
i = 0
dataset = []
trainingSet = []
testSet = []
predictions = []

def euclideanDistance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = euclideanDistance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return 1.0 * correct / len(testSet)


# f = open("myd.dat", 'wb')
data = []
for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory + "/" + folder):
        try:
            (rate, sig) = wav.read(directory + "/" + folder + "/" + file)
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0).reshape(1, 13)
            feature = np.concatenate((mean_matrix.reshape(-1), covariance.reshape(-1)), axis=0)
            # feature = (mean_matrix, covariance, i)
            data.append([feature, folder])
            # pickle.dump(feature, f)
        except Exception as e:
            print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)
# f.close()
shuffle(data)
file_path = 'outputclass3_10.txt'
sys.stdout = open(file_path, "w")
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
ss= ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
x, y = zip(*data)
for train_idx, val_idx in ss.split(x, y):
    X_tr = np.array(x)[train_idx]
    y_tr = np.array(y)[train_idx]

    X_val = np.array(x)[val_idx]
    y_val = np.array(y)[val_idx]
    cl.main(X_tr, y_tr, X_val, y_val)
