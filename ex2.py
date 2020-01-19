import sys
from random import shuffle
import numpy as np


def minMax(trainging_set):
    """
    :param trainging_set: our data set.
    :return: normalized data set
    """
    return (trainging_set - trainging_set.min()) / (trainging_set.max() - trainging_set.min())


def accuracy_on_dataset(dataset, w):
    good = bad = 0.0
    for features, label in dataset:
        y_hat = np.argmax(np.dot(w, features))
        if y_hat == label:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def feats_to_vec(feats):
    m = [0, 0, 1]
    F = [0, 1, 0]
    I = [1, 0, 0]
    n_feats = np.zeros((np.size(feats, 0), np.size(feats, 1) + 3))
    for i, f in enumerate(feats):
        if f[0] == 'M':
            f[0] = 1
            n_feats[i] = np.concatenate((f, m), axis=None)
        if f[0] == 'F':
            f[0] = 1
            n_feats[i] = np.concatenate((f, F), axis=None)
        if f[0] == 'I':
            f[0] = 1
            n_feats[i] = np.concatenate((f, I), axis=None)

    n_feats = n_feats.astype(float)
    return n_feats


def train_perceptron(train_x, train_y, dev=0):
    w = np.zeros((np.size(np.unique(train_y)), np.size(train_x, 1)))
    etaPER = 0.2
    epochs = 60
    for e in range(epochs):
        zipped = list(zip(train_x, train_y))
        shuffle(zipped)
        if e % 6 == 0:
            etaPER *= 0.5
        for x1, y in zipped:
            y_hat = np.argmax(np.dot(w, x1))
            if y != y_hat:
                w[y, :] += etaPER * x1
                w[y_hat, :] -= etaPER * x1
                etaPER *= (1 - e / (2 * epochs))
    return w


def train_svm(train_x, train_y, dev=0):
    w = np.zeros((np.size(np.unique(train_y)), np.size(train_x, 1)))
    etaSVM = 0.3
    LAMBDA = 0.1
    epochs = 100
    for e in range(epochs):
        zipped = list(zip(train_x, train_y))
        shuffle(zipped)
        if e % 10 == 0:
            etaSVM *= 0.5
        for x1, y in zipped:
            y_hat = np.argmax(np.dot(w, x1))
            if y != y_hat:
                mult = (1 - etaSVM * LAMBDA)
                w[y, :] = np.multiply(w[y, :], mult) + etaSVM * x1
                w[y_hat, :] = np.multiply(w[y_hat, :], mult) - etaSVM * x1
                w[np.size(w, 0) - y - y_hat, :] = np.multiply(w[3 - y - y_hat, :], mult)
                etaSVM *= (1 - e / epochs)
            else:
                for i in range(3):
                    if (i != y_hat):
                        w[i] *= (1 - etaSVM * LAMBDA)
    return w


def train_pa(train_x, train_y, dev=0):
    w = np.zeros((np.size(np.unique(train_y)), np.size(train_x, 1)))
    epochs = 80
    for e in range(epochs):
        zipped = list(zip(train_x, train_y))
        for x1, y in zipped:
            y_hat = np.argmax(np.dot(w, x1))
            if y != y_hat:
                tao = max(0, 1 - (np.dot(w[y], x1) + np.dot(w[y_hat], x1))) \
                      / 2 * (np.linalg.norm(x1) ** 2)
                value = tao * x1
                w[y, :] += value
                w[y_hat, :] -= value
    return w


def test(test_set, w1, w2, w3):
    for x in test_set:
        y1 = np.argmax(np.dot(w1, x))
        y2 = np.argmax(np.dot(w2, x))
        y3 = np.argmax(np.dot(w3, x))
        print(f"perceptron: {y1}, svm: {y2}, pa: {y3}")

def main():
    training_set = np.loadtxt(sys.argv[1], dtype='str', delimiter=',')
    training_labels = np.loadtxt(sys.argv[2], dtype='int')
    test_x = np.loadtxt(sys.argv[3], dtype='str', delimiter=',')
    test_x = feats_to_vec(test_x)
    training_set = feats_to_vec(training_set)
    training_set = minMax(training_set)
    w_per = train_perceptron(training_set, training_labels)
    w_svm = train_svm(training_set, training_labels)
    w_pa = train_pa(training_set, training_labels)
    test(test_x, w_per, w_svm, w_pa)