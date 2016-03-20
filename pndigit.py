import pyneural
import numpy as np
from loader import MNIST
import pickle

n_input = 784
n_hidden = 200
n_output = 10
epochs = 50

def buildNet():
    net = pyneural.NeuralNet([n_input, n_hidden, n_output])
    ims, labels = getDataSet()
    net.train(ims, labels, epochs, 10, 0.01, 0.0, 1.0)
    return net

def loadNet(fname):
    f = open(fname, "rb")
    data = pickle.load(f)
    params = data['params']
    newNet = pyneural.NeuralNet([n_input, n_hidden, n_output])
    newNet.set_params(params)
    return newNet

def writeNet(fname, net):
    f = open(fname, "wb")
    data = {}
    data['params'] = net.get_params()
    pickle.dump(data, f)

def getDataSet():
    mn = MNIST(".") #dir of files
    images, labels = mn.load_training()

    images = normalize_images(images)
    labels = vectorize_labels(labels)
    return np.array(images), np.array(labels)

def normalize_images(imgs):
    normalized = []
    def norm(x): return x/255.0
    for img in imgs:
        normalized.append(list(map(norm, img)))

    return normalized

def vectorize_labels(labels):
    vectorized = []
    
    for label in labels:
        vector = [0.01] * 10 #list of 10 zeros
        vector[label] = 0.99
        vectorized.append(vector)

    return vectorized

