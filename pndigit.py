import pyneural
import numpy as np
from loader import MNIST
import pickle


def buildNet():
    net = pyneural.NeuralNet([784, 100, 10])
    ims, labels = getDataSet()
    net.train(ims, labels, 30, 10, 0.01, 0.0, 1.0)
    return net

def loadNet(fname):
    f = open(fname, "rb")
    data = pickle.load(f)
    params = data['params']
    net = pyneural.NeuralNet([784, 100, 10])
    net.set_params(params)
    return net

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
        vector = [0.1] * 10 #list of 10 zeros
        vector[label] = 0.9
        vectorized.append(vector)

    return vectorized

