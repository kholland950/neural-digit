import pybrain
from pybrain.datasets import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from loader import MNIST
import pickle


def buildNet():
    net = buildNetwork(784, 30, 10, bias=True)
    ds = getDataSet()
    trainer = BackpropTrainer(net, ds, learningrate=0.05,verbose=True)
    
    trainer.trainEpochs(epochs=20)

    return net

def loadNet(fname):
    f = open(fname, "rb")
    return pickle.load(f)

def writeNet(fname, net):
    f = open(fname, "wb")
    pickle.dump(net, f)

def getDataSet():
    mn = MNIST(".") #dir of files
    images, labels = mn.load_training()
    ds = SupervisedDataSet(784, 10)

    images = normalize_images(images)
    labels = vectorize_labels(labels)

    for i in range(len(images)):
        ds.addSample(images[i], labels[i])

    return ds

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

mn = MNIST(".")
ims = None
labels = None
ims_test = None
labels_test = None


def test_training(net, n=1000):
    global ims, labels
    if ims == None or labels == None:
        ims, labels = mn.load_training()
        ims = normalize_images(ims)
    return test_on_set(net, ims, labels, n)

def test_testing(net, n=1000):
    global ims_test, labels_test
    if ims_test == None or labels_test == None:
        ims_test, labels_test = mn.load_testing()
        ims_test = normalize_images(ims_test)
    return test_on_set(net, ims_test, labels_test, n)

def test_on_set(net, n_ims, labels, n=1000):
    correct = 0

    for i in range(n):
        result = net.activate(n_ims[i]).tolist()
        max_j = 0
        for j in range(10):
            if result[j] > result[max_j]:
                max_j = j
        
        if max_j == labels[i]:
            correct += 1
        
    return str(correct/float(n) * 100) + "%"

