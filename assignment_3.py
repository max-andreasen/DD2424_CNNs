
'''
Assignment #3 in DD2424 Deep Learning in Data Science.
By Max Andreasen.

In this assignment the aim is to construct and implement a Convolutional Neural Network.
'''

import numpy as np
import matplotlib.pyplot as plt

class CNN:
    def __init__(self, m=100, lr=0.01, lam=0, stride=2, layers=2):

        # Network parameters.
        self.L = layers
        self.m = m

        self.lr = lr
        self.lr_min = 0
        self.lr_max = 1
        self.lam = lam

        self.W = [None] * self.L
        self.B = [None] * self.L
        self.grads = {}

        # CNN parameters
        self.stride = stride
        self.filter = None

        # The model will hold the data.
        self.data = None
        self.labels = None

    def process_data(self, file):
        return

    def init_network(self):
        self.filter = np.zeros((self.stride, self.stride, 3)) # shape (f, f, 3)

    def softmax(self, s):
        return

    def cross_entropy_loss(self):
        return

    def forward_pass(self, X):
        n = 10
        h = np.zeros((X.shape[0] / self.stride, X.shape[1] / self.stride, n))
        H = np.max(0, X * 0) # shape (32/f, 32/f, 1)

