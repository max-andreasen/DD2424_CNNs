
'''
Assignment #3 in DD2424 Deep Learning in Data Science.
By Max Andreasen.

In this assignment the aim is to construct and implement a Convolutional Neural Network.
'''

import numpy as np
import matplotlib.pyplot as plt

class CNN:
    def __init__(self, X=None, y=None, m=100, lr=0.01, lam=0, stride=2):

        self.X = X                  # X is the dataset consisting of images, y labels.
        self.y = y                  # Each image has 3 dimensions, so X is 4D.

        self.width = X.shape[0]
        self.height = X.shape[1]

        # self.depth = X.shape[2]
        self.n_images = X.shape[-1]

        # Network parameters.
        self.L = 2                  # We hardcode a 2 layer network for simplicity.
        self.m = m                  # Hidden layer dimensions (also referred to as 'd').
        self.K = None               # The number of classes.

        # Network parameters
        self.lr = lr                # The initial learning rate of the model.
        self.lr_min = 0
        self.lr_max = 1
        self.lam = lam

        # Weights and parameters
        self.W = [None] * 2
        self.B = [None] * 2
        self.grads = {}

        # CNN parameters
        self.stride = stride        # Also referred to as 'f'.
        self.filters = None         # A list with different filters (numpy arrays).
        self.n_f = None             # Number of filters.
        self.n_p = None             # Number of sub-patches to which the filter is applied.

        # The model will hold the data.
        self.data = None
        self.labels = None



    # -----------------------------------------------
    #  Helper functions
    # -----------------------------------------------

    def process_data(self, file):
        return

    def softmax(self, s):
        e_x = np.exp(s - np.max(s))
        return e_x / np.sum(e_x)

    def cross_entropy_loss(self):
        return

    def convolve(self, X, conv_filter, stride=None):
        if stride is None:
            stride = self.stride
        f = conv_filter.shape[0]                # Enough with width since filter is a square.
        H_out = (X.shape[0] - f) // stride + 1  # Height of the new image.
        W_out = (X.shape[1] - f) // stride + 1  # Width of the new image.
        conv_out = np.zeros( (H_out, W_out), dtype=X.dtype)   # The outputted image shape.

        assert X.shape[2] == conv_filter.shape[2], f"Depth of filter and image is not equal"

        # Loops through and creates each sub-patch.
        for i in range(H_out):
            for j in range(W_out):
                patch = X[i*stride : i*stride+f,j*stride : j*stride+f, :]
                conv_out[i, j] = np.sum(patch * conv_filter)
        return conv_out



    # -----------------------------------------------
    #  Initializations
    # -----------------------------------------------
    def init_network(self):
        self.filters = [np.zeros((self.stride, self.stride, 3)) * 5] # shape (f, f, 3)
        self.n_f = len(self.filters)
        self.n_p = (self.X.shape[0] / self.stride) ** 2
        self.K = len(set(self.y))

        d0 = self.n_f * self.n_p        # The shape after flattening the feature vector (after convolution).

        self.W[0] = np.zeros( (self.m, d0) )
        self.W[1] = np.zeros( (self.K, self.m) )

        assert self.W[0].shape == (self.m, d0), f"W[0] initialized to the wrong shape of: {self.W.shape}"
        assert self.W[1].shape == (self.K, self.m)



    # -----------------------------------------------
    #  Network computations
    # -----------------------------------------------
    def make_prediction(self):
        return


    def forward_pass(self, X):
        # h is a collection of all response maps.
        h = np.zeros((X.shape[0] // self.stride, X.shape[1] // self.stride, self.n_f))

        for i in range(self.n_f):
            H_i = np.max(0, self.convolve(X, self.filters[i]))     # shape (32/f, 32/f, 1)
            H_i = H_i.reshape(-1, 1)                    # H is also denotes S sometimes.
            h[:, i] = H_i

        assert h.shape == (self.n_f, self.n_p, 1), f"h is of wrong shape in forward pass: {h.shape}"

        x1 = np.max(0, self.W[0] @ h + self.B[0])   # Applies ReLu.
        s = self.W[1] @ x1 + self.B[1]
        assert s.shape == (self.K, 1), f"S is of wrong size in forward pass: {s.shape}"
        P = self.softmax(s)

        return P



    def backwards_pass(self):
        return



    def train(self):
        return


if __name__ == "__main__":
    cnn = CNN()