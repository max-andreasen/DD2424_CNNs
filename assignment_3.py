
'''
Assignment #3 in DD2424 Deep Learning in Data Science.
By Max Andreasen.

In this assignment the aim is to construct and implement a Convolutional Neural Network.
'''

import numpy as np
import matplotlib.pyplot as plt

class CNN:
    def __init__(self, X, Y, y, m=100, lr=0.01, lam=0, stride=2, W=None, B=None, n_f=None, n_p=None, filters=None, K=10):

        self.X = X                  # X is the dataset consisting of images, y labels.
        self.Y = Y
        self.y = y                  # Each image has 3 dimensions, so X is 4D.

        self.width = X.shape[0]
        self.height = X.shape[1]

        # self.depth = X.shape[2]
        self.n_images = X.shape[-1]

        # Network parameters.
        self.L = 2                  # We hardcode a 2 layer network for simplicity.
        self.m = m                  # Hidden layer dimensions (also referred to as 'd').
        self.K = K                  # The number of classes.
        self.MX = None              # Initialized to None. Calculated in the forward pass.

        # Network parameters
        self.lr = lr                # The initial learning rate of the model.
        self.lr_min = 0
        self.lr_max = 1
        self.lam = lam

        # Weights and parameters
        self.W = W if W is not None else [None] * 2
        self.B = B if B is not None else [None] * 2
        self.grads = {}

        # CNN parameters
        self.stride = stride        # Also referred to as 'f'.
        self.filters = filters      # A list with different filters (numpy arrays).
        self.n_f = n_f              # Number of filters.
        self.n_p = n_p              # Number of sub-patches to which the filter is applied.

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



    def forward_efficient(self, X_batch, return_testing=False):
        return


    def forward_pass(self, X_batch, return_testing=False):

        hs = []
        Ps = []
        x1s = []
        for j in range(X_batch.shape[-1]):
            X_img=X_batch[:,:,:,j]
            # h is a collection of all response maps.
            h = np.zeros((self.n_f * self.n_p, 1), dtype=X_batch.dtype)
            for i in range(self.n_f):
                H_i = np.maximum(0, self.convolve(X_img, self.filters[:, :, :, i]))     # Applies ReLu.
                assert H_i.shape == (32 // self.stride, 32 // self.stride), f"H_i is of wrong shape in forward pass: {H_i.shape}"
                start = i * self.n_p
                end = (i + 1) * self.n_p
                h[start:end, 0] = H_i.reshape(-1)  # Flatten H_i to shape (n_p,) to create vertically stacked vectors.

            assert h.shape == (self.n_f * self.n_p, 1), f"h is of wrong shape in forward pass: {h.shape}"

            x1 = np.maximum(0, self.W[0] @ h + self.B[0])   # Applies ReLu.
            x1s.append(x1)
            s = self.W[1] @ x1 + self.B[1]
            assert s.shape == (self.K, 1), f"S is of wrong size in forward pass: {s.shape}"
            P = self.softmax(s)
            Ps.append(P)
            hs.append(h)

        P = np.concatenate(Ps, axis=1)
        if return_testing:
            return {
                'P': P,
                'h': hs,
                'x1': x1s,
            }
        return P



    def backwards_pass(self):
        return



    def train(self):
        return


if __name__ == "__main__":
    cnn = CNN()