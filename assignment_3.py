
'''
Assignment #3 in DD2424 Deep Learning in Data Science.
By Max Andreasen.

In this assignment the aim is to construct and implement a Convolutional Neural Network.
'''

import numpy as np
import matplotlib.pyplot as plt
import time


class CNN:
    '''
    The class is currently built around handling square images,
    where WIDTH == HEIGHT.
    '''

    def __init__(self, X, Y, y, m=100, lr=0.01, lam=0, stride=2, n_batches=10,
                 W=None, B=None, filters=None, n_filters=2, K=None,
                 init_MX=True):

        self.X = X                  # X is the dataset consisting of images, y labels.
        self.Y = Y
        self.y = y                  # Each image has 3 dimensions, so X is 4D.

        self.width = X.shape[0]
        self.height = X.shape[1]

        # self.depth = X.shape[2]
        self.n_images = X.shape[-1]

        # Network parameters.
        self.L = 2                          # We hardcode a 2 layer network for simplicity.
        self.m = m                          # Hidden layer dimensions (also referred to as 'd').
        self.K = K if K is not None else len(set(y))   # The number of classes.

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
        self.stride = stride            # Also referred to as 'f'.
        self.filters = filters if filters is not None else np.zeros((self.stride, self.stride, 3, n_filters)) # shape (f, f, 3)
        self.n_f = self.filters.shape[-1]                   # Could also use 'n_filters', but unclear if 'filters' is given.
        self.n_p = (self.width // self.stride) ** 2     # Number of sub-patches to which the filter is applied.
        self.filters_flat = self.filters.reshape( (self.stride * self.stride * 3, self.n_f), order='C')

        # Initializes variables
        if init_MX:
            self.MX = self.construct_MX(self.X)
        else:
            self.MX = None



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
        '''
        Used for the 'ground truth' forward pass.
        '''
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

    def convolve_efficient(self, X_batch):
        if self.MX is None:
            self.MX = self.construct_MX(X_batch)
        assert self.filters_flat.shape == (self.stride * self.stride * 3, self.n_f), f"Filters are of wrong shape: {self.filters_flat.shape}"
        n_img_batch = X_batch.shape[-1]
        conv_outputs_mat = np.zeros((self.n_p, self.n_f, n_img_batch))
        for i in range(n_img_batch):
            conv_outputs_mat[:, :, i] = np.matmul(self.MX[:, :, i], self.filters_flat)
        conv_outputs_mat = np.einsum('ijn, jl->iln', self.MX, self.filters_flat, optimize=True)
        return conv_outputs_mat

    def construct_MX(self, X):
        '''
        THIS FUNCTION ASSUMES A SQUARE IMAGE (WIDTH == HEIGHT).
        :param X:
        :return:
        '''
        print(f"Calculating MX...")
        start = time.time()
        MX = np.zeros((self.n_p, self.stride * self.stride * 3, self.n_images))
        for i in range(self.n_images):
            X_img = X[:, :, :, i]
            patch_id = 0
            for r in range(int(np.sqrt(self.n_p))):
                for col in range(int(np.sqrt(self.n_p))):
                    X_patch = X_img[ r*self.stride:(r+1)*self.stride, col*self.stride:(col+1)*self.stride, :]
                    MX[patch_id, :, i] = X_patch.reshape((1, self.stride * self.stride * 3), order='C')
                    patch_id += 1
        end = time.time()
        print(f"MX calculated in {end - start} seconds.")
        return MX



    # -----------------------------------------------
    #  Network computations
    # -----------------------------------------------
    def make_prediction(self):
        return



    def forward_efficient(self, X_batch, return_testing=False):
        assert self.filters_flat.shape == (
        self.stride * self.stride * 3, self.n_f), f"Filters are of wrong shape: {self.filters_flat.shape}"

        # NOTE: Also fills MX if MX is empty!
        conv_outputs_mat = self.convolve_efficient(X_batch)



    def forward_pass(self, X_batch, return_testing=False):
        hs = []
        Ps = []
        x1s = []
        for j in range(X_batch.shape[-1]):
            X_img = X_batch[:,:,:,j]
            # h is a collection of all response maps.
            h = np.zeros((self.n_f * self.n_p, 1), dtype=X_batch.dtype)
            H_all = np.zeros((self.width // self.stride, self.height // self.stride, self.n_f))
            for i in range(self.n_f):
                H_i = np.maximum(0, self.convolve(X_img, self.filters[:, :, :, i]))     # Applies ReLu.
                assert H_i.shape == (self.width // self.stride, self.height // self.stride), f"H_i is of wrong shape in forward pass: {H_i.shape}"
                H_all[:, :, i] = H_i.reshape((self.width // self.stride, self.height // self.stride), order='C')

            h[0::2] = H_all[:, :, 0].reshape(-1, 1)
            h[1::2] = H_all[:, :, 1].reshape(-1, 1)
            assert h.shape == (self.n_f * self.n_p, 1), f"h is of wrong shape in forward pass: {h.shape}"
            x1 = np.maximum(0, self.W[0] @ h + self.B[0])   # Applies ReLu.
            x1s.append(x1)
            s = self.W[1] @ x1 + self.B[1]
            assert s.shape == (self.K, 1), f"S is of wrong size in forward pass: {s.shape}"
            P = self.softmax(s)
            Ps.append(P)
            hs.append(h)

        P = np.concatenate(Ps, axis=1)
        hs = np.concatenate(hs, axis=1)
        X1 = np.expand_dims(np.concatenate(x1s, axis=1), axis=0)
        if return_testing:
            return {
                'P': P,
                'h': hs,
                'x1': X1,
            }
        return P



    def backwards_pass(self):
        return



    def train(self):
        return


if __name__ == "__main__":
    cnn = CNN()