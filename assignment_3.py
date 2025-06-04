
'''
Assignment #3 in DD2424 Deep Learning in Data Science.
By Max Andreasen.

In this assignment the aim is to construct and implement a Convolutional Neural Network.
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from tenacity import retry_if_result


class CNN:
    '''
    The class is currently built around handling square images,
    where WIDTH == HEIGHT.
    '''

    def __init__(self, X, Y, y, m=100, lr=0.01, lam=0, stride=2, n_batches=1,
                 W=None, B=None, filters=None, n_filters=2, n_hidden=10,
                 init_MX=True):

        self.X = X                  # X is the dataset consisting of images, y labels.
        self.Y = Y
        self.y = y                  # Each image has 3 dimensions, so X is 4D.

        self.width = X.shape[0]
        self.height = X.shape[1]

        # self.depth = X.shape[2]
        self.n_images = X.shape[-1]
        self.batch_size = self.n_images // n_batches

        # Network parameters.
        self.L = 2                          # I hardcode a 2 layer network for simplicity.
        self.m = m                          # Hidden layer dimensions (also referred to as 'd').
        self.K = self.Y.shape[0]            # The number of classes.
        self.hidden_dim = n_hidden          # The number of neuron in the hidden layer.

        # Network parameters
        self.lr = lr                        # The initial learning rate of the model.
        self.lr_min = 0
        self.lr_max = 1
        self.lam = lam

        # Weights and parameters
        self.grads = {}

        # CNN parameters
        self.stride = stride            # Also referred to as 'f'.
        self.filters = filters if filters is not None else (
            self.he_init((self.stride, self.stride, 3, n_filters))) # shape (f, f, 3, n_f)
        self.n_f = self.filters.shape[-1]                   # Could also use 'n_filters', but unclear if 'filters' is given.
        self.n_p = (self.width // self.stride) ** 2         # Number of sub-patches to which the filter is applied.
        self.filters_flat = self.filters.reshape( (self.stride * self.stride * 3, self.n_f), order='C')

        # INITIALIZATIONS
        if init_MX:
            self.MX = self.construct_MX(self.X)
        else:
            self.MX = None
        '''
        These parameters are HARDCODED for 2 LAYERS. 
        Also, W1.shape[1] is passed as an input variable. 
        TODO: In the future, this value can be calculated based on parameters of the netwrok. 
        '''
        self.W = W if W is not None else (
            [self.he_init((self.hidden_dim, 128)), self.he_init((self.K, self.hidden_dim)), ])
        self.B = B if B is not None else(
            [np.zeros((self.hidden_dim, 1)), np.zeros( (self.K, 1))] )


    # -----------------------------------------------
    #  Helper functions
    # -----------------------------------------------
    def he_init(self, shape):
        if len(shape) == 2:             # The weights (W1 and W2).
            fan_in = shape[1]
        elif len(shape) == 4:           # The filter.
            fan_in = shape[0] * shape[1] * shape[2] # Considering shape[3] is n_f.
        else:
            raise ValueError('Invalid shape in he_init.')
        std = np.sqrt(2.0 / fan_in)     # According to the 'He initialization' equation.
        return  np.random.randn(*shape) * std

    def softmax(self, s):
        e_x = np.exp(s - np.max(s, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

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



    def forward_efficient(self, X_batch, return_params=False):
        assert self.filters_flat.shape == (self.stride * self.stride * 3, self.n_f), f"Filters are of wrong shape: {self.filters_flat.shape}"

        # NOTE: 'convolve_efficient' also fills MX if MX is empty!
        conv_outputs_mat = self.convolve_efficient(X_batch)

        # Corresponds to h in the 'regular' forward pass.
        conv_flat = np.fmax(conv_outputs_mat.reshape( (self.n_p*self.n_f, self.batch_size), order='C' ), 0 )

        # Applies the connected layers and activates through ReLu.
        x1 = np.maximum(0, self.W[0] @ conv_flat + self.B[0])
        s = (self.W[1] @ x1 + self.B[1])
        assert s.shape == (self.K, self.batch_size), f"S is of wrong size in forward pass: {s.shape}"
        P = self.softmax(s)

        if return_params:
            return {
                'P': P,
                'h': conv_flat,
                'x1': np.expand_dims(x1, axis=0),
            }
        return P


    def forward_pass(self, X_batch, return_params=False):
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
        if return_params:
            return {
                'P': P,
                'h': hs,
                'x1': X1,
            }
        return P


    def backwards_pass(self, X_batch, Y_batch):
        outputs = self.forward_efficient(X_batch, return_params=True)
        P = outputs['P']
        x1 = outputs['x1'].squeeze(0)
        h = outputs['h']

        G = -(Y_batch-P)
        grad_W2 = (1/self.batch_size) * G @ x1.T

        G = self.W[1].T @ G
        G = G * (x1 > 0).astype(int)

        grad_W1 = (1 / self.batch_size) * G @ h.T

        G_batch = self.W[0].T @ G
        G_batch = G_batch * (h > 0).astype(int)

        GG = G_batch.reshape( (self.n_p, self.n_f, self.batch_size), order='C' )

        MXt = np.transpose(self.MX, (1, 0, 2))
        grad_Fs_flat = np.einsum('ijn, jln ->il', MXt, GG, optimize=True) / self.batch_size

        return {
            'grad_Fs_flat': grad_Fs_flat,
            'grad_W1': grad_W1,
            'grad_W2': grad_W2
        }


    def train(self):
        return
# -----------------------------------------------
    # END OF CLASS METHODS




# TODO: Re-write this function into a cleaner version that can handle ALL batches as well.
def read_data(filename):
    # Loads a batch of training data.
    try:
        cifar_dir = './Datasets/cifar-10-batches-py/'
        with open(cifar_dir + filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
    except:
        cifar_dir = '/Users/maxandreasen/GitHub/DD2424_CNNs/Datasets/cifar-10-batches-py'
        with open(cifar_dir + filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

    # Extract the image data and cast to float from the dict dictionary
    y = np.array(dict[b'labels'])
    k = len(set(y))
    Y = np.zeros((k, len(y)), dtype=np.float64)
    for i in range(Y.shape[1]):
        label = y[i]
        Y[label, i] = 1.0

    X = dict[b'data'].astype(np.float64) / 255.0  # RGB pixel values max at 255, making entries between 0 and 1.
    X = X.transpose()

    return X, Y, y

def preprocess_data(X_train, X_val, X_test):
    # Normalizes the data.
    mean = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    std = np.std(X_train, axis=1).reshape(X_train.shape[0], 1)

    # Normalize
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test


if __name__ == "__main__":
    X, Y, y = read_data('data_batch_1')

