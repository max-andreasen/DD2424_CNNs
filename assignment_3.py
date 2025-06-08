
'''
Assignment #3 in DD2424 Deep Learning in Data Science.
By Max Andreasen.

In this assignment the aim is to construct and implement a Convolutional Neural Network.
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
from pathlib import Path
from tqdm import tqdm
from torch_gradient_computations import ComputeGradsWithTorch


def construct_MX(X, stride, n_p):
    '''
    THIS FUNCTION ASSUMES A SQUARE IMAGE (WIDTH == HEIGHT).
    :param X:
    :return:
    '''
    # print(f"Calculating MX...")
    N = X.shape[-1]
    start = time.time()
    MX = np.zeros((n_p, stride * stride * 3, N))
    for i in range(N):
        X_img = X[:, :, :, i]
        patch_id = 0
        for r in range(int(np.sqrt(n_p))):
            for col in range(int(np.sqrt(n_p))):
                X_patch = X_img[r * stride:(r + 1) * stride, col * stride:(col + 1) * stride, :]
                MX[patch_id, :, i] = X_patch.reshape((1, stride * stride * 3), order='C')
                patch_id += 1
    end = time.time()
    # print(f"MX calculated in {end - start} seconds.")
    return MX

class CNN:
    '''
    The class is currently built around handling square images,
    where WIDTH == HEIGHT.
    It is also HARDCODED for 2 LAYERS.
    '''

    def __init__(self, X, Y, y, lr=0.01, lr_min=0.0005, lr_max=0.01, lam=0.003,
                 stride=2, n_batches=1, W=None, B=None, filters=None, n_filters=2,
                 step_size=800, n_hidden=10, MX=None):

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        self.X = X                  # X is the dataset consisting of images, y labels.
        self.Y = Y
        self.y = y                  # Each image has 3 dimensions, so X is 4D.

        self.width = X.shape[0]
        self.height = X.shape[1]
        if len(X.shape) > 3:        # Only has a depth in the scenario with shape (W, H, depth, N).
            self.depth = X.shape[2]
        else:
            self.depth = 1

        self.n_images = X.shape[-1]
        self.n_batches = n_batches
        self.batch_size = self.n_images // self.n_batches

        # Network parameters.
        self.K = self.Y.shape[0]            # The number of classes.
        self.hidden_dim = n_hidden          # Hidden layer dimensions (also referred to as 'd').

        # Network parameters
        self.lr = lr                        # The initial learning rate of the model.
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lam = lam                      # Regularization term 'lambda'.
        self.step_size = step_size          # The step size used for cyclic learning rates.
        self.base_step_size = step_size

        # Weights and parameters
        self.grads = {}                     # Dict with elements 'W1', 'W2', 'b1', 'b2' and 'Fs'.

        # CNN parameters
        self.stride = stride                # Also referred to as 'f'.
        self.filters = filters if filters is not None else (
            self.he_init((self.stride, self.stride, self.depth, n_filters)))     # shape (f, f, 3, n_f)
        self.n_f = self.filters.shape[-1]               # Could also use 'n_filters', but unclear if 'filters' is given.
        self.n_p = (self.width // self.stride) ** 2     # Number of sub-patches to which the filter is applied.
        self.filters_flat = self.filters.reshape( (self.stride * self.stride * self.depth, self.n_f), order='C')

        # INITIALIZATIONS
        self.MX = MX
        '''
        These parameters are HARDCODED for 2 LAYERS. 
        Also, W1.shape[1] is passed as an input variable. 
        TODO: In the future, this value can be calculated based on parameters of the netwrok. 
        '''
        self.W = W if W is not None else (
            [self.he_init((self.hidden_dim, self.n_f*self.n_p)), self.he_init((self.K, self.hidden_dim)), ])
        self.b = B if B is not None else(
            [np.zeros((self.hidden_dim, 1)), np.zeros( (self.K, 1))] )
        self.b_conv = np.zeros((self.n_f, 1))


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

    def loss(self, Y, P):
        '''
        Computes the cross entropy loss for the one-hot encoded label vector,
        and the probability distribution P which is calculated by the network.
        '''
        eps = 1e-15
        l_cross = -np.sum( Y * np.log(P + eps)) / Y.shape[1] # Eq. 6.
        return l_cross

    def cost(self, Y, P):
        '''
        Calculates the cost function (J), which in this assignment is mainly used for plotting purposes.
        '''
        l_cross = self.loss(Y, P)
        reg_term = np.sum(self.W[0]**2) + np.sum(self.W[1]**2)
        return l_cross + ( self.lam * reg_term)

    def accuracy(self, y, P):
        pred_labels = np.argmax(P, axis=0)
        comparison_vec = (pred_labels == y).astype(int)
        acc = np.mean(comparison_vec)
        return acc

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
        #print(f"Calculating MX...")
        N = X.shape[-1]
        start = time.time()
        MX = np.zeros((self.n_p, self.stride * self.stride * 3, N))
        for i in range(N):
            X_img = X[:, :, :, i]
            patch_id = 0
            for r in range(int(np.sqrt(self.n_p))):
                for col in range(int(np.sqrt(self.n_p))):
                    X_patch = X_img[ r*self.stride:(r+1)*self.stride, col*self.stride:(col+1)*self.stride, :]
                    MX[patch_id, :, i] = X_patch.reshape((1, self.stride * self.stride * 3), order='C')
                    patch_id += 1
        end = time.time()
        #print(f"MX calculated in {end - start} seconds.")
        return MX

    def compare_grads_with_torch(self, X, y, grads):
        network_params = {
            'W': [self.W[0], self.W[1]],
            'b': [self.b[0], self.b[1]],
            'Fs': np.array(self.filters),
            'stride': self.stride,
            'MX': self.MX,
        }
        torch_grads = ComputeGradsWithTorch(X, y, network_params)
        is_w_grads = np.allclose(torch_grads['W'][0], grads['grad_W1']) and np.allclose(torch_grads['W'][1], grads['grad_W2'])
        is_Fs_grad = np.allclose(torch_grads['b'][0], grads['grad_b1']) and np.allclose(torch_grads['b'][1], grads['grad_b2'])
        if is_w_grads and is_Fs_grad:
            print("Grads are OK.")




    # -----------------------------------------------
    #  Network computations
    # -----------------------------------------------
    def make_prediction(self):
        return


    def forward(self, X_batch, return_params=False):
        assert self.filters_flat.shape == (self.stride * self.stride * 3, self.n_f), f"Filters are of wrong shape: {self.filters_flat.shape}"

        # NOTE: 'convolve_efficient' also fills MX if MX is empty!
        conv_outputs_mat = self.convolve_efficient(X_batch)

        # Corresponds to h in the 'regular' forward pass.
        conv_flat = np.fmax(conv_outputs_mat.reshape( (self.n_p*self.n_f, self.batch_size), order='C' ), 0 )

        # Applies the connected layers and activates through ReLu.
        x1 = np.maximum(0, self.W[0] @ conv_flat + self.b[0])
        s = (self.W[1] @ x1 + self.b[1])
        assert s.shape == (self.K, self.batch_size), f"S is of wrong size in forward pass: {s.shape}"
        P = self.softmax(s)

        if return_params:
            return {
                'P': P,
                'h': conv_flat,
                'x1': np.expand_dims(x1, axis=0),
            }
        return P


    def forward_pass_legacy(self, X_batch, return_params=False):
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
            x1 = np.maximum(0, self.W[0] @ h + self.b[0])   # Applies ReLu.
            x1s.append(x1)
            s = self.W[1] @ x1 + self.b[1]
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


    def update_params(self, grads):
        self.W[0] -= self.lr * grads['grad_W1']
        self.W[1] -= self.lr * grads['grad_W2']
        self.b[0] -= self.lr * grads['grad_b1']
        self.b[1] -= self.lr * grads['grad_b2']
        self.filters_flat -= self.lr * grads['grad_Fs_flat']
        self.b_conv -= self.lr * grads['grad_b_conv']



    def backwards(self, X_batch, Y_batch):
        outputs = self.forward(X_batch, return_params=True)
        P = outputs['P']
        x1 = outputs['x1'].squeeze(0)
        h = outputs['h']

        # Adding L2 regularization.
        reg_W1 = (2 * self.lam * self.W[0])
        reg_W2 = (2 * self.lam * self.W[1])
        reg_Fs = (2 * self.lam * self.filters_flat)

        G = -(Y_batch-P)
        grad_W2 = (1/self.batch_size) * G @ x1.T
        grad_b2 = (1 / self.batch_size) * np.sum(G, axis=1, keepdims=True)

        G = self.W[1].T @ G
        G = G * (x1 > 0).astype(int)

        grad_W1 = (1 / self.batch_size) * G @ h.T
        grad_b1 = (1 / self.batch_size) * np.sum(G, axis=1, keepdims=True)

        G_batch = self.W[0].T @ G
        G_batch = G_batch * (h > 0).astype(int)

        GG = G_batch.reshape( (self.n_p, self.n_f, self.batch_size), order='C' )

        MXt = np.transpose(self.MX, (1, 0, 2))
        grad_Fs_flat = np.einsum('ijn, jln ->il', MXt, GG, optimize=True) / self.batch_size
        grad_b_conv = np.sum(GG, axis=(0, 2), keepdims=True) / self.batch_size
        grad_b_conv = grad_b_conv.reshape(self.n_f, 1)

        return {
            'grad_Fs_flat': grad_Fs_flat + reg_Fs,
            'grad_W1': grad_W1 + reg_W1,
            'grad_W2': grad_W2 + reg_W2,
            'grad_b1': grad_b1,
            'grad_b2': grad_b2,
            'grad_b_conv': grad_b_conv
        }

    def set_eta(self, t):
        '''
        Updates the learning-rate (ETA) using a schedule where the cycle length doubles.
        '''
        s = self.base_step_size  # fixed base step size
        total = 0
        l = 0

        # Figure out which cycle t is in
        while True:
            cycle_len = 2 * s * (2 ** l)
            if t < total + cycle_len:
                break
            total += cycle_len
            l += 1

        t_in_cycle = t - total
        half = cycle_len // 2

        if t_in_cycle < half:
            eta_t = self.lr_min + (t_in_cycle / half) * (self.lr_max - self.lr_min)
        else:
            eta_t = self.lr_max - ((t_in_cycle - half) / half) * (self.lr_max - self.lr_min)

        self.lr = eta_t

    def train(self, X, Y, y, val=0.2, epochs=5, seed=None, k=4, n_cycles=4):
        '''
        Takes the data as argument and trains the model parameters.
        For each epoch it randomly shuffles the data.
        '''
        N = X.shape[-1]
        if seed is None:
            np.random.seed(seed)

        # Split indices into train and validation sets
        indices = np.random.permutation(N)
        n_val = int(N * val)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_val = X[:, :, :, val_idx]
        Y_val = Y[:, val_idx]
        y_val = y[val_idx]

        batch_losses_val = []
        batch_accuracies_val = []
        losses_train = []
        accuracies_train = []

        plot_x_axis = []
        etas = []

        K = int(N * (1 - val) // self.batch_size)  # The 'new' number of batches after validation assignment.
        print(K)
        if self.step_size is None:
            self.step_size = int(k * K)  # updates the step size
        self.base_step_size = self.step_size
        print(f"Step size: {self.step_size}")

        # Simulation to calculate a new total updates needed for dynamic updates of step size.
        #total_updates_needed = n_cycles * 2 * self.step_size
        total_updates_needed = 0
        current_cycle_step_size = self.base_step_size
        for _ in range(n_cycles):
            total_updates_needed += 2 * current_cycle_step_size
            current_cycle_step_size *= 2  # Simulate the doubling for calculation
        epochs = int(np.ceil(total_updates_needed / K))
        print(f"Running for {n_cycles} cycles.")
        print(f"Running for {epochs} epochs.")
        print(f"t: {total_updates_needed}")

        t = 1                   # The number of steps in the cyclic learning rates.

        completed_cycles = 0
        step_size = self.step_size

        for i in range(epochs):
            train_ids_shuffled = np.random.permutation(train_idx)
            i += 1
            for j in tqdm(range(K), desc=f"Epoch {i+1}"):
                if (j+1)*self.batch_size > int(N*(1-val)): break
                if t > total_updates_needed: break

                batch_idx = train_ids_shuffled[j * self.batch_size : (j+1) * self.batch_size]
                X_batch = X[:, :, :, batch_idx]
                Y_batch = Y[:, batch_idx]
                y_batch = y[batch_idx]
                self.set_eta(t)
                self.MX = self.construct_MX(X_batch)
                grads = self.backwards(X_batch, Y_batch)
                self.update_params(grads)

                '''
                Code for testing the Gradients. Doesn't work with regularization, since it is not implemented into the torch code. 
                Did test it with lam=0, and it works fine! 
                if t % 500 == 0:
                    batch_size = self.batch_size
                    self.batch_size = X_val.shape[-1]
                    self.MX = self.construct_MX(X_val)
                    grads = self.backwards(X_val, Y_val)
                    self.compare_grads_with_torch(X_val, y_val, grads)
                    self.batch_size = batch_size
                '''

                if t % 100 == 0: # With dynamic step size, I changed this to 100 instead of step/8.
                    # Calculating loss and accuracy for training and validation
                    P = self.forward(X_batch)
                    losses_train.append(self.loss(Y_batch, P))
                    accuracies_train.append(self.accuracy(y_batch, P))
                    # Need to reconstruct some data for the validation batch.
                    batch_size = self.batch_size
                    self.batch_size = X_val.shape[-1]
                    cnn.MX = cnn.construct_MX(X_val)
                    P_val = self.forward(X_val)
                    batch_accuracies_val.append(self.accuracy(y_val, P_val))
                    batch_losses_val.append(self.loss(Y_val, P_val))
                    self.batch_size = batch_size
                    plot_x_axis.append(t)

                t += 1
                etas.append(self.lr)


        # Plot losses
        plt.figure(figsize=(8, 5))
        plt.plot(plot_x_axis,losses_train, label='Training Loss')
        plt.plot(plot_x_axis,batch_losses_val, label='Validation Loss')
        plt.xlabel('Update step')
        plt.ylabel('Loss')
        plt.title('Loss over Update steps')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot accuracies
        plt.figure(figsize=(8, 5))
        plt.plot(plot_x_axis,accuracies_train, label='Training Accuracy')
        plt.plot(plot_x_axis,batch_accuracies_val, label='Validation Accuracy')
        plt.xlabel('Update step')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Update steps')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(etas)
        plt.show()

# -----------------------------------------------
    # END OF CLASS METHODS






# -----------------------------------------------
#  DATA HANDLING FUNCTIONS
# -----------------------------------------------
def extract_data(pickle_dict):
    # Extract the image data and cast to float from the dict dictionary
    y = np.array(pickle_dict[b'labels'])
    k = len(set(y))
    Y = np.zeros((k, len(y)), dtype=np.float64)
    for i in range(Y.shape[1]):
        label = y[i]
        Y[label, i] = 1.0
    X = pickle_dict[b'data'].astype(np.float64) / 255.0  # RGB pixel values max at 255, making entries between 0 and 1.
    X = X.transpose()
    return X, Y, y

def read_data(path_name, dir=False):
    '''
    Can read data from multiple files in a directory,
    or just one file.
    :param path_name: Name of file or directory.
    :param dir: Specifies if the path-name is a directory or not.
    :return: X, Y, y
    '''

    if dir:
        Xs, Ys, ys = [], [], []
        for file in path_name.iterdir():
            if file.is_file() and not (file.name.endswith('.html') or file.name.endswith('.meta')):
                with open(file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                X, Y, y = extract_data(dict)
                Xs.append(X)
                Ys.append(Y)
                ys.append(y)
        return np.concatenate(Xs, axis=1), np.concatenate(Ys, axis=1), np.concatenate(ys, axis=0)
    else:
        # No error handling so make sure it is the correct path!
        with open(path_name, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return extract_data(dict)

def split(X, Y, y, train=0.8, val=0.2, seed=None, wantTest=False):
    assert train+val <= 1.0, f"Train and validation cannot exceed 1.0."
    test = 1.0 - train - val
    N = X.shape[-1]
    indices = np.arange(N)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)

    n_train = int(N * train)
    n_val = int(N * val)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    if wantTest:
        return (
            X[:, train_idx], Y[:, train_idx], y[train_idx],
            X[:, val_idx], Y[:, val_idx], y[val_idx],
            X[:, test_idx], Y[:, test_idx], y[test_idx]
        )
    else:
        return (
            X[:, train_idx], Y[:, train_idx], y[train_idx],
            X[:, val_idx], Y[:, val_idx], y[val_idx]
        )

def preprocess_data(X_train, X_val, X_test, img_shape):
    # Normalizes the data.
    mean = np.mean(X_train, axis=1).reshape (X_train.shape[0], 1)
    std = np.std(X_train, axis=1).reshape(X_train.shape[0], 1)

    # Normalize
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test



# -----------------------------------------------
#  MAIN
# -----------------------------------------------
if __name__ == "__main__":
    print("Running main...")

    file_path_dir = Path("Datasets/cifar-10-batches-py")
    file_path_batch = file_path_dir / 'data_batch_1'

    X, Y, y = read_data(file_path_dir, dir=True)
    #X_train, Y_train, y_train, X_val, Y_val, y_val = split(X, Y, y)
    X_test, Y_test, y_test = read_data(file_path_dir / 'test_batch')

    X_train, X_val, X_test = preprocess_data(X, X_test, X_test, (32, 32, 3, X.shape[-1]))
    X_train = X_train.reshape(32,32,3, X_train.shape[-1])
    X_train = X_train.astype(np.float32)
    print(X_train.shape)

    cnn = CNN(X=X_train, Y=Y, y=y, lr_min=1e-5, lr_max=1e-1, lam=0.0025, n_batches=100,
              n_filters=40, stride=4, n_hidden=300, step_size=800, init_MX=False)
    cnn.train(X_train, Y, y)


    # Final run of the test data.
    X_test = X_test.reshape(32,32,3, X_test.shape[-1])
    cnn.batch_size = X_test.shape[-1]
    cnn.MX = cnn.construct_MX(X_test)
    P = cnn.forward(X_test)
    print(f"Accuracy on test data: {cnn.accuracy(y_test, P)}")


