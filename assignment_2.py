
'''
    Assignment #2, part of the course DD2424 Deep Learning in Data Science. 
    By Max Andreasen. 

    Contructing a 2 layer nerual network using cyclic learning rate as well as doing a 
    lambda search for tuning. 
    
    To improve on my last assignment, I tried the more conventional way of implementing the
    neural network as a class model instead of saving the parameters in a dictionairy. 
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch_gradient_computations import ComputeGradsWithTorch


class Neural_Network: 
    
    # Network parameters. 
    def __init__(self, d, m, k, lam=0, eta=0.001, eta_min=0.00001, eta_max=0.1):

        # Class is specifically constructed for having 2 layers, L. 
        self.L = 2

        # d is the input size, (e.g 3072 for this dataset).
        self.d = d

        # m is the number of neurons in the hidden layer. 
        self.m = m

        # k is the amount of lables / classes. 
        self.k = k

        # L is the amount of layers, specified when initializing the neural network.
        self.W = [None] * self.L
        self.B = [None] * self.L
        self.grads = {}

        # Network parameter
        self.lam = lam
        self.eta = eta # the learning rate
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.step_size = 0 # the step size for dynamic updating of eta.

        # For plotting purposes. Saves the losses & costs of the latest run
        self.losses_training = [] # by epochs
        self.losses_val = []
        self.costs_training = []
        self.costs_val = []
        self.acc_training = []
        self.acc_val = []

        # Updated relative to the update steps (cycles). 
        self.step_losses_training = []
        self.step_losses_val = []
        self.step_costs_training = []
        self.step_costs_val = []
        self.step_acc_training = []
        self.step_acc_val = []

        self.etas = []
    
    def __str__(self):
        return f"Network parameters: \n Layers: {self.L} \n Number of neurons: {self.m} \n Lambda: {self.lam} \n Learning rate: {self.eta}"

    def get_grads(self):
        return self.grads
    def get_network(self):
        return {'W': self.W, 'b': self.B, 'lam': self.lam}

    def init_parameters(self):
        self.B[0] = np.zeros( (self.m, 1) )
        self.B[1] = np.zeros( (self.k, 1) )
        self.W[0] = np.random.normal(0, 1/np.sqrt(self.d), (self.m, self.d) )
        self.W[1] = np.random.normal(0, 1/np.sqrt(self.m), (self.k, self.m) )
    
    # Is this really necessary though? Maybe easier to just create a new isntance of a network.
    def reset_network(self):
        self.W = [None] * self.L
        self.B = [None] * self.L
        self.grads = {}
        self.losses_training = [] # by epochs
        self.losses_val = []
        self.costs_training = []
        self.costs_val = []
        self.acc_training = []
        self.acc_val = []
        self.step_losses_training = [] # by update steps (cycles)
        self.step_losses_val = []
        self.step_costs_training = []
        self.step_costs_val = []
        self.step_acc_training = []
        self.step_acc_val = []
        self.etas = []

    def compute_loss(self, Y, P):
        '''
        Computes the cross entropy loss for the one-hot encoded label vector, 
        and the probability distribution P which is calculated by the network.
        '''
        eps = 1e-15 
        l_cross = -np.sum( Y * np.log(P + eps)) / Y.shape[1] # Eq. 6. 
        return l_cross

    def compute_cost(self, Y, P):
        '''
        Calculates the cost function (J), which in this assignment is mainly used for plotting pursposes. 
        '''
        l_cross = self.compute_loss(Y, P)
        reg_term = np.sum(self.W[0]**2) + np.sum(self.W[1]**2)
        return l_cross + ( self.lam * reg_term)

    def compute_accuracy(self, y, P):
        pred_labels = np.argmax(P, axis=0)
        comparison_vec = (pred_labels == y).astype(int)
        acc = np.mean(comparison_vec)
        return acc

    def _softmax(self, S):
        '''
        Takes an output vector, S, and turns it into a probability distribution. 
        '''
        exp_S = np.exp(S - np.max(S, axis=0, keepdims=True)) 
        return exp_S / np.sum(exp_S, axis=0, keepdims=True)  # Normalize over columns (axis=0)

    def apply_network(self, X):
        '''
        The FORWARD PASS. Runs X through the network and computes an output. 
        The output runs through softmax to create a probability distribution for the labels / classes.
        '''
        H = np.maximum( self.W[0] @ X + self.B[0], 0 ) # shape will be m x n (50, 10 000)
        s = self.W[1] @ H + self.B[1]
        p = self._softmax(s)
        return p, H

    def backwards_pass(self, X, Y):
        '''
        The BACKWARDS PASS. 
        Computes the gradients of the cross entropy loss w.r.t the weights / biases.
        '''
        N = Y.shape[1]
        P, H = self.apply_network(X)
        G = -(Y-P)

        # Adding L2 regularization.
        reg_W1 = (2 * self.lam * self.W[0])
        reg_W2 = (2 * self.lam * self.W[1])

        dL_dW2 = (1/N) * G @ H.T + reg_W2
        dL_dB2 = (1/N) * G @ np.ones(N) # matmul is faster than using np.sum

        G = self.W[1].T @ G
        G = G * (H > 0).astype(int)
        
        dL_dW1 = (1/N) * G @ X.T + reg_W1
        dL_dB1 = (1/N) * G @ np.ones(N)

        dL_dB1 = dL_dB1.reshape(-1, 1)
        dL_dB2 = dL_dB2.reshape(-1, 1)

        self.grads['W'] = [dL_dW1, dL_dW2]
        self.grads['b'] = [dL_dB1, dL_dB2]
    

    def set_eta(self, t):
        '''
        Upadtes the learning-rate (ETA), based on the internally set cycle. 
        t: int 
            The number gets a +1 every time the model updates the gradients.
        '''

        l = int(np.floor(t / (2 * self.step_size))) # l is the number of whole cycles that has been.

        bound_1 = (2*l) * self.step_size
        bound_2 = (2*l + 1) * self.step_size

        # eq. 14. 
        if (bound_1 <= t) and (bound_2 > t):
            eta_t = self.eta_min + ( (t-bound_1)/self.step_size * (self.eta_max - self.eta_min) )
        # eq. 15.
        elif (bound_2 <= t) and ( 2*(l+1)*self.step_size > t): 
            eta_t = self.eta_max - ( (t-bound_2) / self.step_size * (self.eta_max-self.eta_min) )
        else:
            # Could also raise an error here perhaps? 
            print("The value t is not within any bound. Keeping the current learning rate.")
            return

        self.eta = eta_t # updates the learning rate
    

    def minibatchGD(self, train_X, train_Y, train_y, val_X, val_Y, val_y, params):
        '''
        Performs the minibatch gradient descent and return lists of the losses and costs.
        args: 
        train_X: np.array 
        train_Y: np.array
            The whole training data that is later divided into smaller bathes.
        val_X: 
        val_Y:
            The whole validation data. 
        params: 
            A dictionairy with 'n_batches' and 'n_epochs' and 'k' (for dynamic eta updates). 

        '''

        N = train_X.shape[1] # number of data points in large batch.

        n_batch = int(params['n_batch'])
        n_epochs = int(params['n_epochs'])
        k = int(params['k'])

        self.step_size = int(k * (N/n_batch)) # updates the step size
        print(f"Step size: {self.step_size}")

        t = 1
        
        for i in range(n_epochs):
            print(f"Epoch {i+1} completed...")
            for j in range(int(N/n_batch)):
                j_start = j*n_batch
                j_end = (j+1)*n_batch
                batch_X_train = train_X[:, j_start:j_end]
                batch_Y_train = train_Y[:, j_start:j_end]

                self.set_eta(t)
                self.backwards_pass(batch_X_train, batch_Y_train) # updates the gradients of the model.

                # Updates the network
                self.W[0] -= ( self.eta * self.grads['W'][0] ) 
                self.W[1] -= ( self.eta * self.grads['W'][1] )
                self.B[0] -= ( self.eta * self.grads['b'][0] )
                self.B[1] -= ( self.eta * self.grads['b'][1] )

                t += 1
                self.etas.append(self.eta)

                #TODO: Must be a simpler and less cluttery way of appedning to the lists? 
                if t % 100 == 0: # updates every 10th step
                    P_train, H = self.apply_network(train_X)
                    P_val, H = self.apply_network(val_X)
                    self.step_losses_training.append( self.compute_loss(train_Y, P_train) )
                    self.step_losses_val.append( self.compute_loss(val_Y, P_val) )
                    self.step_costs_training.append( self.compute_cost(train_Y, P_train) )
                    self.step_costs_val.append( self.compute_cost(val_Y, P_val) )
                    self.step_acc_training.append( self.compute_accuracy(train_y, P_train))
                    self.step_acc_val.append( self.compute_accuracy(val_y, P_val) )

            # Update losses & costs.
            P_train, H = self.apply_network(train_X)
            P_val, H = self.apply_network(val_X)

            #TODO: And also here. Perhaps create a metrics dictionairy to at least save some space in the init. 
            self.losses_training.append( self.compute_loss(train_Y, P_train) )
            self.losses_val.append( self.compute_loss(val_Y, P_val) )
            self.costs_training.append( self.compute_cost(train_Y, P_train) )
            self.costs_val.append( self.compute_cost(val_Y, P_val) )
            self.acc_training.append( self.compute_accuracy(train_y, P_train))
            self.acc_val.append( self.compute_accuracy(val_y, P_val) )
        
        print(f"Cycles completed: {t/ (2*self.step_size)}")
    

    # Mainly for checking the gradients (according to the assignemtn description).
    def regularGD(self, train_X, train_Y, val_X, val_Y, epochs=100):
        losses_train = []
        losses_val = []
        for _ in range(epochs):
            self.backwards_pass(train_X, train_Y) # updates the gradients of the model.
            # Updates the network
            self.W[0] -= ( self.eta * self.grads['W'][0] ) 
            self.W[1] -= ( self.eta * self.grads['W'][1] )
            self.B[0] -= ( self.eta * self.grads['b'][0] )
            self.B[1] -= ( self.eta * self.grads['b'][1] )
            P_train, H = self.apply_network(train_X)
            P_val, H = self.apply_network(val_X)
            losses_train.append(self.compute_loss(train_Y, P_train))
            losses_val.append(self.compute_loss(val_Y, P_val))
        return losses_train, losses_val

    def broad_lam_search(self, X, Y, y, lam_bounds, filename='results.txt'):
        '''
        Runs a broader lambda search and writes the results to a file. 
        args: 
        X, Y, y: np.array
            The entire data.
        lam_bounds: tuple
            Tuple containing log values (floats) for lam min and lam max. 
        '''
        val_ind = np.random.choice(X.shape[1], 5000, replace=False)
        train_ind = np.setdiff1d(np.arange(X.shape[1]), val_ind)
        train_X = X[:, train_ind]
        train_Y = Y[:, train_ind]
        train_y = y[train_ind]
        val_X = X[:, val_ind]
        val_Y = Y[:, val_ind]
        val_y = y[val_ind]

        l_min, l_max = lam_bounds
        rng = np.random.default_rng()
        params = {'n_batch': 100, 'n_epochs': 8, 'k': 2}
        accs = {} # accuracies as keys, log lambda values as values.

        n = 35
        for i in range(n):
            self.reset_network()
            self.init_parameters() # resets the weights and biases.
            l = l_min + (l_max - l_min) * rng.random() # sets a random sampled eta
            self.lam = 10**l
            self.minibatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, params)
            P_val, H = self.apply_network(val_X) # runs the validations data through the network
            acc = self.compute_accuracy(val_y, P_val) # computes an accuracy of the validation data. 
            accs[acc] = l
            print(f"Loop {i+1} finished...")
        
        accs = dict(sorted(accs.items(), reverse=True))
        with open('output.txt', 'a') as file:
            for element in accs.keys():
                file.write(f"Accuracy: {round(element*100, 2)}%, Lambda (log): {round(accs[element], 5)} \n")


    #TODO Must be a more efficient way of doing this. Maybe extract the lists and onyl use one plot function. 
    #TODO Then I can only use one method or 2 inside of the class. 

    # Plotting methods
    def plot_styling(self, title_str, x_label, y_label):
        plt.grid(alpha=0.3)
        plt.title(f"{title_str}")
        plt.xlabel(f"{x_label}")
        plt.ylabel(f"{y_label}")
    def plot_losses_with_epoch(self): 
        self.plot_styling("Losses with each epoch", "Epoch", "Amount of loss")
        plt.plot(self.losses_training, label="Training data")
        plt.plot(self.losses_val, label="Validation data")
        plt.legend()
        plt.show()
    def plot_costs_with_epoch(self):
        self.plot_styling("Costs with each epoch", "Epoch", "Amount of cost")
        plt.plot(self.costs_training, label="Training data")
        plt.plot(self.costs_val, label="Validation data")
        plt.legend()
        plt.show()
    def plot_etas(self): 
    # Plots the etas (checks the cyclic shape). 
        self.plot_styling("Learning rate over update steps", "Update step", "Learning rate (eta)")
        plt.plot(self.etas)
        plt.show()
    def plot_losses_with_step(self):
        self.plot_styling("Losses with update steps", "Update step", "Loss") 
        plt.plot(self.step_losses_training, label="Training data")
        plt.plot(self.step_losses_val, label="Validation data")
        plt.legend()
        plt.show()
    def plot_costs_with_step(self):
        self.plot_styling("Costs with update steps", "Update step", "Cost") 
        plt.plot(self.step_costs_training, label="Training data")
        plt.plot(self.step_costs_val, label="Validation data")
        plt.legend()
        plt.show()
    def plot_accs_with_step(self):
        self.plot_styling("Accuracy with update steps", "Update step", "Accuracy") 
        plt.plot(self.step_acc_training, label="Training data")
        plt.plot(self.step_acc_val, label="Validation data")
        plt.legend()
        plt.show()



def read_data(filename): 
    # Loads a batch of training data.
    try:
        cifar_dir = './Datasets/cifar-10-batches-py/'
        with open(cifar_dir + filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
    except:
        cifar_dir = '/Users/maxandreasen/GitHub/DD2424_DL/Datasets/cifar-10-batches-py'
        with open(cifar_dir + filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

    # Extract the image data and cast to float from the dict dictionary
    y = np.array(dict[b'labels'])
    k = len(set(y))
    Y = np.zeros((k, len(y)), dtype=np.float64)
    for i in range(Y.shape[1]):
        label = y[i]
        Y[label,i] = 1.0

    X = dict[b'data'].astype(np.float64) / 255.0 # RGB pixel values max at 255, making entries between 0 and 1. 
    X = X.transpose()

    return X, Y, y

def read_all_batches():

    # Creates lists with np.arrays, that will later be concatenated. 
    X = []
    Y = []
    y = []

    cifar_dir = './Datasets/cifar-10-batches-py/'
    for i in range(5): 
        with open(f"{cifar_dir}data_batch_{i+1}", 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        # Extract the image data and cast to float from the dict dictionary
        y_new = np.array(dict[b'labels'])
        k = len(set(y_new))
        Y_new = np.zeros((k, len(y_new)), dtype=np.float64)
        for i in range(Y_new.shape[1]):
            label = y_new[i]
            Y_new[label,i] = 1.0

        X_new = dict[b'data'].astype(np.float64) / 255.0 # RGB pixel values max at 255, making entries between 0 and 1. 
        X_new = X_new.transpose()
        X.append(X_new)
        Y.append(Y_new)
        y.append(y_new)
    
    X = np.concatenate(X, axis=1)
    Y = np.concatenate(Y, axis=1)
    y = np.concatenate(y, axis=0)

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


def test_grads(train_X, train_y, grads, network):
    '''
    Tests the gradients with the torch gradients. 
    Make sure that the gradients is derived from the same data inputted to this function. 
    '''
    torch_grads = ComputeGradsWithTorch(train_X, train_y, network)
    thresh = 0.00001
    # Testing the gradients
    for param in grads:
        for i in range(2):
            grad = grads[param][i]
            t_grad = torch_grads[param][i]
            diff = np.max(np.abs(grad-t_grad))
            if diff > thresh:
                return f"Diff of gradients is too large for {param} and {i}. Mean diff = {diff}"
    return "Gradients OK."



def main(): 


    # ............... CHECKING GRADIENTS .................. #
    # Reads and processes the data. 
    train_X_batch, train_Y_batch, train_y_batch = read_data('/data_batch_1')
    val_X_batch, val_Y_batch, val_y_batch = read_data('/data_batch_2')
    test_X_batch, test_Y_batch, test_y_batch = read_data('/test_batch')

    train_X_batch, val_X_batch, test_X_batch = preprocess_data(train_X_batch, val_X_batch, test_X_batch)


    # Initializing the neural network
    d = train_X_batch.shape[0] 
    k = len(set(train_y_batch))
    m = 50

    nn = Neural_Network(d, m, k, lam=0.01, eta=0.01)
    nn.init_parameters()

    # Checking the gradients with torch.
    nn.backwards_pass(train_X_batch, train_Y_batch) # Updates the gradients of the model once. 
    grads = nn.get_grads()
    network = nn.get_network()
    print(test_grads(train_X_batch, train_y_batch, grads, network))

    # Testing the gradients.
    #test_nn = Neural_Network(d, m, k, lam=0, eta=0.01)
    #test_nn.init_parameters()
    #train_X_small = train_X[:, :100]
    #train_Y_small = train_Y[:, :100]
    #val_X_small = val_X[:, :100]
    #val_Y_small = val_Y[:, :100]
    # losses_train, losses_val = test_nn.regularGD(train_X_small, train_Y_small, val_X_small, val_Y_small, 200)


    # ............ NETWORK TRAINING AND PLOTTING ............ #
    X, Y, y = read_all_batches()
    val_ind = np.random.choice(X.shape[1], 1000, replace=False)
    train_ind = np.setdiff1d(np.arange(X.shape[1]), val_ind)
    
    train_X = X[:, train_ind]
    train_Y = Y[:, train_ind]
    train_y = y[train_ind]
    val_X = X[:, val_ind]
    val_Y = Y[:, val_ind]
    val_y = y[val_ind]

    train_X, val_X, test_X_batch = preprocess_data(train_X, val_X, test_X_batch)

    # Initializing the neural network
    d = train_X.shape[0] 
    k = len(set(train_y))
    m = 50

    nn = Neural_Network(d, m, k, lam=0.00002, eta=0.01, eta_min=0.0001, eta_max=0.01)
    nn.init_parameters()
    #nn.broad_lam_search(X, Y, y, (-6, -4))

    params = {'n_batch': 50, 'n_epochs': 48, 'k': 8}
    nn.minibatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, params)
    nn.plot_accs_with_step()
    nn.plot_losses_with_step()
    nn.plot_costs_with_step()


if __name__ == "__main__":
    main()

        

