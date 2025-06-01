import numpy as np

class Sample_parameters():
    def __init__(self):
        data = np.load('debug_info.npz')
        self.X = data['X']
        self.Y = data['Y']
        self.y = data['y']
        self.Fs = data['Fs']
        self.fw = data['fw']
        self.nf = data['nf']
        self.Fs_flat = data['Fs_flat']
        self.nh = data['nh']
        self.W1 = data['W1']
        self.W2 = data['W2']
        self.b1 = data['b1']
        self.b2 = data['b2']
        self.MX = data['MX']

        self.conv_outputs = data['conv_outputs']
        self.conv_outputs_mat = data['conv_outputs_mat']
        self.conv_outputs_flat = data['conv_outputs_flattened']
        self.conv_flat = data['conv_flat']

        self.X1 = data['X1']
        self.P = data['P']

        self.grad_Fs_flat = data['grad_Fs_flat']
        self.grad_W1 = data['grad_W1']
        self.grad_W2 = data['grad_W2']
        self.grad_b1 = data['grad_b1']
        self.grad_b2 = data['grad_b2']

