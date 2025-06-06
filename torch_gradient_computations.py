import torch
import numpy as np

def ComputeGradsWithTorch(X, y, network_params):
    '''
    :param X: A BATCH of data (32x32x3xN)
    :param y: Corresponding labels, NOT one-hot encoding.
    :param network_params:
        W and b are lists of numpy arrays.
        Fs is a numpy array.
        MX is a numpy array.
        Stride is the stride.
    :return:
    '''

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    L = len(network_params['W'])
    stride = network_params['stride']
    n_f = network_params['Fs'].shape[-1] # The number of filters
    n_p = (X.shape[0] // stride) ** 2
    batch_size = X.shape[-1]

    # Re-constructs the weights and biases to torch tensors.
    W = [None] * L
    b = [None] * L
    for i in range(len(network_params['W'])):
        W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
        b[i] = torch.tensor(network_params['b'][i], requires_grad=True)

    Fs = torch.tensor(network_params['Fs'], requires_grad=True)
    MX = torch.from_numpy(network_params['MX'])

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    filters_flat = Fs.reshape((stride * stride * 3, n_f))

    # The FORWARD PASS
    conv_outputs_mat = torch.einsum('ijn,jl->iln', MX, filters_flat)
    h = apply_relu( conv_outputs_mat.reshape( (n_p*n_f, batch_size) ) ) # Also denoted as conv_flat

    x1 = apply_relu( torch.matmul(W[0],  h) + b[0] )
    scores = torch.matmul(W[1], x1) + b[1]

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(batch_size)]))
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()
    grads['Fs'] = Fs.grad.view(-1, n_f).numpy()

    return grads
