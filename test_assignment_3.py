import unittest
import numpy as np
from assignment_3 import CNN

class TestCNN(unittest.TestCase):
    data = np.load('debug_info.npz')
    X = data['X']
    Y = data['Y']
    y = data['y']
    Fs = data['Fs']
    fw = data['fw']
    nf = data['nf']
    Fs_flat = data['Fs_flat']
    nh = data['nh']
    W1 = data['W1']
    W2 = data['W2']
    b1 = data['b1']
    b2 = data['b2']
    MX = data['MX']

    conv_outputs = data['conv_outputs']
    conv_outputs_mat = data['conv_outputs_mat']
    conv_outputs_flat = data['conv_outputs_flattened']
    conv_flat = data['conv_flat']

    x1 = data['X1']
    P = data['P']

    grad_Fs_flat = data['grad_Fs_flat']
    grad_W1 = data['grad_W1']
    grad_W2 = data['grad_W2']
    grad_b1 = data['grad_b1']
    grad_b2 = data['grad_b2']

    stride = Fs.shape[0]
    n_p = (32 // stride) ** 2

    X = np.transpose(X.reshape((32, 32, 3, 5), order='F'), (1, 0, 2, 3))

    cnn = CNN(X=X, Y=Y, y=y, W=[W1, W2], B=[b1, b2], stride=stride,
              filters=Fs, init_MX=False)

    # CONVOLUTION
    def test_single_filter_convolution(self):
        # Input image: 3x3x2.
        X = np.array([
            [[1, 2], [2, 0], [1, 1]],
            [[0, 1], [1, 2], [2, 1]],
            [[1, 0], [0, 1], [1, 1]]
        ])
        # Filter: 2x2.
        conv_filter = np.array([
            [[1, 0], [0, 1]],
            [[1, -1], [0, 1]]
        ])
        # Manual calculation.
        expected_output = np.array([
            [2, 3],
            [4, 2]
        ])

        stride = 1
        cnn = CNN(X=X, Y=self.Y, y=self.y, init_MX=False)
        output = cnn.convolve(X, conv_filter, stride)

        self.assertEqual(output.shape, expected_output.shape)
        np.testing.assert_allclose(output, expected_output)

    def test_convolve_efficient(self):
        out_conv_outputs_mat = self.cnn.convolve_efficient(self.X)
        self.assertEqual(self.conv_outputs_mat.shape, out_conv_outputs_mat.shape)
        self.assertEqual(self.conv_outputs_mat.dtype, out_conv_outputs_mat.dtype)
        np.testing.assert_allclose(self.conv_outputs_mat, out_conv_outputs_mat, rtol=1e-5, atol=1e-8)

    def test_construct_MX(self):
        out_MX = self.cnn.construct_MX(self.X)
        self.assertEqual(self.MX.shape, out_MX.shape)
        self.assertEqual(self.MX.dtype, out_MX.dtype)
        np.testing.assert_allclose(self.MX, out_MX, rtol=1e-5, atol=1e-8)

    # FORWARD PASSES
    def test_forward_pass(self):
        outputs = self.cnn.forward_pass(self.X, return_params=True)
        self.assertEqual(outputs['P'].shape, self.P.shape)
        self.assertEqual(outputs['P'].dtype, self.P.dtype)
        self.assertEqual(outputs['x1'].shape, self.x1.shape)
        self.assertEqual(outputs['h'].shape, self.conv_flat.shape)

        np.testing.assert_allclose(outputs['h'], self.conv_flat, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(outputs['x1'], self.x1, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(outputs['P'], self.P, rtol=1e-5, atol=1e-8)

    def test_forward_efficient(self):
        outputs = self.cnn.forward_efficient(self.X, return_params=True)
        self.assertEqual(outputs['P'].shape, self.P.shape)
        self.assertEqual(outputs['P'].dtype, self.P.dtype)
        self.assertEqual(outputs['x1'].shape, self.x1.shape)
        self.assertEqual(outputs['h'].shape, self.conv_flat.shape) # h is 'conv_flat'

        np.testing.assert_allclose(outputs['h'], self.conv_flat, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(outputs['x1'], self.x1, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(outputs['P'], self.P, rtol=1e-5, atol=1e-8)

    # BACKWARDS PASS
    def test_backwards_pass(self):
        outputs = self.cnn.backwards_pass(self.X, self.Y)
        self.assertEqual(outputs['grad_Fs_flat'].shape, self.grad_Fs_flat.shape)
        self.assertEqual(outputs['grad_W1'].shape, self.grad_W1.shape)
        self.assertEqual(outputs['grad_W2'].shape, self.grad_W2.shape)

        np.testing.assert_allclose(outputs['grad_W1'], self.grad_W1, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(outputs['grad_W2'], self.grad_W2, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(outputs['grad_Fs_flat'], self.grad_Fs_flat, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()