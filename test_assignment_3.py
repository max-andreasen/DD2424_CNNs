import unittest
import numpy as np
from assignment_3 import CNN
from debugging import Sample_parameters

class TestCNN(unittest.TestCase):

    samples = Sample_parameters()
    X = samples.X  # The first of the 5 images.
    Y = samples.Y
    y = samples.y  # The label (not necessary).
    W1 = samples.W1
    W2 = samples.W2
    B1 = samples.b1
    B2 = samples.b2
    P = samples.P  # The softmax prediction of the first image.
    Fs = samples.Fs  # The filters (there are 2).
    MX = samples.MX

    x1 = samples.X1
    conv_outputs = samples.conv_outputs
    conv_flat = samples.conv_flat
    conv_outputs_mat = samples.conv_outputs_mat

    stride = Fs.shape[0]
    n_p = (32 // stride) ** 2

    X = np.transpose(X.reshape((32, 32, 3, 5), order='F'), (1, 0, 2, 3))

    cnn = CNN(X=X, Y=Y, y=y, W=[W1, W2], B=[B1, B2], stride=stride,
              filters=Fs, K=10, init_MX=False)

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
        cnn = CNN(X=X, Y=None, y=None, K=10, init_MX=False)
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
        # Running the forward pass to see if any errors occur within the forward pass.
        outputs = self.cnn.forward_pass(self.X, return_testing=True)
        self.assertEqual(outputs['P'].shape, self.P.shape)
        self.assertEqual(outputs['P'].dtype, self.P.dtype)
        self.assertEqual(outputs['x1'].shape, self.x1.shape)
        self.assertEqual(outputs['h'].shape, self.conv_flat.shape)

        print(outputs['h'][:10, 0])
        print(self.conv_flat[:10, 0])

        np.testing.assert_allclose(outputs['h'], self.conv_flat, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(outputs['x1'], self.x1, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(outputs['P'], self.P, rtol=1e-5, atol=1e-8)

    def test_forward_efficient(self):
        outputs = self.cnn.forward_efficient(self.X, return_testing=True)


if __name__ == "__main__":
    unittest.main()