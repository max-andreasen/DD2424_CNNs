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

        stride = 1

        # Manual calculation.
        expected_output = np.array([
            [2, 3],
            [4, 2]
        ])

        cnn = CNN(X=X, Y = None, y=None)
        output = cnn.convolve(X, conv_filter, stride)

        self.assertEqual(output.shape, expected_output.shape)
        np.testing.assert_allclose(output, expected_output)


    # FORWARD PASSES
    def test_forward_pass(self):
        # Don't need to test these. Maybe I will for good practice if I have time left.
        X1 = self.samples.X1
        conv_outputs = self.samples.conv_outputs[:, :, :, 0] # H_i

        stride = self.Fs.shape[0]            # Calculating the stride.
        n_p = (32 // stride) ** 2
        X = np.transpose( self.X.reshape((32, 32, 3, 5), order='F'), (1,0,2,3) )

        cnn = CNN(X=X, Y=self.Y, y=self.y, K=10, W=[self.W1, self.W2], B=[self.B1, self.B2], stride=stride, n_f=self.samples.nf, n_p=n_p, filters=self.Fs)

        # Running the forward pass to see if any errors occur within the forward pass.
        outputs = cnn.forward_pass(X, return_testing=True)
        self.assertEqual(outputs['P'].shape, self.P.shape)


    def test_forward_efficient(self):
        X = np.transpose(self.X.reshape((32, 32, 3, 5), order='F'), (1, 0, 2, 3))
        stride = self.Fs.shape[0]
        n_p = (X.shape[0] // stride) ** 2

        cnn = CNN(X=X, Y=self.Y, y=self.y, K=10, W=[self.W1, self.W2], B=[self.B1, self.B2], stride=stride, n_f=self.samples.nf, n_p=n_p, filters=self.Fs)
        outputs = cnn.forward_efficient(X, return_testing=True)


if __name__ == "__main__":
    unittest.main()