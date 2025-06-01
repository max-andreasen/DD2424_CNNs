import unittest
import numpy as np
from assignment_3 import CNN
from debugging import Sample_parameters

class TestCNN(unittest.TestCase):

    samples = Sample_parameters()

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

        cnn = CNN(X=X)
        output = cnn.convolve(X, conv_filter, stride)

        self.assertEqual(output.shape, expected_output.shape)
        np.testing.assert_allclose(output, expected_output)


    # FORWARD PASS
    def test_forward_pass(self):
        return

if __name__ == "__main__":
    unittest.main()