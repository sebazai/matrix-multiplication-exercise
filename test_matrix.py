import unittest
import numpy as np

from matrix import calculateMatrix
class TestMatrixMultiplication(unittest.TestCase):
    def test_matrix_multiplication(self):
        # Matrices shapes, smaller for test purposes
        A_shape = (10**4, 10**2)
        B_shape = (10**2, 10**4)
        C_shape = (10**4, 1)

        # Initialize matrices with random values
        A = np.random.uniform(low=0.0001, high=0.9999, size=A_shape)
        B = np.random.uniform(low=0.0001, high=0.9999, size=B_shape)
        C = np.random.uniform(low=0.0001, high=0.9999, size=C_shape)

        result = calculateMatrix(A, B, C)
        expected_result = np.dot(np.dot(A, B), C)
        expected_again = np.dot(A, np.dot(B, C))

        np.testing.assert_array_almost_equal(result,expected_result)
        np.testing.assert_array_almost_equal(result,expected_again)

if __name__ == '__main__':
    unittest.main()
