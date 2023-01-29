import numpy as np
import unittest
from memory_profiler import profile

# Works, but is A*(B*C)
class TestMatrixMultiplication(unittest.TestCase):
  @profile
  def test_matrix_multiplication(self):
    # Sizes of matrices according to exercise definition
    matrix_size_one = 10**6
    matrix_size_two = 10**3

    # Matrice shapes
    A_shape = (matrix_size_one, matrix_size_two)
    B_shape = (matrix_size_two, matrix_size_one)
    C_shape = (matrix_size_one, 1)

    A = np.random.uniform(low=0.0001, high=0.9999, size=A_shape)
    B = np.random.uniform(low=0.0001, high=0.9999, size=B_shape)
    C = np.random.uniform(low=0.0001, high=0.9999, size=C_shape)
    D = A @ (B @ C)

    # (A*B)*C = A*(B*C)
    expected_result = np.dot(A, np.dot(B,C))
    np.testing.assert_array_almost_equal(D,expected_result)

if __name__ == '__main__':
    unittest.main()
