import unittest
import numpy as np
import gc

# Can't get batch size higher to work
def test(A, B, C):
  # Set the batch size
  batch_size = 10000

  # Number of batches of matrix A and B
  A_row_batches = A.shape[0] // batch_size
  B_column_batches = B.shape[1] // batch_size

  # Initialize the result matrix
  result = np.array([], dtype=np.float64)

  for i in range(A_row_batches):
    temp_row = [[] for _ in range(batch_size)]
    for j in range(B_column_batches):
        # Select sub matrix rows in batches from 2D A_mem, and all columns
        block_a = A[i*batch_size:(i+1)*batch_size,:]
        # Select sub matrix columns in batches from 2D B_mem, and all rows
        block_b = B[:,j*batch_size:(j+1)*batch_size]
        # Calculate the sub matrixes product
        batch_result = np.dot(block_a, block_b)
        # Append the correct batch_result row to the correct temporary row
        for m, row_temp in enumerate(batch_result):
          temp_row[m] = np.append(temp_row[m], row_temp)
        del block_a, block_b, batch_result
        gc.collect()
    # For each batch temporary_row, we want to multiply with C
    for row in temp_row:
      with_c = np.dot(row, C)
      result = np.append(result, with_c)
  # Make the result a 2D array and reshape it to be dimension x,1
  return np.array([result]).reshape(-1, 1)


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
        result = test(A, B, C)
        print(result)
        expected_result = np.dot(np.dot(A, B), C)
        expected_again = np.dot(A, np.dot(B, C))

        np.testing.assert_array_almost_equal(result,expected_result)
        np.testing.assert_array_almost_equal(result,expected_again)

if __name__ == '__main__':
    unittest.main()
