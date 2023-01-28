import numpy as np
import gc

# Sizes of matrices according to exercise definition
matrix_size_one = 10**6
matrix_size_two = 10**2

# Matrice shapes
A_shape = (matrix_size_one, matrix_size_two)
B_shape = (matrix_size_two, matrix_size_one)
C_shape = (matrix_size_one, 1)

# Create Matrix A, B, C into memory, (0,1), i.e. no 0 or 1 accepted
# np.random.rand can't be used, as its [0,1)
A = np.random.uniform(low=0.0001, high=0.9999, size=A_shape)
B = np.random.uniform(low=0.0001, high=0.9999, size=B_shape)
C = np.random.uniform(low=0.0001, high=0.9999, size=C_shape)

# # Save matrices as data on hard drive, Matrix A and B will allocate 10^6*10^3*8 bytes of space, which is 8 Gigabytes
# # These need to be on hard disk
# np.save('data/matrix_a.npy', A)
# np.save('data/matrix_b.npy', B)
# np.save('data/matrix_c.npy', C)

# # Call explicitely Python garbage collect, to remove A, B, C from memory
# del A, B, C
# gc.collect()

# # Fetch the data from hard disk to memory when needed, out-of-core algorithm
# A_mem = np.memmap('data/matrix_a.npy', dtype='float64', mode='r', shape=(matrix_size_one, matrix_size_two))
# B_mem = np.memmap('data/matrix_b.npy', dtype='float64', mode='r', shape=(matrix_size_two, matrix_size_one))
# C_mem = np.memmap('data/matrix_c.npy', dtype='float64', mode='r', shape=(matrix_size_one, 1))

# Set the batch size
batch_size = 1000

# Number of batches of matrix A and B
A_row_batches = A.shape[0] // batch_size
B_column_batches = B.shape[1] // batch_size

# Initialize the result matrix
result = np.array([], dtype=np.float64)

for i in range(A_row_batches):
  # Collect all rows from batches
  temp_rows = [[] for _ in range(batch_size)]
  for j in range(B_column_batches):
    # Select sub matrix rows in batches from 2D A_mem, and all columns
    block_a = A[i*batch_size:(i+1)*batch_size,:]
    # Select sub matrix columns in batches from 2D B_mem, and all rows
    block_b = B[:,j*batch_size:(j+1)*batch_size]
    # Calculate the sub matrixes product
    batch_result = np.dot(block_a, block_b)
    # Append the correct batch_result row to the correct temp_row
    for m, row_temp in enumerate(batch_result):
      temp_rows[m] = np.append(temp_rows[m], row_temp)
    # Garbage collection
    del block_a, block_b, batch_result
    gc.collect()

  # For each batch temporary_row, we want to multiply with C
  for row in temp_rows:
    with_c = np.dot(row, C)
    result = np.append(result, with_c)

# Make the result a 2D array and reshape it to be dimension x,1
print(np.array([result]).reshape(-1, 1))