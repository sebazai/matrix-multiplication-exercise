import numpy as np
import gc
import time
import sys
import os.path
import matplotlib.pyplot as plt
# from memory_profiler import profile

start_time = time.time()

# @profile
def calculateMatrix(Ap=None, Bp=None, Cp=None):
  print("Starting")
  # time.sleep(30) was used only so that I have time to use psrecord to see mem/cpu usage.
  # time.sleep(30)
  print("started at %s" % start_time)
  if (Ap is None and Bp is None and Cp is None):
    # Sizes of matrices according to exercise definition 
    matrix_size_one = 10**6
    matrix_size_two = 10**3

    # Matrice shapes
    A_shape = (matrix_size_one, matrix_size_two)
    B_shape = (matrix_size_two, matrix_size_one)
    C_shape = (matrix_size_one, 1)

    # If Matrices are not on disk, create the
    if not (os.path.exists('data/matrix_a.npy') and os.path.exists('data/matrix_b.npy') and os.path.exists('data/matrix_c.npy')):
      # Create Matrix A, B, C into memory, (0,1), i.e. no 0 or 1 accepted
      # np.random.rand can't be used, as its [0,1), np.random.rand is also unfiromly distributed
      A = np.random.uniform(low=0.0001, high=1, size=A_shape)
      B = np.random.uniform(low=0.0001, high=1, size=B_shape)
      C = np.random.uniform(low=0.0001, high=1, size=C_shape)
      print("size of A in mem", sys.getsizeof(A))

      # Save matrices as data on hard drive, Matrix A and B will allocate 10^6*10^3*8 bytes of space, which is 8 Gigabytes
      # These need to be on hard disk
      np.save('data/matrix_a.npy', A)
      np.save('data/matrix_b.npy', B)
      np.save('data/matrix_c.npy', C)

      # Call explicitely Python garbage collect, to remove A, B, C from memory
      del A, B, C
      gc.collect()

    # Fetch the data from hard disk to memory when needed, out-of-core algorithm idea
    A_mem = np.load('data/matrix_a.npy', mmap_mode='r+')
    B_mem = np.load('data/matrix_b.npy', mmap_mode='r+')
    C_mem = np.load('data/matrix_c.npy', mmap_mode='r+')
    print("Size of a when loaded", sys.getsizeof(A_mem))
  else:
    A_mem = Ap
    B_mem = Bp
    C_mem = Cp
  # Set the batch size
  batch_size = 2000

  # Number of batches of matrix A and B
  A_row_batches = A_mem.shape[0] // batch_size
  B_column_batches = B_mem.shape[1] // batch_size

  # Initialize the result matrix
  result = np.array([], dtype=np.float64)

  for i in range(A_row_batches):
    # Collect all rows from batches
    temp_rows = [[] for _ in range(batch_size)]
    cycle_start = time.time()
    print("i", i)
    for j in range(B_column_batches):
      print("j", j)
      # Select sub matrix rows in batches from 2D A_mem, and all columns
      block_a = A_mem[i*batch_size:(i+1)*batch_size,:]
      # Select sub matrix columns in batches from 2D B_mem, and all rows
      block_b = B_mem[:,j*batch_size:(j+1)*batch_size]
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
      with_c = np.dot(row, C_mem)
      result = np.append(result, with_c)
      del with_c
    del temp_rows
    print("--- %s seconds ---" % (time.time() - cycle_start))
    gc.collect()

  # Make the result a 2D array and reshape it to be dimension X,1
  return_value = np.array([result]).reshape(-1, 1)
  print(return_value)
  print("--- %s seconds ---" % (time.time() - start_time))
  return return_value

# Plot Matrix A row ECDF
def ecdf(data):
  length = len(data)
  x = np.sort(data)
  y = np.arange(1, length+1) / length
  return x, y

def plotCDF():
  # Load A into memory
  A = np.load('data/matrix_a.npy')

  for i in range(10**3):
    # print(i)
    x, y = ecdf(A[:,i])
    plt.plot(x, y, marker='.', linestyle='none', markersize=3)
  # print("DONE")
  plt.xlabel('Value')
  plt.ylabel('ECDF')
  plt.show()

if __name__ == "__main__":
  calculateMatrix()
  # plotCDF()