# matrix-multiplication-exercise

## Exercise description:

```
Multiply the three matrices: A, B, and C;  i.e., you are expected to find the matrix D where D=(A*B)*C.
A, B, and, C contain random numbers in the range of (0,1) and the dimensions of the matrices are as follows.
 - A is a matrix with dimension 10^6 x 10^3.
 - B is a matrix with dimension 10^3 x 10^6.
 - C is a matrix with dimension 10^6 x 1.
```

We can implement it with (A*B)*C = A*(B*C), which works very fast. But as the exercise description has the parentheses for A\*B, I will try to implement it that way.

## Implementation

Using Python and Numpy.

Making the assumption, that we need float64, i.e. double, we need 8 bytes for each element, so the required memory space for the Matrix A and B would be, `10^6*10^3*8bytes = 8000000000 bytes = 8 Gb*2 = 16 Gb`.
Matrix C will require `10^6*8bytes = 8000000 bytes = 8 Mb`.

These could be written to disk, as shown in the commented out code, but will do it all in memory.
If we would use `np.dot(np.dot(A, B), C), yea, we would need many terabytes of memory, so we need to do it with [matrix blocks](https://en.wikipedia.org/wiki/Block_matrix).
