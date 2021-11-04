# Suppose you have an array that contains values
# A = [1, 2, 3, 4, ..., N] and you want to
# compute the sum of A on the GPU.

import numpy as np
from numba import cuda

# This is a tempting solution, but it is not
# correct! When many asynchronous threads work
# together on the same space in memory (in this
# case adding to a number), they will overwrite
# each other's progress and hence mangle the
# final result.
@cuda.jit
def wrong_reduce(A, result):
    i = cuda.grid(1)
    if i < A.size:
        result[0] += A[i]

# To get the correct result, we need to use an
# "atomic" operation. Atomic operations limit
# the context of our computation on the
# hardware side, indicating to the computer that
# we want to perform a very specific operation.
# Modern compute hardware has been designed with
# specific capability to accelerate atomic
# operations.
@cuda.jit
def correct_reduce(A, result):
    i = cuda.grid(1)
    if i < A.size:
        # "Add A[i] to index 0 in 'result'"
        # You can also specify a tuple of
        # indices for multi-dimensional arrays
        cuda.atomic.add(result, 0, A[i])

N = 100000

# Create host arrays
A = np.arange(1,N+1, dtype='uint')
wrong_result = np.array([0], dtype='uint')
correct_result = np.array([0], dtype='uint')

# Copy host arrays to the device
device_A = cuda.to_device(A)
device_wrong_result = cuda.to_device(wrong_result)
device_correct_result = cuda.to_device(correct_result)

# Specify the total nubmer of threads
threadsperblock = 512
blockspergrid = N // threadsperblock + 1

# Compute the wrong result
wrong_reduce[blockspergrid,threadsperblock](
    device_A,
    device_wrong_result,
)
cuda.synchronize()
wrong_result = device_wrong_result.copy_to_host()
print("Wrong result   =",wrong_result)

# Compute the correct result
correct_reduce[blockspergrid,threadsperblock](
    device_A,
    device_correct_result,
)
cuda.synchronize()
correct_result = device_correct_result.copy_to_host()
print("Correct result =",correct_result)

# Show the analytic result
print("Analytic result N(N+1)/2 =",0.5*N*(N+1))
