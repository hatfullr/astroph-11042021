# This is a simple example showing how to add 1+1
# on the GPU and printing the result. Please make
# sure you have Numba and NumPy installed and that
# you are using Python 3.6+.
#
# Whenever we do work on a GPU, we build a set of
# instructions, prepare all the resources that the
# instructions work on (such as arrays, variables,
# etc) and then we sent it to the GPU and wait for
# the result. The packet of information that we
# send to the GPU is called a "kernel".
#
# A GPU is essentially a big cluster of tiny
# processors, and there is some software magic that
# goes into orchestrating the overall processing.
# The way that GPUs are understood by one big
# analogy. On a loom, there are many threads held
# in tension, and each thread is parallel to each
# other. All the threads combined are called the
# "warp". A fabric is created by passing another
# strand called the "weft" between the threads in
# the warp.
#
# In the GPU, there are also warps and threads, but
# the concept of a weft is not present. Each warp
# consists of 32 threads, and each "block" has some
# number of warps within it, where that number
# depends on the specific GPU architecture. From a
# coding perspective, we typically do not think in
# warps, but rather in blocks and threads, since the
# number of threads in a warp is always the same.
# When we request some number of blocks in the GPU,
# it is usually helpful to think of those blocks as
# being part of a grid. That grid can have any number
# of dimensions you desire, so you should pick a
# number of dimensions that makes your problem
# easiest to conceptualize.
#
# In this simple example, we will consider the grid
# to have a single dimension.
#
# So, to recap: We compile a set of instructions to
# send off to the GPU in what is called a "kernel".
# In our instructions, we tell the GPU how many
# threads we want to use per block, as well as how
# many blocks we would like to use in our 1D grid.

from numba import cuda
import numpy as np
import math


# Define a "device" (GPU) function:
@cuda.jit # This thing is called a "decorator"
def add(A,B,C):
    # A and B are input
    # C is output

    # Get the position of the current thread in the
    # entire grid of blocks. We use "1" as input
    # because we are using a single dimension grid.
    i = cuda.grid(1)
    
    # Make sure this thread is one that is actually
    # being used in our calculations. The number of
    # threads we requested may not be a perfect
    # multiple of 32 (32 threads in a warp), so
    # this step just makes sure we are only using
    # the threads we actually requested.
    if i < C.size: 
        C[i] = A[i] + B[i]


# We will have two arrays, each filled with ones
# and we will compute C[i] = A[i] + B[i] on the GPU.
N = 100000000
A = np.ones(N)
B = np.ones(N)
C = np.empty(N) # An empty array

# Arrays A, B, and C live in the "host" (CPU) memory.
# In order for our GPU to see and use them, we need
# to send a copy to the device. 
device_A = cuda.to_device(A)
device_B = cuda.to_device(B)
device_C = cuda.to_device(C)

# Define how many threads we would like to use per
# block that we request. This should always be a
# multiple of 32, which is the number of threads in
# a warp. If you instead requested 511 threads per
# block, then the GPU would still allocate 16 warps
# (16x32=512 threads) per block, but only 1 of the
# threads would receive no instructions at all.
threadsperblock=512

# We request to use some number of blocks such that
# exactly 1 thread corresponds to 1 index of our A,
# B, and C arrays. We round the number up so that
# we don't accidentally request too few blocks.
blockspergrid = math.ceil(N/threadsperblock)

# Now, finally, we can call our device function:
add[blockspergrid,threadsperblock](
    device_A,
    device_B,
    device_C,
)

# "Synchronize" the results. That is, wait until all
# the threads have finished processing before
# gathering all the data together.
cuda.synchronize()

# Finally, copy the result from the device back to the
# host.
C = device_C.copy_to_host()

if all(C == 2):
    print("1+1=2. GPUs aren't so scary!")
else:
    print("Oh, no. 1+1!=2. GPUs are terrifying...")
