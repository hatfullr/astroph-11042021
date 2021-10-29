# In this example we will learn how we can use the
# GPU to color pixels in a matplotlib plot.

import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
import math
from time import time

def cpu(ax,xpixels,ypixels,center,radius,color,extent):
    # Create a 2D grid with RGB values
    data = np.full((xpixels,ypixels,3),np.nan)

    xmin,xmax,ymin,ymax = extent
    dx = (xmax-xmin)/float(xpixels)
    dy = (ymax-ymin)/float(ypixels)
    
    radius2 = radius**2
    for i in range(xpixels):
        xpos = xmin + i*dx + 0.5*dx # centered on pixel
        delta_x2 = (xpos-center[0])**2
        for j in range(ypixels):
            ypos = ymin + j*dy + 0.5*dy
            delta_y2 = (ypos-center[1])**2
            dr2 = delta_x2 + delta_y2
            if dr2 < radius2:
                data[i,j] = color
    
    ax.imshow(data,extent=extent)

def gpu(ax,xpixels,ypixels,center,radius,color,extent):
    data = np.full((xpixels,ypixels,3),np.nan)
    device_data = cuda.to_device(data)
    
    xmin,xmax,ymin,ymax = extent

    dx = (xmax-xmin)/float(xpixels)
    dy = (ymax-ymin)/float(ypixels)

    # We use a 2D grid this time. Instead of 512 threads
    # per block, we will use 16x16=256 threads. Remember,
    # the total number of threads per block must be a
    # multiple of 32 -- the size of a warp, and blocks
    # contain only so many threads! On my GPU, I have
    # access to exactly 32*32 = 1024 threads, so I can
    # choose (32,32), (64,16), (8,128), (1024,1), etc.
    # All these choices have similar performance, but
    # you can check for yourself which works best for your
    # machine.
    threadsperblock=(32,32)
    blockspergrid=(
        math.ceil(xpixels/threadsperblock[0]),
        math.ceil(ypixels/threadsperblock[1]),
    )
    kernel[blockspergrid,threadsperblock](
        xmin,
        ymin,
        dx,
        dy,
        center[0],
        center[1],
        radius**2,
        color,
        device_data,
    )
    cuda.synchronize()
    data = device_data.copy_to_host()
    
    ax.imshow(data,extent=extent)
    
# Sometimes GPU code is cleaner and easier to read than CPU code :)    
@cuda.jit
def kernel(xmin,ymin,dx,dy,centerx,centery,radius2,color,data):
    i,j = cuda.grid(2)
    if i < data.shape[0] and j < data.shape[1]:
        xpos = xmin + i*dx + 0.5*dx
        ypos = ymin + j*dy + 0.5*dy
        dr2 = (xpos-centerx)*(xpos-centerx) + (ypos-centery)*(ypos-centery)
        if dr2 < radius2:
            data[i,j] = color
    

# Create a Matplotlib figure with an associated axis
fig, ax = plt.subplots(ncols=2)
ax_gpu = ax[0]
ax_cpu = ax[1]

# Define our image's resolution
xpixels = 5000
ypixels = 5000

# Set the data limits to ensure an equal aspect ratio
xmin, xmax = 0., 1.
ymin, ymax = 0., 1.
ax_cpu.set_xlim(xmin,xmax)
ax_cpu.set_ylim(ymin,ymax)
ax_gpu.set_xlim(xmin,xmax)
ax_gpu.set_ylim(ymin,ymax)

extent = [xmin,xmax,ymin,ymax]
center = np.array([0.5*(xmax+xmin),0.5*(ymax+ymin)])

color = (1,0,0) # Make a red circle
radius = 0.25

start = time()
gpu(ax_gpu,xpixels,ypixels,center,radius,color,extent)
gpu_time = time()-start
ax_gpu.set_title("GPU took %f seconds" % (gpu_time))

start = time()
cpu(ax_cpu,xpixels,ypixels,center,radius,color,extent)
cpu_time = time()-start
ax_cpu.set_title("CPU took %f seconds" % (cpu_time))

# On my machine, the GPU code runs 10x faster:
# "GPU took  1.601440 seconds"
# "CPU took 16.270854 seconds"

plt.show()
