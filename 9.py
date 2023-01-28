import cupy as cp

arr=cp.ones(7*10**9,dtype=cp.uint8)
nn=cp.random.randint(1,10**9,size=10**8)
import cupy as cp
import time
from numba import cuda
start =time.time()                           
@cuda.jit
def my_kernel(arr,nn):
    i = cuda.grid(1)
    if i<nn.size:
       # if arr[i] ==1:
        arr[nn[i]]=0  
my_kernel[256,36](arr,nn)
x=cp.where(arr==0)
print(x[:10000])
print(time.time()-start)
