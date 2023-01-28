
import time
import numpy as np
from numba import njit
from numba import cuda
@njit
def simpleSieve(limit):
	primes=[]
	mark =np.full(limit+1,True)
	p = 2
	while (p * p <= limit):
		if (mark[p] == True):
			for i in range(p * p, limit + 1, p):
				mark[i] = False
		p += 1
	for p in range(2, limit):
		if mark[p]==1:
			primes.append(p)
	return primes
start=time.time()
tar=2**601
primes=simpleSieve(2*10**5)
print(len(primes))
primorial =2*3*5*7*11*13*17*19*23*29*31*37*41*43*47*53*59*61*67*71*73*79*83*89*97*101*103
print(primorial)
offset = 30402250951007051
tarn = (tar+primorial)-(tar%primorial)
tarn
def inverse(hash,p):
     return pow(hash, p-2, p)
x=tarn+offset
fp=[]
for prime in primes:
  a = ((prime-(x%prime))*inverse(primorial,prime))%prime
  fp.append(a)
x=tarn+offset+2
fp2=[]
for prime in primes:
  a = ((prime-(x%prime))*inverse(primorial,prime))%prime
  fp2.append(a)
x=tarn+offset+6
fp3=[]
for prime in primes:
  a = ((prime-(x%prime))*inverse(primorial,prime))%prime
  fp3.append(a)
x=tarn+offset+8
fp4=[]
for prime in primes:
  a = ((prime-(x%prime))*inverse(primorial,prime))%prime
  fp4.append(a)
x=tarn+offset+12
fp5=[]
for prime in primes:
  a = ((prime-(x%prime))*inverse(primorial,prime))%prime
  fp5.append(a)
import cupy as cp
primes_gpu= cp.array(primes)
fp_gpu = cp.array(fp)
fp2_gpu = cp.array(fp2)
fp3_gpu = cp.array(fp3)
fp4_gpu = cp.array(fp4)
fp5_gpu = cp.array(fp5)
arr=cp.ones(5*10**9,dtype=cp.uint8)  
@cuda.jit
def my_kernel(arr,arr1,fp,fp2,fp3,fp4,fp5):
    i = cuda.grid(1)
    if i<arr1.size:
       # if arr[i] ==1:
        arr[fp[i]::arr1[i]]=0
        arr[fp2[i]::arr1[i]]=0
        arr[fp3[i]::arr1[i]]=0
        arr[fp4[i]::arr1[i]]=0
        arr[fp5[i]::arr1[i]]=0   
my_kernel[256,36](arr,primes_gpu,fp_gpu,fp2_gpu,fp3_gpu,fp4_gpu,fp5_gpu)
x=cp.where(arr==1)
print(x[:100])
abc=x[0].get()
print(abc.size) 
print(time.time()-start)
