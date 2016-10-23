from pycuda import autoinit
from pycuda.autoinit import context
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray


import numpy as np

with open('transpose3d.cu') as f:
     kernels = f.read()
mod = compiler.SourceModule(source=kernels, options=["-O2"], arch="sm_35", include_dirs=["/home/atrikut/projects/cuTranspose/src"])
func = mod.get_function("dev_transpose_210_in_place")
func.prepare('Pii')

A = np.random.rand(3, 3, 3)
A_gpu = gpuarray.to_gpu(A)

print(A_gpu)
func.prepared_call((1, 1, 3), (16, 16, 1), A_gpu.gpudata, 3, 3)
context.synchronize()
print(A_gpu)
