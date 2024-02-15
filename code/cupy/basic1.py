import numpy as np
import cupy as cp 
import torch
import time

start_time = time.time()
for _ in range(1000):
    np_mat = np.random.rand(1000, 1000)
    # tensor_mat = torch.as_tensor(np_mat).cuda()
    np_mat2 = np_mat @ np_mat
print(time.time() - start_time)
# 8.527007102966309


start_time = time.time()
for _ in range(1000):
    cp_mat = cp.random.rand(1000, 1000)
    cp_mat2 = cp_mat @ cp_mat

    # tensor_mat = torch.as_tensor(a, device="cuda")
print(time.time() - start_time)
a = np.random.rand(5)
start_time = time.time()
for _ in range(1000):
    tensor_mat = torch.as_tensor(a, device="cuda")
    tensor_mat2 = tensor_mat @ tensor_mat
print(time.time() - start_time)

# 0.28897643089294434
print(torch.cuda.is_available())