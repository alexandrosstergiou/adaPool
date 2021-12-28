import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import adapool_cuda
from adaPool import adapool1d, adapool2d, adapool3d, AdaPool1d, AdaPool2d, AdaPool3d, adaunpool, EDSCWPool1d, EDSCWPool2d, EDSCWPool3d, EMPool1d, EMPool2d, EMPool3d

import timeit
import traceback

import sys


print('\033[\033[38;2;50;50;50;48;2;85;217;192m' + ' = = = Checks for float16 = = = ' + '\033[0m')

x_1d = torch.rand((1, 1, 8), device='cuda:0').half()
beta_1d = (4)

x_2d = torch.rand((1, 1, 8, 8), device='cuda:0').half()
beta_2d = (4,4)

x_3d = torch.rand((1, 1, 8, 8, 8), device='cuda:0').half()
beta_3d = (4,4,4)


print('\033[38;2;77;216;173m' + '--- Performing checks for forward ---' + '\033[0m')


print('\033[38;2;199;246;236m' + '> Checking 1D ...' + '\033[0m')
k=4
s=4
p_1d = AdaPool1d(kernel_size=k, beta=(1), stride=s, return_mask=True, device='cuda:0')
_ ,mask = p_1d(x_1d)
print('kernel size:',k,'stride:',s,'\n mask:',mask[0].data)

p_2d = AdaPool2d(kernel_size=k, beta=(1,1), stride=s, return_mask=True, device='cuda:0')
_ ,mask = p_2d(x_2d)
print('kernel size:',k,'stride:',s,'\n mask:',mask[0].data)

p_3d = AdaPool3d(kernel_size=k, beta=(1,1,1), stride=s, return_mask=True, device='cuda:0')
_ ,mask = p_3d(x_3d)
print('kernel size:',k,'stride:',s,'\n mask:',mask[0].data)

try:
    p_1d = AdaPool1d(dtype=x_1d.dtype,device=x_1d.get_device(),beta=beta_1d)
    pool_1d = p_1d(x_1d)

    p_1d = EDSCWPool1d()
    pool_1d = p_1d(x_1d)

    p_1d = EMPool1d()
    pool_1d = p_1d(x_1d)

    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)

print('\033[38;2;199;246;236m' + '> Checking 2D ...' + '\033[0m')
try:
    p_2d = AdaPool2d(dtype=x_2d.dtype,device=x_2d.get_device(),beta=beta_2d)
    pool_2d = p_2d(x_2d)

    p_2d = EDSCWPool2d()
    pool_2d = p_2d(x_2d)

    p_2d = EMPool2d()
    pool_2d = p_2d(x_2d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)

print('\033[38;2;199;246;236m' + '> Checking 3D ...' + '\033[0m')
try:
    p_3d = AdaPool3d(dtype=x_3d.dtype,device=x_3d.get_device(),beta=beta_3d)
    pool_3d = p_3d(x_3d)

    p_3d = EDSCWPool3d()
    pool_3d = p_3d(x_3d)

    p_3d = EMPool3d()
    pool_3d = p_3d(x_3d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


x_1d.requires_grad = True
x_2d.requires_grad = True
x_3d.requires_grad = True


print('\033[38;2;77;216;173m' + '--- Performing checks for backward ---' + '\033[0m')


print('\033[38;2;199;246;236m' + '> Checking 1D ...' + '\033[0m')
try:
    p_1d = AdaPool1d(dtype=x_1d.dtype,device=x_1d.get_device(),beta=beta_1d)
    p_1d(x_1d).pow(2).mean().backward()

    p_1d = EDSCWPool1d()
    p_1d(x_1d).pow(2).mean().backward()

    p_1d = EMPool1d()
    p_1d(x_1d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


print('\033[38;2;199;246;236m' + '> Checking 2D ...' + '\033[0m')
try:
    p_2d = AdaPool2d(dtype=x_2d.dtype,device=x_2d.get_device(),beta=beta_2d)
    p_2d(x_2d).pow(2).mean().backward()

    p_2d = EDSCWPool2d()
    p_2d(x_2d).pow(2).mean().backward()

    p_2d = EMPool2d()
    p_2d(x_2d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)

print('\033[38;2;199;246;236m' + '> Checking 3D ...' + '\033[0m')
try:
    p_3d = AdaPool3d(dtype=x_3d.dtype,device=x_3d.get_device(),beta=beta_3d)
    p_3d(x_3d).pow(2).mean().backward()

    p_3d = EDSCWPool3d()
    p_3d(x_3d).pow(2).mean().backward()

    p_3d = EMPool3d()
    p_3d(x_3d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)





print('\033[38;2;50;50;50;48;2;85;217;192m' + ' = = = Checks for float32 = = = ' + '\033[0m')

x_1d = torch.rand((4, 16, 56), device='cuda:0').float()
beta_1d = (28)

x_2d = torch.rand((4, 16, 56, 56), device='cuda:0').float()
beta_2d = (28,28)

x_3d = torch.rand((4, 16, 4, 56, 56), device='cuda:0').float()
beta_3d = (2,28,28)


print('\033[38;2;77;216;173m' + '--- Performing checks for forward ---' + '\033[0m')


print('\033[38;2;199;246;236m' + '> Checking 1D ...' + '\033[0m')

try:
    p_1d = AdaPool1d(dtype=x_1d.dtype,device=x_1d.get_device(),beta=beta_1d)
    pool_1d = p_1d(x_1d)

    p_1d = EDSCWPool1d()
    pool_1d = p_1d(x_1d)

    p_1d = EMPool1d()
    pool_1d = p_1d(x_1d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)

print('\033[38;2;199;246;236m' + '> Checking 2D ...' + '\033[0m')
try:
    p_2d = AdaPool2d(dtype=x_2d.dtype,device=x_2d.get_device(),beta=beta_2d)
    pool_2d = p_2d(x_2d)

    p_2d = EDSCWPool2d()
    pool_2d = p_2d(x_2d)

    p_2d = EMPool2d()
    pool_2d = p_2d(x_2d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)

print('\033[38;2;199;246;236m' + '> Checking 3D ...' + '\033[0m')
try:
    p_3d = AdaPool3d(dtype=x_3d.dtype,device=x_3d.get_device(),beta=beta_3d)
    pool_3d = p_3d(x_3d)

    p_3d = EDSCWPool3d()
    pool_3d = p_3d(x_3d)

    p_3d = EMPool3d()
    pool_3d = p_3d(x_3d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


x_1d.requires_grad = True
x_2d.requires_grad = True
x_3d.requires_grad = True

print('\033[38;2;77;216;173m' + '--- Performing checks for backward ---' + '\033[0m')


print('\033[38;2;199;246;236m' + '> Checking 1D ...' + '\033[0m')
try:
    p_1d = AdaPool1d(dtype=x_1d.dtype,device=x_1d.get_device(),beta=beta_1d)
    p_1d(x_1d).pow(2).mean().backward()

    p_1d = EDSCWPool1d()
    p_1d(x_1d).pow(2).mean().backward()

    p_1d = EMPool1d()
    p_1d(x_1d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


print('\033[38;2;199;246;236m' + '> Checking 2D ...' + '\033[0m')
try:
    p_2d = AdaPool2d(dtype=x_2d.dtype,device=x_2d.get_device(),beta=beta_2d)
    p_2d(x_2d).pow(2).mean().backward()

    p_2d = EDSCWPool2d()
    p_2d(x_2d).pow(2).mean().backward()

    p_2d = EMPool2d()
    p_2d(x_2d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


print('\033[38;2;199;246;236m' + '> Checking 3D ...' + '\033[0m')
try:
    p_3d = AdaPool3d(dtype=x_3d.dtype,device=x_3d.get_device(),beta=beta_3d)
    p_3d(x_3d).pow(2).mean().backward()

    p_3d = EDSCWPool3d()
    p_3d(x_3d).pow(2).mean().backward()

    p_3d = EMPool3d()
    p_3d(x_3d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)





print('\033[38;2;50;50;50;48;2;85;217;192m' + ' = = = Checks for float64 = = = ' + '\033[0m')

x_1d = torch.rand((4, 16, 56), device='cuda:0').double()
beta_1d = (28)

x_2d = torch.rand((4, 16, 56, 56), device='cuda:0').double()
beta_2d = (28,28)

x_3d = torch.rand((4, 16, 4, 56, 56), device='cuda:0').double()
beta_3d = (2,28,28)


print('\033[38;2;77;216;173m' + '--- Performing checks for forward ---' + '\033[0m')


print('\033[38;2;199;246;236m' + '> Checking 1D ...' + '\033[0m')

try:
    p_1d = AdaPool1d(dtype=x_1d.dtype,device=x_1d.get_device(),beta=beta_1d)
    pool_1d = p_1d(x_1d)

    p_1d = EDSCWPool1d()
    pool_1d = p_1d(x_1d)

    p_1d = EMPool1d()
    pool_1d = p_1d(x_1d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)

print('\033[38;2;199;246;236m' + '> Checking 2D ...' + '\033[0m')
try:
    p_2d = AdaPool2d(dtype=x_2d.dtype,device=x_2d.get_device(),beta=beta_2d)
    pool_2d = p_2d(x_2d)

    p_2d = EDSCWPool2d()
    pool_2d = p_2d(x_2d)

    p_2d = EMPool2d()
    pool_2d = p_2d(x_2d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


print('\033[38;2;199;246;236m' + '> Checking 3D ...' + '\033[0m')
try:
    p_3d = AdaPool3d(dtype=x_3d.dtype,device=x_3d.get_device(),beta=beta_3d)
    pool_3d = p_3d(x_3d)

    p_3d = EDSCWPool3d()
    pool_3d = p_3d(x_3d)

    p_3d = EMPool3d()
    pool_3d = p_3d(x_3d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


x_1d.requires_grad = True
x_2d.requires_grad = True
x_3d.requires_grad = True

print('\033[38;2;77;216;173m' + '--- Performing checks for backward ---' + '\033[0m')


print('\033[38;2;199;246;236m' + '> Checking 1D ...' + '\033[0m')
try:
    p_1d = AdaPool1d(dtype=x_1d.dtype,device=x_1d.get_device(),beta=beta_1d)
    p_1d(x_1d).pow(2).mean().backward()

    p_1d = EDSCWPool1d()
    p_1d(x_1d).pow(2).mean().backward()

    p_1d = EMPool1d()
    p_1d(x_1d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


print('\033[38;2;199;246;236m' + '> Checking 2D ...' + '\033[0m')
try:
    p_2d = AdaPool2d(dtype=x_2d.dtype,device=x_2d.get_device(),beta=beta_2d)
    p_2d(x_2d).pow(2).mean().backward()

    p_2d = EDSCWPool2d()
    p_2d(x_2d).pow(2).mean().backward()

    p_2d = EMPool2d()
    p_2d(x_2d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


print('\033[38;2;199;246;236m' + '> Checking 3D ...' + '\033[0m')
try:
    p_3d = AdaPool3d(dtype=x_3d.dtype,device=x_3d.get_device(),beta=beta_3d)
    p_3d(x_3d).pow(2).mean().backward()

    p_3d = EDSCWPool3d()
    p_3d(x_3d).pow(2).mean().backward()

    p_3d = EMPool3d()
    p_3d(x_3d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)

print('\033[38;2;50;50;50;48;2;85;217;192m' + ' = = = Checks for Unpool (float32) = = = ' + '\033[0m')

x_1d = torch.rand((4, 16, 40), device='cuda:0').float()
x_2d = torch.rand((4, 16, 40, 40), device='cuda:0').float()
x_3d = torch.rand((4, 16, 4, 40, 40), device='cuda:0').float()
x_1d.requires_grad = True
x_2d.requires_grad = True
x_3d.requires_grad = True


beta_1d = (20)

beta_2d = (20,20)

beta_3d = (2,20,20)


p_1d = AdaPool1d(return_mask=True, dtype=x_1d.dtype,device=x_1d.get_device(),beta=beta_1d)
p_2d = AdaPool2d(return_mask=True, dtype=x_2d.dtype,device=x_2d.get_device(),beta=beta_2d)
p_3d = AdaPool3d(return_mask=True, dtype=x_3d.dtype,device=x_3d.get_device(),beta=beta_3d)


tmp, mask = p_1d(x_1d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        adaunpool(tmp, mask)
print('\033[38;2;199;246;236m' +'AdaUnpool1d'+ '\033[0m')
print('\033[92m' + '> PASSED' + '\033[0m')
print(prof.key_averages())


try:
    tmp, mask = p_2d(x_2d)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(100):
            adaunpool(tmp, mask)
    print('\033[38;2;199;246;236m' +'AdaUnpool2d'+ '\033[0m')
    print('\033[92m' + '> PASSED' + '\033[0m')
    print(prof.key_averages())
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


try:
    tmp, mask = p_3d(x_3d)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(100):
            adaunpool(tmp, mask)
    print('\033[38;2;199;246;236m' +'AdaUnpool3d'+ '\033[0m')
    print('\033[92m' + '> PASSED' + '\033[0m')
    print(prof.key_averages())
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)
    traceback.print_tb(e.__traceback__)


print('\n'+'\033[38;2;50;50;50;48;2;199;246;236m' + '--- Float point precision Forward/Backward tests completed ---' + '\033[0m'+'\n')



print('\033[38;2;50;50;50;48;2;85;217;192m' + '= = = Profiling checks = = =' + '\033[0m')

x_1d = torch.rand((4, 16, 80), device='cuda:0').float()
x_2d = torch.rand((4, 16, 80, 80), device='cuda:0').float()
x_3d = torch.rand((4, 16, 8, 80, 80), device='cuda:0').float()
x_1d.requires_grad = True
x_2d.requires_grad = True
x_3d.requires_grad = True

beta_1d = (40)

beta_2d = (40,40)

beta_3d = (4,40,40)

p_1d = AdaPool1d(dtype=x_1d.dtype,device=x_1d.get_device(), beta=beta_1d)
p_2d = AdaPool2d(dtype=x_2d.dtype,device=x_2d.get_device(), beta=beta_2d)
p_3d = AdaPool3d(dtype=x_3d.dtype,device=x_3d.get_device(), beta=beta_3d)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_1d(x_1d)
print('\033[38;2;199;246;236m' +'AdaPool1d [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_1d_cuda = ''.join(str(prof).split('\n')[-3:])
_tt = p_1d(x_1d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        tmp = p_1d(x_1d)
        tmp.backward(tmp,retain_graph=True)
print('\033[38;2;199;246;236m' +'AdaPool1d [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_1d_cuda = ' '.join(str(prof).split('\n')[-3:])


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_2d(x_2d)
print('\033[38;2;199;246;236m' +'AdaPool2d [foward]'+ '\033[0m')
time_f_2d_cuda = ''.join(str(prof).split('\n')[-3:])
print(prof.key_averages())
_tt = p_2d(x_2d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        tmp = p_2d(x_2d)
        tmp.backward(tmp,retain_graph=True)
print('\033[38;2;199;246;236m' +'AdaPool2d [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_2d_cuda = ' '.join(str(prof).split('\n')[-3:])


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_3d(x_3d)
print('\033[38;2;199;246;236m' +'AdaPool3d [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_3d_cuda = ''.join(str(prof).split('\n')[-3:])
_tt = p_3d(x_3d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_3d(x_3d).backward(_tt)
print('\033[38;2;199;246;236m' +'AdaPool3d [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_3d_cuda = ' '.join(str(prof).split('\n')[-3:])





p_1d = EDSCWPool1d()
p_2d = EDSCWPool2d()
p_3d = EDSCWPool3d()

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_1d(x_1d)
print('\033[38;2;199;246;236m' +'EDSCWPool1d [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_1d_cuda_EDSCW = ''.join(str(prof).split('\n')[-3:])
_tt = p_1d(x_1d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        tmp = p_1d(x_1d)
        tmp.backward(tmp,retain_graph=True)
print('\033[38;2;199;246;236m' +'EDSCWPool1d [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_1d_cuda_EDSCW = ' '.join(str(prof).split('\n')[-3:])


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_2d(x_2d)
print('\033[38;2;199;246;236m' +'EDSCWPool2d [foward]'+ '\033[0m')
time_f_2d_cuda_EDSCW = ''.join(str(prof).split('\n')[-3:])
print(prof.key_averages())
_tt = p_2d(x_2d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        tmp = p_2d(x_2d)
        tmp.backward(tmp,retain_graph=True)
print('\033[38;2;199;246;236m' +'EDSCWPool2d [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_2d_cuda_EDSCW = ' '.join(str(prof).split('\n')[-3:])


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_3d(x_3d)
print('\033[38;2;199;246;236m' +'EDSCWPool3d [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_3d_cuda_EDSCW = ''.join(str(prof).split('\n')[-3:])
_tt = p_3d(x_3d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_3d(x_3d).backward(_tt)
print('\033[38;2;199;246;236m' +'EDSCWPool3d [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_3d_cuda_EDSCW = ' '.join(str(prof).split('\n')[-3:])





p_1d = EMPool1d()
p_2d = EMPool2d()
p_3d = EMPool3d()

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_1d(x_1d)
print('\033[38;2;199;246;236m' +'EMPool1d [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_1d_cuda_em = ''.join(str(prof).split('\n')[-3:])
_tt = p_1d(x_1d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        tmp = p_1d(x_1d)
        tmp.backward(tmp,retain_graph=True)
print('\033[38;2;199;246;236m' +'EMPool1d [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_1d_cuda_em = ' '.join(str(prof).split('\n')[-3:])


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_2d(x_2d)
print('\033[38;2;199;246;236m' +'EMPool2d [foward]'+ '\033[0m')
time_f_2d_cuda_em = ''.join(str(prof).split('\n')[-3:])
print(prof.key_averages())
_tt = p_2d(x_2d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        tmp = p_2d(x_2d)
        tmp.backward(tmp,retain_graph=True)
print('\033[38;2;199;246;236m' +'EMPool2d [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_2d_cuda_em = ' '.join(str(prof).split('\n')[-3:])


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_3d(x_3d)
print('\033[38;2;199;246;236m' +'EMPool3d [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_3d_cuda_em = ''.join(str(prof).split('\n')[-3:])
_tt = p_3d(x_3d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        p_3d(x_3d).backward(_tt)
print('\033[38;2;199;246;236m' +'EMPool3d [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_3d_cuda_em = ' '.join(str(prof).split('\n')[-3:])






print('\n'+'\033[38;2;199;246;236m' +'-------------------------------'+ '\033[0m')
print('\033[38;2;50;50;50;48;2;85;217;192m' +'AdaPool1d [forward + backward]'+ '\033[0m')
print(time_b_1d_cuda, 'for 100 iterations.')
print('\033[38;2;50;50;50;48;2;85;217;192m' +'EDSCWPool1d [forward + backward]'+ '\033[0m')
print(time_b_1d_cuda_EDSCW, 'for 100 iterations.')
print('\033[38;2;50;50;50;48;2;85;217;192m' +'EMPool1d [forward + backward]'+ '\033[0m')
print(time_b_1d_cuda_em, 'for 100 iterations.')
print('\n'+'\033[38;2;199;246;236m' +'-------------------------------'+ '\033[0m')
print('\033[38;2;50;50;50;48;2;85;217;192m' +'AdaPool2d [forward + backward]'+ '\033[0m')
print(time_b_2d_cuda, 'for 100 iterations.')
print('\033[38;2;50;50;50;48;2;85;217;192m' +'EDSCWPool2d [forward + backward]'+ '\033[0m')
print(time_b_2d_cuda_EDSCW, 'for 100 iterations.')
print('\033[38;2;50;50;50;48;2;85;217;192m' +'EMPool2d [forward + backward]'+ '\033[0m')
print(time_b_2d_cuda_em, 'for 100 iterations.')
print('\n'+'\033[38;2;199;246;236m' +'-------------------------------'+ '\033[0m')
print('\033[38;2;50;50;50;48;2;85;217;192m' +'AdaPool3d [forward + backward]'+ '\033[0m')
print(time_b_3d_cuda, 'for 100 iterations.')
print('\033[38;2;50;50;50;48;2;85;217;192m' +'EDSCWPool3d [forward + backward]'+ '\033[0m')
print(time_b_3d_cuda_EDSCW, 'for 100 iterations.')
print('\033[38;2;50;50;50;48;2;85;217;192m' +'EMPool3d [forward + backward]'+ '\033[0m')
print(time_b_3d_cuda_em, 'for 100 iterations.')
print('\n'+'\033[38;2;199;246;236m' +'-------------------------------'+ '\033[0m')

print('\n'+'\033[38;2;50;50;50;48;2;199;246;236m' + '--- Tests finished ---' + '\033[0m')
