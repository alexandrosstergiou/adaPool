from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='adaPool',
    version='0.1',
    description='CUDA-accelerated package for performing 1D/2D/3D adaPool and adaUpsample with PyTorch',
    author='Alexandros Stergiou',
    author_email='alexstergiou5@gmail.com',
    license='MIT',
    python_requires='>=3',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('adapool_cuda', [
            'CUDA/adapool_cuda.cpp',
            'CUDA/adapool_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })
