# setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='icache_module',
    ext_modules=[
        CUDAExtension(
            'icache_module',
            ['icache_binder.cu'],
            extra_compile_args={'nvcc': ['-O2']} 
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


'''
cmd:
python setup.py install

Usage:
import icache_module
icache_module.flush_icache()
'''