from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules=[
    CUDAExtension('dapalib', [
        'association.cpp',
        'gpu/nmsBase.cu',
        'gpu/bodyPartConnectorBase.cu',
        'gpu/cuda_cal.cu',
        ], 
        include_dirs=['/usr/local/cuda-10.1/include', '/usr/local/lib'] ,   # '/usr/include/eigen3'   '/usr/local/cuda-11.1/include'
    ),          
]

setup(
    name='dapalib',
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
