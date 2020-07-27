# Author: Deepak Pathak (c) 2016
# Edited: Elijah Hyndman

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals


from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from glob import glob
import numpy


sourcefiles = ['pyflow.pyx', ]
sourcefiles.extend(glob("src/*.cpp"))

# Setup for OpenMP compiling
#       Using this ext_modules setup will allow us to compile our cpp code
#       using OpenMP and the -fopenmp compiling flag
ext_modules=[
    # OpenMP compiling extension
    Extension(
        "pyflow",
        sourcefiles,
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="pyflow",
    version="1.0",
    description="Python wrapper for the Coarse2Fine Optical Flow code.",
    author="Deepak Pathak",
    ext_modules= cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
