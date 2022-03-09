# -*- coding: utf-8 -*-
"""
generate and compile cython code
"""
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
ext_modules=[
    Extension("neuron_simulation_lib",
              ["neuron_simulation_lib.pyx"],
              language="c++", 
              libraries=["m"],
              extra_compile_args = ["-std=c++11", "-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp'],
              include_dirs=[numpy.get_include()]
              ) 
]

setup( 
  name = "neuron_simulation_lib",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()]
)



