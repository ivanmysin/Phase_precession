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
    Extension("lib",
              ["lib.pyx"],
              language="c++", 
              libraries=["m"],
              extra_compile_args = ["-std=c++11", "-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp'],
              include_dirs=[numpy.get_include()]
              ) 
]

setup( 
  name = "lib",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()]
)

"""
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules = cythonize(Extension(
    "lib2",                # the extension name
     sources=["lib2.pyx"], # the Cython source and
                           # additional C++ source files
     language="c++",       # generate and compile C++ code
     extra_compile_args=["-std=c++11", "-O3", "-ffast-math", "-march=native", "-fopenmp"],
     extra_link_args=['-fopenmp'],
     libraries=["m"],
      )),
      cmdclass = {"build_ext": build_ext},)
"""


