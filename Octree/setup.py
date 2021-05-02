from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

octree_extension = Extension(
    name="octree",
    sources=["octree.pyx"],
    libraries=["mol.0.0.6"],
    library_dirs=["FromBU/minilibmol"],
    include_dirs=["FromBU/minilibmol", np.get_include()],
)

setup(
    name="octree",
    ext_modules=cythonize([octree_extension], compiler_directives={'language_level' : "3"})
)
