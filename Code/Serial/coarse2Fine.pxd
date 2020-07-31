from libcpp.map cimport map
from libcpp.string cimport string

from cython.parallel cimport parallel
cimport openmp

openmp.omp_set_dynamic(1)
cdef extern from "src/Coarse2FineFlowWrapper.h":
    map[string,string] Coarse2FineFlowWrapper(  double * vx, double * vy, double * warpI2,
                                  const double * Im1, const double * Im2,
                                  int pyramidLevels,
                                  int h, int w, int c);
