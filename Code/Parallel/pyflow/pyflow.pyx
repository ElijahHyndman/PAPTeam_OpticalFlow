# distutils: language = c++
# distutils: sources = src/Coarse2FineFlowWrapper.cpp
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
cimport numpy as np
cimport coarse2Fine
from libcpp.map cimport map
from libcpp.string cimport string

# Author: Deepak Pathak (c) 2016


'''
    Call order of Coarse2FineFlow from Python, to Cython, to C++. enumerated as checklist for
      editing the function calls.
    Python level: pyflow.coarse2fine_flow
      Cython level: pyflow.coarse2fine_flow -> pyflow.Coarse2FineFlowWrapper
        C++ level: pyflow.Coarse2FineFlowWrapper -> Coase2FineFlowWrapper.h.Coarse2FineFlowWrapper
        C++ level: Coase2FineFlowWrapper.h.Coarse2FineFlowWrapper -> Coarse2FineFlowWrapper.cpp.Coarse2FineFlowWrapper
        C++ level: Coase2FineFlowWrapper.cpp.Coarse2FineFlowWrapper -> OpticalFlow.h.Coarse2FineFlow
        C++ level: OpticalFlow.h.Coarse2FineFlow -> OpticalFlow.cpp.Coarse2FineFlow
'''

# Python only function that calls the wrapper function
def coarse2fine_flow(   np.ndarray[double, ndim=3, mode="c"] Im1 not None,
                        np.ndarray[double, ndim=3, mode="c"] Im2 not None,
                        int pyramidLevels, int nCores):
    """
    Input Format:ds
      :return: double * vx, double * vy, double * warpI2,
      :param: const double * Im1 (range [0,1]), const double * Im2 (range [0,1]),
      :param: double alpha (1), double ratio (0.5), int minWidth (40),
      :param: int nOuterFPIterations (3), int nInnerFPIterations (1),
      :param: int nSORIterations (20),
      :param: int colType (0 or default:RGB, 1:GRAY)
    Images Format: (h,w,c): float64: [0,1]
    """
    cdef int h = Im1.shape[0]
    cdef int w = Im1.shape[1]
    cdef int c = Im1.shape[2]
    cdef np.ndarray[double, ndim=2, mode="c"] vx = \
        np.ascontiguousarray(np.zeros((h, w), dtype=np.float64))
    cdef np.ndarray[double, ndim=2, mode="c"] vy = \
        np.ascontiguousarray(np.zeros((h, w), dtype=np.float64))
    cdef np.ndarray[double, ndim=3, mode="c"] warpI2 = \
        np.ascontiguousarray(np.zeros((h, w, c), dtype=np.float64))
    Im1 = np.ascontiguousarray(Im1)
    Im2 = np.ascontiguousarray(Im2)

    '''
        The only way I could pass the time measurements from the cpp code is to
          save it as a map<string,string>, convert it to a Cython map[string,string]
          and save each entry into a python dictionary
    '''
    # Calculate Optical Flow and retreive timer map
    cdef map[string,string] TIMER_AS_MAP
    TIMER_AS_MAP = coarse2Fine.Coarse2FineFlowWrapper(&vx[0, 0], &vy[0, 0], &warpI2[0, 0, 0],
                            &Im1[0, 0, 0], &Im2[0, 0, 0],
                            pyramidLevels, nCores,
                            h, w, c)
    # Convert the ('Phase','time') strings to utf-8 encoding
    TIMER_AS_DICTIONARY={ PAIR.first.decode('utf-8') : PAIR.second.decode('utf-8') for PAIR in TIMER_AS_MAP}

    return TIMER_AS_DICTIONARY, vx, vy, warpI2
