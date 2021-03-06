// This is a wrapper for Ce Liu's Coarse2Fine optical flow implementation.
// It converts the contiguous image array to the format needed by the optical
// flow code. Handling conversion in the wrapper makes the cythonization
// simpler.
// Author: Deepak Pathak (c) 2016
#include <Python.h>
#include <iostream>
#include <map>
using namespace std;

// override-include-guard
extern map<string,string> Coarse2FineFlowWrapper( double * vx, double * vy, double * warpI2,
                              const double * Im1, const double * Im2,
                              int pyramidLevels,
                              int h, int w, int c);
