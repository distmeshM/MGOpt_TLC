#include <vector>
#include <omp.h>
#include <mex.h>
#include <Eigen/Dense>
#include <transfer.h>

using namespace Eigen;

void accumArrayAtomic(const int* subs, const double* vals, double* result, int sizeSubs, int sizeRes) {
#pragma omp parallel for
    for (int i = 0; i < sizeSubs; ++i) {
        int idx = subs[i];
        double val = vals[i];
#pragma omp atomic
        result[idx] += val;
    }
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    const int* ids = (int*)mxGetData(prhs[0]);
    int nId = mxGetM(prhs[0]);
    const double* vals = mxGetPr(prhs[1]);
    int nRes = mxGetScalar(prhs[2]);

    plhs[0] = mxCreateDoubleMatrix(nRes, 1, mxREAL);
    double* res = mxGetPr(plhs[0]);
    std::fill(res, res + nRes, 0.0);
    accumArrayAtomic(ids, vals, res, nId, nRes);
}