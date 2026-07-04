#include <vector>
#include <omp.h>
#include <mex.h>
#include <Eigen/Dense>
#include <transfer.h>

using namespace Eigen;

/* *
* @brief Atomic accumulation into an array using OpenMP
* @param subs Indices to accumulate into
* @param vals Values to accumulate
* @param result Output array
* @param sizeSubs Number of index¨Cvalue pairs
* @param sizeRes Size of the result array (unused here)
*/
void accumArrayAtomic(const int* subs, const double* vals, double* result,
                      int sizeSubs, int sizeRes) {
#pragma omp parallel for
    for (int i = 0; i < sizeSubs; ++i) {
        int idx = subs[i];
        double val = vals[i];
#pragma omp atomic
        result[idx] += val;
    }
}

/* *
* @brief MATLAB mexFunction entry point
*/
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Inputs: y1 (nv*2), a (nv), c (nv)
    const double* y1 = mxGetPr(prhs[0]);
    const double* a  = mxGetPr(prhs[1]);
    const double* c  = mxGetPr(prhs[2]);

    int nv = mxGetM(prhs[0]) / 2;      // number of vertices
    const double* y2 = y1 + nv;        // second half of y1
    const double* b  = a  + nv;        // second half of a

    // Output: x (nv*2)
    plhs[0] = mxCreateDoubleMatrix(nv * 2, 1, mxREAL);
    double* x1 = mxGetPr(plhs[0]);
    double* x2 = x1 + nv;

    // Parallel solve of 2x2 linear systems
#pragma omp parallel for
    for (int i = 0; i < nv; i++) {
        double A = a[i];
        double B = b[i];
        double C = c[i];
        double Y1 = y1[i];
        double Y2 = y2[i];

        double D = A * B - C * C;  // determinant

        if (D <= 0) {
            mexErrMsgIdAndTxt("MyToolbox:solve2x2",
                              "Non-positive determinant encountered in 2x2 solve.");
        }

        double X1 = (B * Y1 - C * Y2) / D;
        double X2 = (A * Y2 - C * Y1) / D;

        x1[i] = X1;
        x2[i] = X2;
    }
}