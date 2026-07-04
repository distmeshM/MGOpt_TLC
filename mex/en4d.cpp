#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "mex.h"

using namespace Eigen;

/* *
* @brief Compute triangle area with lifted variables (numerically stable Heron formula)
* @param d1, d2, d3 squared edge lengths + lift variables r
* @return triangle area (returns 0 if inputs are invalid)
*/
double HeronTriArea(double d1, double d2, double d3) {
    // sort d1,d2,d3 so that a >= b >= c
    double a, b, c;
    if (d1 > d2) {
        a = d1;
        b = d2;
    }
    else {
        a = d2;
        b = d1;
    }
    c = d3;
    if (d3 > b) {
        c = b;
        b = d3;
        if (d3 > a) {
            b = a;
            a = d3;
        }
    }

    a = std::sqrt(a);
    b = std::sqrt(b);
    c = std::sqrt(c);

    return 0.25 * std::sqrt(std::abs(
        (a + (b + c)) *
        (c - (a - b)) *
        (c + (a - b)) *
        (a + (b - c))
    ));
}

/* *
* @brief Compute triangle area with lifted variables
* @param vert Triangle vertices (3 x 3 matrix, one vertex per column)
* @param r Lift variables (Vector3d)
* @param[out] area Triangle area
* @param nlhs Number of requested output arguments (unused here)
*/
void liftedTriAreaGradHessian(const MatrixXd& vert, const Vector3d& r,
    double& area, int nlhs) {

    auto v1 = vert.col(0);
    auto v2 = vert.col(1);
    auto v3 = vert.col(2);

    auto e1 = v2 - v3;
    auto e2 = v3 - v1;
    auto e3 = v1 - v2;

    double d1 = e1.squaredNorm() + r(0);
    double d2 = e2.squaredNorm() + r(1);
    double d3 = e3.squaredNorm() + r(2);

    // compute area
    area = HeronTriArea(d1, d2, d3);
}

/* *
* @brief MATLAB mexFunction entry point
*/
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Input validation
    if (nrhs != 3) {
        mexErrMsgTxt("Expected 3 input arguments: V, F, restD");
    }

    // Read inputs
    Map<const MatrixXd> V(mxGetPr(prhs[0]), mxGetM(prhs[0]), mxGetN(prhs[0]));
    Map<const MatrixXi> F((int*)mxGetPr(prhs[1]), mxGetM(prhs[1]), mxGetN(prhs[1]));
    Map<const MatrixXd> restD(mxGetPr(prhs[2]), 3, mxGetN(prhs[2]));

    // Prepare outputs
    int nFaces = F.cols();
    int vDim = V.rows();

    // Output 1: areas (1 x nFaces)
    plhs[0] = mxCreateDoubleMatrix(1, nFaces, mxREAL);
    Map<RowVectorXd> areas(mxGetPr(plhs[0]), nFaces);

    // Parallel computation
#pragma omp parallel for
    for (int i = 0; i < nFaces; ++i) {
        // Convert MATLAB 1-based indices to C++ 0-based
        int i1 = F(0, i) - 1;
        int i2 = F(1, i) - 1;
        int i3 = F(2, i) - 1;

        // Assemble vertex matrix (vDim x 3)
        MatrixXd vert(vDim, 3);
        vert.col(0) = V.col(i1);
        vert.col(1) = V.col(i2);
        vert.col(2) = V.col(i3);

        // Lift variables for this face
        Vector3d r = restD.col(i);

        // Compute area
        double area;
        liftedTriAreaGradHessian(vert, r, area, nlhs);

        // Store result
        areas(i) = area;
    }
}