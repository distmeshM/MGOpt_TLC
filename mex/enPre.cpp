#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "mex.h"

using namespace Eigen;


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

    int vDim = v1.size();

    //


}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 4) {
        mexErrMsgTxt("Expected 4 input arguments: V, F, D, restD");
    }

    Map<const MatrixXd> V(mxGetPr(prhs[0]), mxGetM(prhs[0]), mxGetN(prhs[0]));
    Map<const MatrixXi> F((int*)mxGetPr(prhs[1]), mxGetM(prhs[1]), mxGetN(prhs[1]));
    Map<const MatrixXd> D(mxGetPr(prhs[2]), mxGetM(prhs[2]), mxGetN(prhs[2]));
    Map<const MatrixXd> restD(mxGetPr(prhs[3]), mxGetM(prhs[3]), mxGetN(prhs[3]));

    int nFaces = F.cols();
    int vDim = V.rows();


    plhs[0] = mxCreateDoubleMatrix(3, nFaces, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(3, nFaces, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(3, nFaces, mxREAL);
    Map<MatrixXd> A(mxGetPr(plhs[0]), 3, nFaces);
    Map<MatrixXd> B(mxGetPr(plhs[1]), 3, nFaces);
    Map<MatrixXd> C(mxGetPr(plhs[2]), 3, nFaces);


#pragma omp parallel for
    for (int i = 0; i < nFaces; ++i) {
        int i1 = F(0, i) - 1;
        int i2 = F(1, i) - 1;
        int i3 = F(2, i) - 1;

        Vector3d r = restD.col(i);
        MatrixXd vert(vDim, 3);
        auto v1 = V.col(i1);
        auto v2 = V.col(i2);
        auto v3 = V.col(i3);

        auto w1 = D.col(i1);
        auto w2 = D.col(i2);
        auto w3 = D.col(i3);
        auto e1 = v2 - v3;
        auto e2 = v3 - v1;
        auto e3 = v1 - v2;
        auto E1 = w2 - w3;
        auto E2 = w3 - w1;
        auto E3 = w1 - w2;
        double d1 = e1.squaredNorm() + r(0);
        double d2 = e2.squaredNorm() + r(1);
        double d3 = e3.squaredNorm() + r(2);
        double D1 = E1.squaredNorm();
        double D2 = E2.squaredNorm();
        double D3 = E3.squaredNorm();
        double dt1 = 2 * E1.dot(e1);
        double dt2 = 2 * E2.dot(e2);
        double dt3 = 2 * E3.dot(e3);
        C(0, i) = d1; C(1, i) = d2; C(2, i) = d3;
        B(0, i) = dt1; B(1, i) = dt2; B(2, i) = dt3;
        A(0, i) = D1; A(1, i) = D2; A(2, i) = D3;
    }
}