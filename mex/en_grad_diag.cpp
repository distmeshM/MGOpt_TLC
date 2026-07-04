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
* @brief Compute triangle area, gradient, and diagonal Hessian entries
* @param vert Triangle vertices (3 x 3 matrix, one vertex per column)
* @param r Lift variables (Vector3d)
* @param[out] area Triangle area
* @param[out] grad Gradient (3 x 3 matrix)
* @param[out] diagHess Diagonal of the Hessian (length 3 * vDim)
*/
void liftedTriAreaGradHessian(const MatrixXd& vert, const Vector3d& r,
    double& area, MatrixXd& grad, VectorXd& diagHess, int nlhs) {

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

    // compute area
    area = HeronTriArea(d1, d2, d3);

    if (nlhs >= 2) {
        double g1 = d2 + d3 - d1;
        double g2 = d3 + d1 - d2;
        double g3 = d1 + d2 - d3;

        auto ge1 = g1 * e1;
        auto ge2 = g2 * e2;
        auto ge3 = g3 * e3;

        auto av1 = ge3 - ge2;
        auto av2 = ge1 - ge3;
        auto av3 = ge2 - ge1;

        // gradient has the same layout as vert
        grad.resize(vert.rows(), vert.cols());
        grad.col(0) = av1;
        grad.col(1) = av2;
        grad.col(2) = av3;

        double s = 1.0 / (8.0 * area);
        grad *= s;

        if (nlhs >= 3) {
            // Hessian term 1: Laplacian part
            Matrix3d Lap;
            Lap << g2 + g3, -g3,      -g2,
                   -g3,  g1 + g3,    -g1,
                   -g2,      -g1, g1 + g2;
            Lap *= s;

            MatrixXd Hess1(3 * vDim, 3 * vDim);
            MatrixXd I = MatrixXd::Identity(vDim, vDim);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    Hess1.block(i * vDim, j * vDim, vDim, vDim) = Lap(i, j) * I;
                }
            }

            // Hessian term 2: edge-based outer products
            MatrixXd E11 = e1 * e1.transpose();
            MatrixXd E12 = e1 * e2.transpose();
            MatrixXd E13 = e1 * e3;
            MatrixXd E22 = e2 * e2.transpose();
            MatrixXd E33 = e3 * e3.transpose();
            MatrixXd E231 = (e2 - e3) * e1.transpose();
            MatrixXd E312 = (e3 - e1) * e2.transpose();

            MatrixXd Hess2(3 * vDim, 3 * vDim);
            Hess2.block(0, 0, vDim, vDim) = E11;
            Hess2.block(0, vDim, vDim, vDim) = E13 + E231;
            Hess2.block(0, 2 * vDim, vDim, vDim) = E12 - E231;
            Hess2.block(vDim, vDim, vDim, vDim) = E22;
            Hess2.block(vDim, 2 * vDim, vDim, vDim) = E12.transpose() + E312;
            Hess2.block(2 * vDim, 2 * vDim, vDim, vDim) = E33;

            Hess2.block(vDim, 0, vDim, vDim) =
                Hess2.block(0, vDim, vDim, vDim).transpose();
            Hess2.block(2 * vDim, 0, vDim, vDim) =
                Hess2.block(0, 2 * vDim, vDim, vDim).transpose();
            Hess2.block(2 * vDim, vDim, vDim, vDim) =
                Hess2.block(vDim, 2 * vDim, vDim, vDim).transpose();

            s *= 2.0; // 1 / (4 * area)
            Hess2 *= s;

            // Hessian term 3: gradient outer products
            MatrixXd Hess3(3 * vDim, 3 * vDim);
            Hess3.block(0, 0, vDim, vDim) = av1 * av1.transpose();
            Hess3.block(0, vDim, vDim, vDim) = av1 * av2.transpose();
            Hess3.block(0, 2 * vDim, vDim, vDim) = av1 * av3.transpose();
            Hess3.block(vDim, vDim, vDim, vDim) = av2 * av2.transpose();
            Hess3.block(vDim, 2 * vDim, vDim, vDim) = av2 * av3.transpose();
            Hess3.block(2 * vDim, 2 * vDim, vDim, vDim) = av3 * av3.transpose();

            Hess3.block(vDim, 0, vDim, vDim) =
                Hess3.block(0, vDim, vDim, vDim).transpose();
            Hess3.block(2 * vDim, 0, vDim, vDim) =
                Hess3.block(0, 2 * vDim, vDim, vDim).transpose();
            Hess3.block(2 * vDim, vDim, vDim, vDim) =
                Hess3.block(vDim, 2 * vDim, vDim, vDim).transpose();

            s = s * s / (4.0 * area); // 1 / (64 * area^3)
            Hess3 *= s;

            // Full Hessian and extract diagonal
            MatrixXd Hess(3 * vDim, 3 * vDim);
            Hess = Hess1 - Hess2 - Hess3;
            diagHess = Hess.diagonal();
        }
    }
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
    int gradSize = vDim * 3;

    static Map<MatrixXd> empty_map(nullptr, 0, 0);

    // Output 1: areas (1 x nFaces)
    plhs[0] = mxCreateDoubleMatrix(1, nFaces, mxREAL);
    Map<RowVectorXd> areas(mxGetPr(plhs[0]), nFaces);

    // Output 2: gradients (gradSize x nFaces)
    if (nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(gradSize, nFaces, mxREAL);
    }
    Map<MatrixXd>& grads = (nlhs >= 2)
        ? *(new Map<MatrixXd>(mxGetPr(plhs[1]), gradSize, nFaces))
        : empty_map;

    // Output 3: diagonal Hessian (gradSize x nFaces)
    Map<MatrixXd>& hessians = empty_map;
    if (nlhs >= 3) {
        plhs[2] = mxCreateDoubleMatrix(gradSize, nFaces, mxREAL);
        hessians = *(new Map<MatrixXd>(mxGetPr(plhs[2]), gradSize, nFaces));
    }

    // Reorder indices: [x1_z, x2_z, x3_z, x1_w, x2_w, x3_w]
    Eigen::VectorXi reorder_indices(6);
    reorder_indices << 0, 2, 4, 1, 3, 5;

    // Parallel computation
#pragma omp parallel for
    for (int i = 0; i < nFaces; ++i) {
        int i1 = F(0, i) - 1;
        int i2 = F(1, i) - 1;
        int i3 = F(2, i) - 1;

        MatrixXd vert(vDim, 3);
        vert.col(0) = V.col(i1);
        vert.col(1) = V.col(i2);
        vert.col(2) = V.col(i3);

        Vector3d r = restD.col(i);

        double area;
        MatrixXd grad;
        VectorXd diagHess;
        liftedTriAreaGradHessian(vert, r, area, grad, diagHess, nlhs);

        // Store results
        areas(i) = area;

        if (nlhs >= 2) {
            VectorXd grad_reordered(grad.size());
            for (int j = 0; j < 6; ++j) {
                grad_reordered(j) = grad(reorder_indices(j));
            }
            grads.col(i) = Map<VectorXd>(grad_reordered.data(), gradSize);
        }

        if (nlhs >= 3) {
            VectorXd hess_reordered(diagHess.size());
            for (int j = 0; j < 6; ++j) {
                hess_reordered(j) = diagHess(reorder_indices(j));
            }
            hessians.col(i) = Map<VectorXd>(hess_reordered.data(), gradSize);
        }
    }
}