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
* @brief Compute triangle area, gradient, and diagonal Hessian blocks
* @param vert Triangle vertices (3 x 3 matrix, one vertex per column)
* @param r Lift variables (Vector3d)
* @param[out] area Triangle area
* @param[out] grad Gradient (3 x 3 matrix)
* @param[out] diagHess1 First diagonal Hessian vector
* @param[out] diagHess2 Second diagonal/off-diagonal Hessian vector
*/
void liftedTriAreaGradHessian(const MatrixXd& vert, const Vector3d& r,
    double& area, MatrixXd& grad, MatrixXd& diagHess1, MatrixXd& diagHess2, int nlhs) {

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
            // Hessian term 1: Laplacian (diagonal only)
            Matrix3d Lap;
            Lap << g2 + g3, -g3,      -g2,
                   -g3,  g1 + g3,    -g1,
                   -g2,      -g1, g1 + g2;
            Lap *= s;

            VectorXd Diag11(3 * vDim);
            for (int i = 0; i < 3; ++i) {
                Diag11.segment(i * vDim, vDim).setConstant(Lap(i, i));
            }

            // Hessian term 2: outer-product contributions
            s *= 2.0; // 1 / (4 * area)
            MatrixXd E11 = s * e1 * e1.transpose();
            MatrixXd E22 = s * e2 * e2.transpose();
            MatrixXd E33 = s * e3 * e3.transpose();

            VectorXd Diag12(3 * vDim), Diag22(3 * vDim);
            Diag12 << E11(0, 0), E11(1, 1),
                      E22(0, 0), E22(1, 1),
                      E33(0, 0), E33(1, 1);
            Diag22 << E11(0, 1), E11(1, 0),
                      E22(0, 1), E22(1, 0),
                      E33(0, 1), E33(1, 0);

            // Hessian term 3: gradient outer products
            s = s * s / (4.0 * area); // 1 / (64 * area^3)
            MatrixXd F11 = s * av1 * av1.transpose();
            MatrixXd F22 = s * av2 * av2.transpose();
            MatrixXd F33 = s * av3 * av3.transpose();

            VectorXd Diag13(3 * vDim), Diag23(3 * vDim);
            Diag13 << F11(0, 0), F11(1, 1),
                      F22(0, 0), F22(1, 1),
                      F33(0, 0), F33(1, 1);
            Diag23 << F11(0, 1), F11(1, 0),
                      F22(0, 1), F22(1, 0),
                      F33(0, 1), F33(1, 0);

            // Final diagonal Hessian vectors
            diagHess1 = Diag11 - Diag12 - Diag13;
            diagHess2 = -Diag22 - Diag23;
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

    // Output 3 & 4: diagonal Hessian blocks
    Map<MatrixXd>& diag1 = empty_map;
    Map<MatrixXd>& diag2 = empty_map;
    if (nlhs >= 3) {
        plhs[2] = mxCreateDoubleMatrix(gradSize, nFaces, mxREAL);
        plhs[3] = mxCreateDoubleMatrix(gradSize, nFaces, mxREAL);
        diag1 = *(new Map<MatrixXd>(mxGetPr(plhs[2]), gradSize, nFaces));
        diag2 = *(new Map<MatrixXd>(mxGetPr(plhs[3]), gradSize, nFaces));
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
        MatrixXd grad, hess1, hess2;
        liftedTriAreaGradHessian(vert, r, area, grad, hess1, hess2, nlhs);

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
            VectorXd hess1_reordered(grad.size());
            VectorXd hess2_reordered(grad.size());
            for (int j = 0; j < 6; ++j) {
                hess1_reordered(j) = hess1(reorder_indices(j));
                hess2_reordered(j) = hess2(reorder_indices(j));
            }
            diag1.col(i) = Map<VectorXd>(hess1_reordered.data(), gradSize);
            diag2.col(i) = Map<VectorXd>(hess2_reordered.data(), gradSize);
        }
    }
}