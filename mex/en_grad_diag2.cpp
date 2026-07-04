#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "mex.h"

using namespace Eigen;

/**
 * @brief Compute triangle area using a numerically stable Heron formula
 *        with lifted edge lengths.
 *
 * @param d1, d2, d3 Squared edge lengths plus lift variables r
 * @return Triangle area (returns 0 if input is degenerate)
 */
double HeronTriArea(double d1, double d2, double d3) {
    // Sort so that a >= b >= c
    double a, b, c;
    if (d1 > d2) {
        a = d1;
        b = d2;
    } else {
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

    return 0.25 * std::sqrt(
        std::abs((a + (b + c)) *
                 (c - (a - b)) *
                 (c + (a - b)) *
                 (a + (b - c))));
}

/**
 * @brief Compute triangle area, gradient, and diagonal Hessian
 *        for a lifted triangle.
 *
 * @param vert      3x3 vertex matrix (each column is a vertex)
 * @param r         Lift variables (Vector3d)
 * @param[out] area Triangle area
 * @param[out] grad Gradient w.r.t. vertices (3 x dim)
 * @param[out] diagHess Diagonal of the Hessian (9-vector)
 * @param nlhs      Number of requested output arguments from MATLAB
 */
void liftedTriAreaGradHessian(const MatrixXd& vert,
                              const Vector3d& r,
                              double& area,
                              MatrixXd& grad,
                              VectorXd& diagHess,
                              int nlhs) {
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

    // Area
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

        // Gradient has the same shape as vert
        grad.resize(vert.rows(), vert.cols());
        grad.col(0) = av1;
        grad.col(1) = av2;
        grad.col(2) = av3;

        double s = 1.0 / (8.0 * area);
        grad *= s;

        if (nlhs >= 3) {
            // Hessian term 1: Laplacian-like block
            Matrix3d Lap;
            Lap << g2 + g3, -g3,      -g2,
                   -g3, g1 + g3,      -g1,
                   -g2,      -g1, g1 + g2;
            Lap *= s;

            // Expand Laplacian diagonal to full vector
            VectorXd Diag1(3 * vDim);
            for (int i = 0; i < 3; ++i) {
                Diag1.segment(i * vDim, vDim).setConstant(Lap(i, i));
            }

            // Hessian term 2
            VectorXd Diag2(3 * vDim);
            Diag2 << e1.array().square(),
                     e2.array().square(),
                     e3.array().square();
            s *= 2.0; // now s = 1/(4*area)
            Diag2 *= s;

            // Hessian term 3
            VectorXd Diag3(3 * vDim);
            Diag3 << av1.array().square(),
                     av2.array().square(),
                     av3.array().square();
            s = s * s / (4.0 * area); // now s = 1/(64*area^3)
            Diag3 *= s;

            // Final diagonal Hessian
            diagHess = Diag1 - Diag2 - Diag3;
        }
    }
}

/**
 * @brief MATLAB MEX entry point
 *
 * @param nlhs Number of output arguments
 * @param plhs Output argument pointers
 * @param nrhs Number of input arguments
 * @param prhs Input argument pointers
 */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {

    // Input validation
    if (nrhs != 3) {
        mexErrMsgTxt("Expected 3 inputs: V, F, restD");
    }

    // Map MATLAB inputs to Eigen types
    Map<const MatrixXd> V(mxGetPr(prhs[0]),
                           mxGetM(prhs[0]), mxGetN(prhs[0]));
    Map<const MatrixXi> F((int*)mxGetPr(prhs[1]),
                           mxGetM(prhs[1]), mxGetN(prhs[1]));
    Map<const MatrixXd> restD(mxGetPr(prhs[2]), 3, mxGetN(prhs[2]));

    int nFaces = F.cols();
    int vDim = V.rows();
    int gradSize = vDim * 3;
    int hessSize = gradSize;

    // Empty fallback maps
    static Map<MatrixXd> empty_map(nullptr, 0, 0);

    // Output 1: areas (1 x nFaces)
    plhs[0] = mxCreateDoubleMatrix(1, nFaces, mxREAL);
    Map<RowVectorXd> areas(mxGetPr(plhs[0]), nFaces);

    // Output 2: gradients (gradSize x nFaces)
    Map<MatrixXd>* grads_ptr = &empty_map;
    if (nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(gradSize, nFaces, mxREAL);
        grads_ptr = new Map<MatrixXd>(mxGetPr(plhs[1]), gradSize, nFaces);
    }

    // Output 3: diagonal Hessians (hessSize x nFaces)
    Map<MatrixXd>* hessians_ptr = &empty_map;
    if (nlhs >= 3) {
        plhs[2] = mxCreateDoubleMatrix(hessSize, nFaces, mxREAL);
        hessians_ptr = new Map<MatrixXd>(mxGetPr(plhs[2]), hessSize, nFaces);
    }

    Map<MatrixXd>& grads = *grads_ptr;
    Map<MatrixXd>& hessians = *hessians_ptr;

    // Reordering indices:
    // MATLAB layout [x1 y1 x2 y2 x3 y3]
    // C++ layout     [x1 x2 x3 y1 y2 y3]
    Eigen::VectorXi reorder_indices(6);
    reorder_indices << 0, 2, 4, 1, 3, 5;

    // Parallel face loop
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

        areas(i) = area;

        if (nlhs >= 2) {
            VectorXd grad_reordered(grad.size());
            for (int j = 0; j < 6; ++j) {
                grad_reordered(j) = grad(reorder_indices(j));
            }
            grads.col(i) = grad_reordered;
        }

        if (nlhs >= 3) {
            VectorXd hess_reordered(diagHess.size());
            for (int j = 0; j < 6; ++j) {
                hess_reordered(j) = diagHess(reorder_indices(j));
            }
            hessians.col(i) = hess_reordered;
        }
    }

    // Cleanup dynamically allocated maps
    if (nlhs >= 2) delete grads_ptr;
    if (nlhs >= 3) delete hessians_ptr;
}