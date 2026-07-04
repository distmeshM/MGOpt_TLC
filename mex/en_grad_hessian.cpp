#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <iostream>
#include <mex.h>
#include <transfer.h>
#include <Eigen/Eigenvalues>

using Eigen::MatrixXd;
using Eigen::Matrix2d;
using Eigen::Vector2d;
using Eigen::Vector4d;
using Eigen::VectorXd;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
using std::vector;
using std::tuple;

/* *
* @brief Compute gradients of triangle areas with respect to vertex coordinates (z,w components)
* @param X1 Vertex 1 positions (n x 4)
* @param X2 Vertex 2 positions (n x 4)
* @param X3 Vertex 3 positions (n x 4)
* @return Gradients stored column-wise (6 x n)
*/
MatrixXd compute_triangle_gradients(
    const MatrixXd& X1,
    const MatrixXd& X2,
    const MatrixXd& X3)
{
    int n = X1.rows();  // number of triangles
    MatrixXd g(6, n);

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        // Extract vertices of the current triangle
        Vector4d x1 = X1.row(i);
        Vector4d x2 = X2.row(i);
        Vector4d x3 = X3.row(i);

        // Edge vectors
        Vector4d u = x2 - x1;
        Vector4d v = x3 - x1;

        // Intermediate quantities
        double norm_u_sq = u.squaredNorm();
        double norm_v_sq = v.squaredNorm();
        double u_dot_v = u.dot(v);
        double D = norm_u_sq * norm_u_sq - u_dot_v * u_dot_v;
        double S = 0.5 * std::sqrt(D);  // half area (clamped implicitly)

        // Gradient terms
        Vector4d G1 = -(norm_v_sq * u + norm_u_sq * v - u_dot_v * (u + v)) / (4.0 * S);
        Vector4d G2 = (norm_v_sq * u - u_dot_v * v) / (4.0 * S);
        Vector4d G3 = (norm_u_sq * v - u_dot_v * u) / (4.0 * S);

        // Store gradients for z and w components
        g.col(i) << G1(2), G2(2), G3(2), G1(3), G2(3), G3(3);
    }

    return g;
}

/* *
* @brief Compute reduced Hessians (z,w components only) for triangle areas
* @param X1 Vertex 1 positions (n x 4)
* @param X2 Vertex 2 positions (n x 4)
* @param X3 Vertex 3 positions (n x 4)
* @return Reduced Hessians stored column-wise (36 x n)
*/
MatrixXd compute_reduced_hessians(
    const MatrixXd& X1,
    const MatrixXd& X2,
    const MatrixXd& X3)
{
    int n = X1.rows();  // number of triangles
    MatrixXd Htri(36, n);

    // Reordering: [x1_z, x2_z, x3_z, x1_w, x2_w, x3_w]
    int order[6] = { 0, 2, 4, 1, 3, 5 };

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        // Extract (z,w) components of vertices
        VectorXd x1_zw = X1.row(i).tail(2);
        VectorXd x2_zw = X2.row(i).tail(2);
        VectorXd x3_zw = X3.row(i).tail(2);

        // Edge vectors in (z,w)
        VectorXd u_zw = x2_zw - x1_zw;
        VectorXd v_zw = x3_zw - x1_zw;

        // Full edge vectors (for area computation)
        VectorXd u = X2.row(i) - X1.row(i);
        VectorXd v = X3.row(i) - X1.row(i);

        // Intermediate quantities
        double norm_u_sq = u.squaredNorm();
        double norm_v_sq = v.squaredNorm();
        double u_dot_v = u.dot(v);
        double D = norm_u_sq * norm_v_sq - u_dot_v * u_dot_v;
        double S = 0.5 * std::sqrt(D);

        // Precompute coefficients
        double inv_2S = 1.0 / (4.0 * S);
        double inv_4S3 = -1.0 / S;

        // Gradient terms (z,w components)
        Vector2d G1_zw = -(norm_v_sq * u_zw + norm_u_sq * v_zw - u_dot_v * (u_zw + v_zw)) * inv_2S;
        Vector2d G2_zw = (norm_v_sq * u_zw - u_dot_v * v_zw) * inv_2S;
        Vector2d G3_zw = (norm_u_sq * v_zw - u_dot_v * u_zw) * inv_2S;

        // Diagonal Hessian blocks
        Matrix2d H11_zw =
            (-(u_zw - v_zw) * (u_zw - v_zw).transpose()
             + (norm_u_sq + norm_v_sq - 2.0 * u_dot_v) * Matrix2d::Identity())
            * inv_2S
            + (G1_zw * G1_zw.transpose()) * inv_4S3;

        Matrix2d H22_zw =
            (norm_v_sq * Matrix2d::Identity() - v_zw * v_zw.transpose())
            * inv_2S
            + (G2_zw * G2_zw.transpose()) * inv_4S3;

        Matrix2d H33_zw =
            (norm_u_sq * Matrix2d::Identity() - u_zw * u_zw.transpose())
            * inv_2S
            + (G3_zw * G3_zw.transpose()) * inv_4S3;

        // Off-diagonal Hessian blocks
        Matrix2d H12_zw =
            ((u_dot_v - norm_v_sq) * Matrix2d::Identity()
             - 2.0 * v_zw * u_zw.transpose()
             + (u_zw + v_zw) * v_zw.transpose())
            * inv_2S
            + (G1_zw * G2_zw.transpose()) * inv_4S3;

        Matrix2d H13_zw =
            ((u_dot_v - norm_u_sq) * Matrix2d::Identity()
             - 2.0 * u_zw * v_zw.transpose()
             + (u_zw + v_zw) * u_zw.transpose())
            * inv_2S
            + (G1_zw * G3_zw.transpose()) * inv_4S3;

        Matrix2d H23_zw =
            (-u_dot_v * Matrix2d::Identity()
             + 2.0 * u_zw * v_zw.transpose()
             - v_zw * u_zw.transpose())
            * inv_2S
            + (G2_zw * G3_zw.transpose()) * inv_4S3;

        // Assemble full 6x6 Hessian
        MatrixXd H(6, 6);
        H.block<2, 2>(0, 0) = H11_zw;
        H.block<2, 2>(2, 2) = H22_zw;
        H.block<2, 2>(4, 4) = H33_zw;

        H.block<2, 2>(0, 2) = H12_zw;
        H.block<2, 2>(2, 0) = H12_zw.transpose();

        H.block<2, 2>(0, 4) = H13_zw;
        H.block<2, 2>(4, 0) = H13_zw.transpose();

        H.block<2, 2>(2, 4) = H23_zw;
        H.block<2, 2>(4, 2) = H23_zw.transpose();

        // Project Hessian to PSD
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(H);
        Eigen::Matrix<double, 6, 6> V = es.eigenvectors();
        Eigen::Matrix<double, 6, 1> d = es.eigenvalues().cwiseMax(0.0);
        H.noalias() = V * d.asDiagonal() * V.transpose();

        // Store reordered Hessian column-wise
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                Htri(j + k * 6, i) = H(order[j], order[k]);
            }
        }
    }

    return Htri;
}

/* *
* @brief MATLAB mexFunction entry point
*/
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    MatrixXd X1 = convertDoubleToMatrixXd(prhs[0]);
    MatrixXd X2 = convertDoubleToMatrixXd(prhs[1]);
    MatrixXd X3 = convertDoubleToMatrixXd(prhs[2]);

    MatrixXd g = compute_triangle_gradients(X1, X2, X3);
    MatrixXd H = compute_reduced_hessians(X1, X2, X3);

    plhs[0] = convertEigenDenseToMatlab(g);
    plhs[1] = convertEigenDenseToMatlab(H);
}