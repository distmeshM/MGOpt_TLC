#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "mex.h"
#include "include/transfer.h"

using namespace Eigen;
using namespace std;

void mexFunction(int nlhs, mxArray* plhs[],
    int nrhs, const mxArray* prhs[]) {

    // Expect at most two input arguments:
    //   1) level index
    //   2) vector x
    if (nrhs > 2) {
        mexErrMsgIdAndTxt("sparse_matmul_single_mex:invalidInput",
            "Expected at most 2 input arguments: level and vector x.");
    }

    // Static storage for sparse matrices across calls
    static std::vector<SparseMatrix<double, RowMajor>> As;

    // --------------------------------------------------------
    // Initialization mode: only one input argument
    // --------------------------------------------------------
    if (nrhs == 1) {
        As.clear();
        As.shrink_to_fit();

        std::vector<SparseMatrix<double>> As_input =
            ReadCellOfSparseMatrices(prhs[0]);

        int nlevels = static_cast<int>(As_input.size());
        for (int lv = 0; lv < nlevels; ++lv) {
            SparseMatrix<double, RowMajor> A = As_input[lv];
            A.makeCompressed();
            As.push_back(A);
        }

        mexPrintf("Initialization successful. Total levels: %d\n",
            static_cast<int>(As.size()));
        return;
    }

    int nlevels = static_cast<int>(As.size());
    if (nlevels == 0) {
        mexErrMsgIdAndTxt("sparse_matmul_single_mex:notInitialized",
            "Sparse matrices not initialized. Call with one input first.");
    }

    if (nlhs != 1) {
        mexErrMsgIdAndTxt("sparse_matmul_single_mex:invalidOutput",
            "Expected one output argument: result vector.");
    }

    // --------------------------------------------------------
    // Validate inputs
    // --------------------------------------------------------
    if (!mxIsScalar(prhs[0])) {
        mexErrMsgIdAndTxt("sparse_matmul_single_mex:invalidInput",
            "First input must be a scalar (level index).");
    }

    if (!mxIsDouble(prhs[1]) ||
        mxGetNumberOfDimensions(prhs[1]) != 2 ||
        mxGetN(prhs[1]) != 1) {
        mexErrMsgIdAndTxt("sparse_matmul_single_mex:invalidInput",
            "Second input must be a dense column vector of type double.");
    }

    int level = static_cast<int>(mxGetScalar(prhs[0])) - 1;
    if (level < 0 || level >= nlevels) {
        mexErrMsgIdAndTxt("sparse_matmul_single_mex:invalidInput",
            "Level index out of bounds.");
    }

    const SparseMatrix<double, RowMajor>& A = As[level];

    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(A.cols());

    double* x_ptr = mxGetPr(prhs[1]);
    mwSize x_rows = mxGetM(prhs[1]);
    mwSize x_cols = mxGetN(prhs[1]);

    if (x_cols != 1) {
        mexErrMsgIdAndTxt("sparse_matmul_single_mex:invalidInput",
            "Input vector x must be a column vector.");
    }

    if (static_cast<int>(x_rows) != n) {
        mexErrMsgIdAndTxt("sparse_matmul_single_mex:invalidInput",
            "Dimension mismatch: number of rows of x must equal number of columns of A.");
    }

    Map<const VectorXd> x_map(x_ptr, x_rows);

    // --------------------------------------------------------
    // Allocate output
    // --------------------------------------------------------
    plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
    double* result_ptr = mxGetPr(plhs[0]);
    Map<VectorXd> result(result_ptr, m);

    // Optional: set Eigen thread pool
    Eigen::setNbThreads(24);

    // Sparse matrix-vector multiplication
    result = A * x_map;
}