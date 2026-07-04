#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <iostream>
#include <mex.h>

/* *
* @brief Convert a MATLAB double dense matrix to Eigen::MatrixXd
*/
Eigen::MatrixXd convertDoubleToMatrixXd(const mxArray* matlabArray) {
    if (!mxIsDouble(matlabArray)) {
        throw std::runtime_error("Input must be a double matrix");
    }

    mwSize rows = mxGetM(matlabArray);
    mwSize cols = mxGetN(matlabArray);
    double* data = mxGetPr(matlabArray);

    return Eigen::Map<const Eigen::MatrixXd>(data, rows, cols);
}

/* *
* @brief Convert a MATLAB int32 dense matrix to Eigen::MatrixXi
*/
Eigen::MatrixXi convertInt32ToMatrixXi(const mxArray* matlabArray) {
    if (!mxIsInt32(matlabArray)) {
        throw std::runtime_error("Input must be an int32 matrix");
    }

    mwSize rows = mxGetM(matlabArray);
    mwSize cols = mxGetN(matlabArray);
    int32_t* data = static_cast<int32_t*>(mxGetData(matlabArray));

    return Eigen::Map<const Eigen::MatrixXi>(data, rows, cols);
}

/* *
* @brief Convert an Eigen::SparseMatrix<double> to a MATLAB sparse matrix
*/
mxArray* convertEigenSparseToMatlab(const Eigen::SparseMatrix<double>& mat) {
    mwSize rows = static_cast<mwSize>(mat.rows());
    mwSize cols = static_cast<mwSize>(mat.cols());
    mwSize nnz = static_cast<mwSize>(mat.nonZeros());

    mxArray* mxSparse = mxCreateSparse(rows, cols, nnz, mxREAL);

    double* pr = pr;
    mwIndex* ir = mxGetIr(mxSparse);
    mwIndex* jc = mxGetJc(mxSparse);

    mwIndex index = 0;
    for (int k = 0; k < mat.outerSize(); ++k) {
        jc[k] = index;
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
            pr[index] = it.value();
            ir[index] = static_cast<mwIndex>(it.row());
            ++index;
        }
    }
    jc[mat.outerSize()] = index;

    return mxSparse;
}

/* *
* @brief Convert an Eigen::MatrixXd to a MATLAB dense matrix
*/
mxArray* convertEigenDenseToMatlab(const Eigen::MatrixXd& mat) {
    mwSize rows = static_cast<mwSize>(mat.rows());
    mwSize cols = static_cast<mwSize>(mat.cols());

    mxArray* mxDense = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double* pr = mxGetPr(mxDense);

    // MATLAB uses column-major storage
    for (mwSize j = 0; j < cols; ++j) {
        for (mwSize i = 0; i < rows; ++i) {
            pr[j * rows + i] = mat(i, j);
        }
    }

    return mxDense;
}

/* *
* @brief Convert an Eigen::MatrixXi to a MATLAB dense matrix (stored as double)
*/
mxArray* convertEigenDenseIntToMatlab(const Eigen::MatrixXi& mat) {
    mwSize rows = static_cast<mwSize>(mat.rows());
    mwSize cols = static_cast<mwSize>(mat.cols());

    mxArray* mxDense = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double* pr = mxGetPr(mxDense);

    for (mwSize j = 0; j < cols; ++j) {
        for (mwSize i = 0; i < rows; ++i) {
            pr[j * rows + i] = static_cast<double>(mat(i, j));
        }
    }

    return mxDense;
}

/* *
* @brief Convert a MATLAB char array (string) to a C-style string
*/
const char* matlabStringToChar(const mxArray* mxStr) {
    if (!mxIsChar(mxStr)) {
        mexErrMsgIdAndTxt("MATLAB:invalidInput", "Input must be a string.");
        return nullptr;
    }

    char* cStr = mxArrayToString(mxStr);
    if (cStr == nullptr) {
        mexErrMsgIdAndTxt("MATLAB:conversionFailed", "String conversion failed.");
    }
    return cStr;
}

/* *
* @brief Convert a MATLAB sparse matrix to Eigen::SparseMatrix<double>
*/
Eigen::SparseMatrix<double> convertMatlabSparseToEigen(const mxArray* matlabSparse) {
    // Input validation
    if (!mxIsSparse(matlabSparse)) {
        mexErrMsgIdAndTxt("ConvertMatlabSparseToEigen:InputError",
                         "Input must be a MATLAB sparse matrix.");
    }
    if (mxIsComplex(matlabSparse)) {
        mexErrMsgIdAndTxt("ConvertMatlabSparseToEigen:InputError",
                         "Complex sparse matrices are not supported (input must be real).");
    }

    // Basic matrix information
    const mwSize nrows = mxGetM(matlabSparse);
    const mwSize ncols = mxGetN(matlabSparse);
    const mwSize nnz = mxGetNzmax(matlabSparse);

    // MATLAB sparse matrix internals
    const mwIndex* ir = mxGetIr(matlabSparse);
    const mwIndex* jc = mxGetJc(matlabSparse);
    const double* pr = mxGetPr(matlabSparse);

    // Build triplet list
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    for (mwIndex col = 0; col < ncols; ++col) {
        const mwIndex start = jc[col];
        const mwIndex end = jc[col + 1];

        for (mwIndex k = start; k < end; ++k) {
            const mwIndex row = ir[k];
            const double val = pr[k];

            triplets.emplace_back(
                static_cast<Eigen::Index>(row),
                static_cast<Eigen::Index>(col),
                val
            );
        }
    }

    // Construct Eigen sparse matrix
    Eigen::SparseMatrix<double> eigenMat(
        static_cast<Eigen::Index>(nrows),
        static_cast<Eigen::Index>(ncols)
    );
    eigenMat.setFromTriplets(triplets.begin(), triplets.end());

    return eigenMat;
}

/* *
* @brief Read a MATLAB cell array of sparse matrices into a std::vector
*/
std::vector<Eigen::SparseMatrix<double>>
ReadCellOfSparseMatrices(const mxArray* cellArray) {
    if (!mxIsCell(cellArray)) {
        mexErrMsgIdAndTxt("ReadCellSparse:InputNotCell",
                         "Input must be a MATLAB cell array.");
    }

    mwSize numCells = mxGetNumberOfElements(cellArray);
    std::vector<Eigen::SparseMatrix<double>> sparseMatrices;

    for (mwIndex i = 0; i < numCells; ++i) {
        const mxArray* cellElement = mxGetCell(cellArray, i);

        if (!mxIsSparse(cellElement)) {
            mexPrintf("Warning: Cell #%d is not a sparse matrix; skipped.\n",
                      static_cast<int>(i));
            continue;
        }

        if (mxGetNumberOfDimensions(cellElement) != 2) {
            mexPrintf("Warning: Cell #%d is not a 2D matrix; skipped.\n",
                      static_cast<int>(i));
            continue;
        }

        if (!mxIsDouble(cellElement)) {
            mexPrintf("Warning: Cell #%d is not double-precision; skipped.\n",
                      static_cast<int>(i));
            continue;
        }

        sparseMatrices.push_back(convertMatlabSparseToEigen(cellElement));
    }

    return sparseMatrices;
}