#include <igl/read_triangle_mesh.h>

#include <igl/opengl/glfw/Viewer.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <iostream>
#include <vector>

#include <mg_data.h>
#include <mg_precompute.h>
#include <mex.h>

// ���Eigenδ����VectorXb���ֶ�����
#if !defined(EIGEN_VECTORXB_H) && !defined(Eigen_VectorXb_H)  // ���ݲ�ͬEigen�汾
#define EIGEN_VECTORXB_H
using VectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
#endif

// �� MATLAB double ����ת��Ϊ Eigen::MatrixXd
Eigen::MatrixXd convertDoubleToMatrixXd(const mxArray * matlabArray) {
    if (!mxIsDouble(matlabArray)) {
        throw std::runtime_error("Input must be a double matrix");
    }

    mwSize rows = mxGetM(matlabArray);
    mwSize cols = mxGetN(matlabArray);
    double* data = mxGetPr(matlabArray);

    return Eigen::Map<const Eigen::MatrixXd>(data, rows, cols);
}

// �� MATLAB int32 ����ת��Ϊ Eigen::MatrixXi
Eigen::MatrixXi convertInt32ToMatrixXi(const mxArray* matlabArray) {
    if (!mxIsInt32(matlabArray)) {
        throw std::runtime_error("Input must be an int32 matrix");
    }

    mwSize rows = mxGetM(matlabArray);
    mwSize cols = mxGetN(matlabArray);
    int32_t* data = (int32_t*)mxGetData(matlabArray);

    return Eigen::Map<const Eigen::MatrixXi>(data, rows, cols);
}

// �� Eigen::SparseMatrix<double> ת��Ϊ MATLAB ��ϡ�����
mxArray* convertEigenSparseToMatlab(const Eigen::SparseMatrix<double>& mat) {
    // ��ȡ����������������ͷ���Ԫ������
    mwSize rows = mat.rows();
    mwSize cols = mat.cols();
    mwSize nnz = mat.nonZeros();

    // ���� MATLAB ϡ�����
    mxArray* mxSparse = mxCreateSparse(rows, cols, nnz, mxREAL);

    // ��ȡ MATLAB ϡ����������ָ��
    double* pr = mxGetPr(mxSparse);
    mwIndex* ir = mxGetIr(mxSparse);
    mwIndex* jc = mxGetJc(mxSparse);

    // ��� MATLAB ϡ�����
    mwIndex index = 0;
    for (int k = 0; k < mat.outerSize(); ++k) {
        jc[k] = index;
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
            pr[index] = it.value();
            ir[index] = it.row();
            index++;
        }
    }
    jc[mat.outerSize()] = index;

    return mxSparse;
}

// �� Eigen::Matrix<double, Dynamic, Dynamic> ת��Ϊ MATLAB �ĳ��ܾ���
mxArray* convertEigenDenseToMatlab(const Eigen::MatrixXd& mat) {
    mwSize rows = mat.rows();
    mwSize cols = mat.cols();

    // ���� MATLAB ����
    mxArray* mxDense = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double* pr = mxGetPr(mxDense);

    // �������� (MATLAB ����������洢)
    for (mwSize j = 0; j < cols; ++j) {
        for (mwSize i = 0; i < rows; ++i) {
            pr[j * rows + i] = mat(i, j);
        }
    }

    return mxDense;
}

// �� Eigen::Matrix<int, Dynamic, Dynamic> ת��Ϊ MATLAB �ĳ��ܾ���
mxArray* convertEigenDenseIntToMatlab(const Eigen::MatrixXi& mat) {
    mwSize rows = mat.rows();
    mwSize cols = mat.cols();

    // MATLAB û���������͵� mxArray����Ҫת��Ϊ double
    mxArray* mxDense = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double* pr = mxGetPr(mxDense);

    // �������� (MATLAB ����������洢)
    for (mwSize j = 0; j < cols; ++j) {
        for (mwSize i = 0; i < rows; ++i) {
            pr[j * rows + i] = static_cast<double>(mat(i, j));
        }
    }

    return mxDense;
}

const char* matlabStringToChar(const mxArray* mxStr) {
    if (!mxIsChar(mxStr)) {
        mexErrMsgIdAndTxt("MATLAB:invalidInput", "Input must be a string.");
        return nullptr;
    }

    // �� mxArray ת��Ϊ C ����ַ���
    char* cStr = mxArrayToString(mxStr);
    if (cStr == nullptr) {
        mexErrMsgIdAndTxt("MATLAB:conversionFailed", "String conversion failed.");
    }
    return cStr;
}



void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    using namespace Eigen;
    using namespace std;

    if (nrhs < 2) {
        mexErrMsgIdAndTxt("MATLAB:invalidNumInputs", "At least two input required.");
    }
    
    int decimation_type = 1;
    // load mesh
    MatrixXd VO, V;
    MatrixXi FO, F;
    MatrixXi constrainId;
    constrainId.resize(0, 0);
    int* data;
    double w1 = 0.01;
    if (nrhs == 2)
    {
        igl::read_triangle_mesh(matlabStringToChar(prhs[0]), VO, FO);
        //cout << "original mesh: |V| " << VO.rows() << ", |F|: " << FO.rows() << endl;
        mexPrintf("original mesh: |V| %d, |F|: %d", VO.rows(), FO.rows());
        data = (int*)mxGetData(prhs[1]);
    }
    else if (nrhs == 3)
    {
        VO = convertDoubleToMatrixXd(prhs[0]);
        FO = convertInt32ToMatrixXi(prhs[1]);
        data = (int*)mxGetData(prhs[2]);
    }
    else if (nrhs == 4)
    {
        VO = convertDoubleToMatrixXd(prhs[0]);
        FO = convertInt32ToMatrixXi(prhs[1]);
        decimation_type = int(mxGetScalar(prhs[3]));
        data = (int*)mxGetData(prhs[2]);
    }
    else if (nrhs == 5)
    {
        VO = convertDoubleToMatrixXd(prhs[0]);
        FO = convertInt32ToMatrixXi(prhs[1]);
        decimation_type = int(mxGetScalar(prhs[3]));
        data = (int*)mxGetData(prhs[2]);
        constrainId = convertInt32ToMatrixXi(prhs[4]);
    }
    else if (nrhs == 6)
    {
        VO = convertDoubleToMatrixXd(prhs[0]);
        FO = convertInt32ToMatrixXi(prhs[1]);
        decimation_type = int(mxGetScalar(prhs[3]));
        data = (int*)mxGetData(prhs[2]);
        constrainId = convertInt32ToMatrixXi(prhs[4]);
        w1 = mxGetScalar(prhs[5]);
    }
    else
    {
        mexErrMsgIdAndTxt("MATLAB:invalidNumInputs", "At most three input required.");
        return;
    }
    VectorXb constrained(VO.rows() + 1);  // Լ��������
    for (int i = 0; i < VO.rows() + 1; i++)
    {
        constrained(i) = false;
    }
    for (int i = 0; i < constrainId.rows(); i++)
    {
        constrained(constrainId(i)) = true;
    }
    // construct the multigrid hierarchy
    int min_coarsest_nV = 4;
    float coarsening_ratio = 1.0 / data[0];
    vector<mg_data> mg;
    mg_precompute(VO, FO, coarsening_ratio, min_coarsest_nV, decimation_type, mg, constrained, w1);

    int nlevel = mg.size();
    plhs[0] = mxCreateCellMatrix(nlevel, 1);
    plhs[1] = mxCreateCellMatrix(nlevel, 1);
    plhs[2] = mxCreateCellMatrix(nlevel - 1, 1);
    plhs[3] = mxCreateCellMatrix(nlevel - 1, 1);

    for (mwIndex i = 0; i < nlevel; ++i) {
        mxArray* mxDense = convertEigenDenseToMatlab(mg[i].V);
        mxSetCell(plhs[0], i, mxDense);
    }

    for (mwIndex i = 0; i < nlevel; ++i) {
        mxArray* mxDense = convertEigenDenseIntToMatlab(mg[i].F);
        mxSetCell(plhs[1], i, mxDense);
    }

    for (mwIndex i = 0; i < nlevel - 1; ++i) {
        mxArray* mxSparse = convertEigenSparseToMatlab(mg[i + 1].P_full);
        mxSetCell(plhs[2], i, mxSparse);
    }

    for (mwIndex i = 0; i < nlevel - 1; ++i) {
        mxArray* mxSparse = convertEigenSparseToMatlab(mg[i + 1].P2);
        mxSetCell(plhs[3], i, mxSparse);
    }
}