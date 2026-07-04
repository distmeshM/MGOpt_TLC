#include "mqwf_dense.h"

typedef CGAL::Gmpq ET;
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

void saveMatrixXdToTxt(const Eigen::MatrixXd& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Set output format (scientific notation, aligned)
    file << std::scientific << std::setprecision(15);

    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j);
            if (j < matrix.cols() - 1) {
                file << " ";  // space-separated columns
            }
        }
        if (i < matrix.rows() - 1) {
            file << "\n";     // newline between rows
        }
    }
    file.close();
}

void check_matrix_definiteness(const Eigen::MatrixXd& A, const std::string& matrix_name = "Matrix") {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
    double min_eigenvalue = solver.eigenvalues().minCoeff();
    double max_eigenvalue = solver.eigenvalues().maxCoeff();

    mexPrintf("%s eigenvalue analysis:\n", matrix_name.c_str());
    mexPrintf("  Min eigenvalue: %.15g\n", min_eigenvalue);
    mexPrintf("  Max eigenvalue: %.15g\n", max_eigenvalue);

    if (min_eigenvalue <= 0) {
        mexWarnMsgIdAndTxt("EigenCheck:NonPositive",
            "WARNING: %s is not positive definite (min eigenvalue <= 0)\n",
            matrix_name.c_str());
    }

    double condition_number = max_eigenvalue / std::max(min_eigenvalue, 1e-16);
    if (condition_number > 1e12) {
        mexWarnMsgIdAndTxt("EigenCheck:IllConditioned",
            "WARNING: %s has high condition number: %.3g\n",
            matrix_name.c_str(), condition_number);
    }
}

void print_constraints(const CGAL::Quadratic_program<double>& qp, size_t num_constraints) {
    mexPrintf("Total number of constraints: %zu\n", num_constraints);

    for (size_t i = 0; i < num_constraints; ++i) {
        const char* rel_str = "";
        switch (qp.get_r()[i]) {
        case CGAL::SMALLER: rel_str = "<="; break;
        case CGAL::EQUAL:   rel_str = "=="; break;
        case CGAL::LARGER:  rel_str = ">="; break;
        }

        mexPrintf("Constraint %zu: type=%s, bound=%.6f\n",
            i, rel_str, qp.get_b()[i]);
    }
}

/**
* @brief Solve a constrained quadratic programming problem
*
* @param A Symmetric positive-definite matrix
* @param b Right-hand side vector
* @param constrained_idx Index of the variable to be constrained (0-based)
* @param lower_bound Lower bound of the constraint
* @param upper_bound Upper bound of the constraint
* @return std::vector<double> Solution vector
*/
Eigen::VectorXd solve_constrained_qp(
    const Eigen::MatrixXd& AO,
    const Eigen::VectorXd& b,
    int constrained_idx,
    double lower_bound,
    double upper_bound,
    const Eigen::VectorXi& known = Eigen::VectorXi(),
    const Eigen::VectorXd& known_val = Eigen::VectorXd())
{
    const int n = AO.rows();
    Program qp(CGAL::SMALLER, false, 0, false, 0);
    Eigen::MatrixXd A = AO;

    // Input validation (MATLAB MEX version)
    if (constrained_idx < 0 || constrained_idx >= n) {
        mexErrMsgIdAndTxt("QP:InvalidIndex",
            "Constraint index out of bounds: constrained_idx=%d (valid range: 0-%d)",
            constrained_idx, n - 1);
    }

    if (known.size() != known_val.size()) {
        mexErrMsgIdAndTxt("QP:SizeMismatch",
            "known and known_val size mismatch: %d != %d",
            known.size(), known_val.size());
    }

    for (int i = 0; i < known.size(); ++i) {
        if (known(i) < 0 || known(i) >= n) {
            mexErrMsgIdAndTxt("QP:InvalidKnownIndex",
                "known contains invalid index: known(%d)=%d (valid range: 0-%d)",
                i, known(i), n - 1);
        }
    }

    // Set objective function
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double coeff = (i == j) ? A(i, j) : (A(i, j) + A(j, i)) / 2;
            qp.set_d(i, j, coeff);
        }
        qp.set_c(i, -b(i));
    }

    // Add constraints
    int constraint_counter = 0;

    // Inequality constraints (upper bound)
    if (!std::isinf(upper_bound)) {
        qp.set_a(constrained_idx, constraint_counter, 1);
        qp.set_b(constraint_counter++, upper_bound);
        qp.set_r(constraint_counter, CGAL::SMALLER);
    }

    // Inequality constraints (lower bound)
    if (!std::isinf(lower_bound)) {
        qp.set_a(constrained_idx, constraint_counter, -1);
        qp.set_b(constraint_counter++, -lower_bound);
        qp.set_r(constraint_counter, CGAL::SMALLER);
    }

    // Equality constraints
    for (int i = 0; i < known.size(); ++i) {
        qp.set_a(known(i), constraint_counter, 1);
        qp.set_b(constraint_counter, known_val(i));
        qp.set_r(constraint_counter, CGAL::EQUAL);
        constraint_counter++;
    }

    Eigen::VectorXd solution(n);
    try {
        // Solve QP
        Solution s = CGAL::solve_quadratic_program(qp, ET());

        // Process results
        if (s.is_optimal()) {
            auto x = s.variable_values_begin();
            for (int i = 0; i < n; ++i) {
                solution(i) = CGAL::to_double(*x++);
            }
        }
        else {
            throw std::runtime_error(
                s.is_unbounded() ? "Problem is unbounded" :
                s.is_infeasible() ? "Problem is infeasible" :
                "Unknown solver error");
        }
    }
    catch (const CGAL::Assertion_exception& e) {
        std::cerr << "CGAL assertion error: " << e.what() << std::endl;
        throw std::runtime_error("CGAL solver internal error");
    }

    return solution;
}

Eigen::VectorXd solve_constrained_qp(
    const Eigen::MatrixXd& AO,
    const Eigen::VectorXd& b,
    int constrained_idx,
    double lower_bound,
    double upper_bound,
    int constrained_idx2,
    double lower_bound2,
    double upper_bound2,
    const Eigen::VectorXi& known = Eigen::VectorXi(),
    const Eigen::VectorXd& known_val = Eigen::VectorXd())
{
    const int n = AO.rows();
    Program qp(CGAL::SMALLER, false, 0, false, 0);
    Eigen::MatrixXd A = AO;

    // Input validation
    if (constrained_idx < 0 || constrained_idx >= n) {
        throw std::out_of_range("Constraint index out of bounds");
    }
    if (constrained_idx2 < 0 || constrained_idx2 >= n) {
        throw std::out_of_range("Constraint index out of bounds");
    }
    if (known.size() != known_val.size()) {
        throw std::invalid_argument("known and known_val must have the same size");
    }
    for (int i = 0; i < known.size(); ++i) {
        if (known(i) < 0 || known(i) >= n) {
            throw std::out_of_range("known indices out of bounds");
        }
    }

    // Set objective function
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double coeff = (i == j) ? A(i, j) : (A(i, j) + A(j, i)) / 2;
            qp.set_d(i, j, coeff);
        }
        qp.set_c(i, -b(i));
    }

    // Add constraints
    int constraint_counter = 0;

    // Inequality constraints (first variable)
    if (!std::isinf(upper_bound)) {
        qp.set_a(constrained_idx, constraint_counter, 1);
        qp.set_b(constraint_counter++, upper_bound);
        qp.set_r(constraint_counter, CGAL::SMALLER);
    }
    if (!std::isinf(lower_bound)) {
        qp.set_a(constrained_idx, constraint_counter, -1);
        qp.set_b(constraint_counter++, -lower_bound);
        qp.set_r(constraint_counter, CGAL::SMALLER);
    }

    // Inequality constraints (second variable)
    if (!std::isinf(upper_bound2)) {
        qp.set_a(constrained_idx2, constraint_counter, 1);
        qp.set_b(constraint_counter++, upper_bound2);
        qp.set_r(constraint_counter, CGAL::SMALLER);
    }
    if (!std::isinf(lower_bound2)) {
        qp.set_a(constrained_idx2, constraint_counter, -1);
        qp.set_b(constraint_counter++, -lower_bound2);
        qp.set_r(constraint_counter, CGAL::SMALLER);
    }

    // Equality constraints
    for (int i = 0; i < known.size(); ++i) {
        qp.set_a(known(i), constraint_counter, 1);
        qp.set_b(constraint_counter, known_val(i));
        qp.set_r(constraint_counter, CGAL::EQUAL);
        constraint_counter++;
    }

    Eigen::VectorXd solution(n);
    try {
        Solution s = CGAL::solve_quadratic_program(qp, ET());

        if (s.is_optimal()) {
            auto x = s.variable_values_begin();
            for (int i = 0; i < n; ++i) {
                solution(i) = CGAL::to_double(*x++);
            }
        }
        else {
            throw std::runtime_error(
                s.is_unbounded() ? "Problem is unbounded" :
                s.is_infeasible() ? "Problem is infeasible" :
                "Unknown solver error");
        }
    }
    catch (const CGAL::Assertion_exception& e) {
        std::cerr << "CGAL assertion error: " << e.what() << std::endl;
        throw std::runtime_error("CGAL solver internal error");
    }

    return solution;
}

void mqwf_dense_precompute(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXi& known,
    mqwf_dense_data& data)
{
    using namespace std;
    using namespace Eigen;

    int n = A.rows();

    // Store known indices
    data.k.resize(known.size());
    data.k = known;

    // Compute unknown indices
    data.u.resize(n - data.k.size());
    {
        VectorXi known_sort;
        igl::sort(known, 1, true, known_sort);
        VectorXi tmp = VectorXi::LinSpaced(n, 0, n - 1);
        auto it = std::set_difference(tmp.data(), tmp.data() + tmp.size(),
            known_sort.data(), known_sort.data() + known_sort.size(),
            data.u.data());
        data.u.conservativeResize(std::distance(data.u.data(), it));
    }
    assert((data.u.size() + data.k.size()) == n);

    // Extract submatrices
    MatrixXd Auu, Auk, Aku;
    Auu.resize(data.u.size(), data.u.size());
    igl::slice(A, data.u, data.u, Auu);
    igl::slice(A, data.u, data.k, Auk);
    igl::slice(A, data.k, data.u, Aku);

    // Store factorization and auxiliary data
    data.Auu_pref = Auu.ldlt();
    data.Auk_plus_AkuT = Auk + Aku.transpose();
    data.n = n;
}

void mqwf_dense_solve(
    const mqwf_dense_data& data,
    const Eigen::VectorXd& RHS,
    const Eigen::VectorXd& known_val,
    Eigen::VectorXd& sol)
{
    using namespace std;
    using namespace Eigen;

    VectorXi col_colon = VectorXi::LinSpaced(RHS.cols(), 0, RHS.cols() - 1);

    // Construct reduced right-hand side
    VectorXd RHS_reduced;
    if (data.k.size() == 0) {
        RHS_reduced = -RHS;
    }
    else {
        VectorXd RHS_unknown;
        igl::slice(RHS, data.u, col_colon, RHS_unknown);
        RHS_reduced = -0.5 * data.Auk_plus_AkuT * known_val - RHS_unknown;
    }

    // Solve reduced system
    sol.resize(data.n, RHS.cols());

    VectorXd sol_unknown;
    sol_unknown.resize(data.u.size(), RHS.cols());
    sol_unknown = data.Auu_pref.solve(RHS_reduced);

    igl::slice_into(sol_unknown, data.u, col_colon, sol);
    igl::slice_into(known_val, data.k, col_colon, sol);
}

void mqwf_constraint_solve(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& RHS,
    const Eigen::MatrixXi& known,
    const Eigen::VectorXd& known_val,
    Eigen::VectorXd& sol,
    int nonlinearId, int dir)
{
    using namespace std;
    using namespace Eigen;

    int n = A.rows();
    sol.resize(n, RHS.cols());

    if (dir == 0) {
        sol = solve_constrained_qp(A, RHS, nonlinearId, -100, -0.1, known, known_val);
    }
    else if (dir == 1) {
        sol = solve_constrained_qp(A, RHS, nonlinearId, 1.1, 100, known, known_val);
    }
    else {
        std::cerr << "Error: invalid direction flag.\n" << std::endl;
    }
}

void mqwf_nonlinear_precompute(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXi& known,
    mqwf_dense_data& data,
    int nonlinearId, int dir)
{
    using namespace std;
    using namespace Eigen;

    int n = A.rows();

    data.k.resize(known.size());
    data.k = known;

    // Compute unknown indices
    data.u.resize(n - data.k.size());
    {
        VectorXi known_sort;
        igl::sort(known, 1, true, known_sort);
        VectorXi tmp = VectorXi::LinSpaced(n, 0, n - 1);
        auto it = std::set_difference(tmp.data(), tmp.data() + tmp.size(),
            known_sort.data(), known_sort.data() + known_sort.size(),
            data.u.data());
        data.u.conservativeResize(std::distance(data.u.data(), it));
    }
    assert((data.u.size() + data.k.size()) == n);

    // Locate nonlinear variable in reduced system
    for (int i = 0; i < data.u.size(); i++) {
        if (data.u[i] == nonlinearId)
            data.nonlinear = i;
    }
    assert(data.nonlinear != -1);

    // Extract submatrices
    MatrixXd Auu, Auk, Aku;
    Auu.resize(data.u.size(), data.u.size());
    igl::slice(A, data.u, data.u, Auu);
    igl::slice(A, data.u, data.k, Auk);
    igl::slice(A, data.k, data.u, Aku);

    // Store data
    data.Auu_pref = Auu.ldlt();
    data.Auu = Auu;
    data.Auk_plus_AkuT = Auk + Aku.transpose();
    data.n = n;
}

void mqwf_nonlinear_solve(
    const mqwf_dense_data& data,
    const Eigen::VectorXd& RHS,
    const Eigen::VectorXd& known_val,
    Eigen::VectorXd& sol, int dir)
{
    using namespace std;
    using namespace Eigen;

    VectorXi col_colon = VectorXi::LinSpaced(RHS.cols(), 0, RHS.cols() - 1);

    // Construct reduced right-hand side
    VectorXd RHS_reduced;
    if (data.k.size() == 0) {
        RHS_reduced = -RHS;
    }
    else {
        VectorXd RHS_unknown;
        igl::slice(RHS, data.u, col_colon, RHS_unknown);
        RHS_reduced = -0.5 * data.Auk_plus_AkuT * known_val - RHS_unknown;
    }

    // Solve reduced system
    sol.resize(data.n, RHS.cols());

    VectorXd sol_unknown;
    sol_unknown.resize(data.u.size(), RHS.cols());
    sol_unknown = data.Auu_pref.solve(RHS_reduced);

    if (dir == 0) {
        sol_unknown = solve_constrained_qp(data.Auu, RHS_reduced, data.nonlinear, -100, -0.1);
    }
    else if (dir == 1) {
        sol_unknown = solve_constrained_qp(data.Auu, RHS_reduced, data.nonlinear, 1.1, 100);
    }
    else {
        std::cerr << "Error: invalid direction flag.\n" << std::endl;
    }

    igl::slice_into(sol_unknown, data.u, col_colon, sol);
    igl::slice_into(known_val, data.k, col_colon, sol);
}

void mqwf_nonlinear_precompute(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXi& known,
    mqwf_dense_data& data,
    Eigen::VectorXi& nonlinearIds)
{
    using namespace std;
    using namespace Eigen;

    int n = A.rows();

    data.k.resize(known.size());
    data.k = known;

    // Compute unknown indices
    data.u.resize(n - data.k.size());
    {
        VectorXi known_sort;
        igl::sort(known, 1, true, known_sort);
        VectorXi tmp = VectorXi::LinSpaced(n, 0, n - 1);
        auto it = std::set_difference(tmp.data(), tmp.data() + tmp.size(),
            known_sort.data(), known_sort.data() + known_sort.size(),
            data.u.data());
        data.u.conservativeResize(std::distance(data.u.data(), it));
    }
    assert((data.u.size() + data.k.size()) == n);

    // Map nonlinear indices into reduced system
    Eigen::VectorXi nonlinearIds2(2);
    for (int i = 0; i < data.u.size(); i++) {
        if (data.u[i] == nonlinearIds(0))
            nonlinearIds2(0) = i;
        if (data.u[i] == nonlinearIds(2))
            nonlinearIds2(1) = i;
    }
    nonlinearIds = nonlinearIds2;

    // Extract submatrices
    MatrixXd Auu, Auk, Aku;
    Auu.resize(data.u.size(), data.u.size());
    igl::slice(A, data.u, data.u, Auu);
    igl::slice(A, data.u, data.k, Auk);
    igl::slice(A, data.k, data.u, Aku);

    // Store data
    data.Auu_pref = Auu.ldlt();
    data.Auu = Auu;
    data.Auk_plus_AkuT = Auk + Aku.transpose();
    data.n = n;
}

void mqwf_nonlinear_solve(
    const mqwf_dense_data& data,
    const Eigen::VectorXd& RHS,
    const Eigen::VectorXd& known_val,
    Eigen::VectorXd& sol,
    Eigen::VectorXi& nonlinearIds)
{
    using namespace std;
    using namespace Eigen;

    VectorXi col_colon = VectorXi::LinSpaced(RHS.cols(), 0, RHS.cols() - 1);

    // Construct reduced right-hand side
    VectorXd RHS_reduced;
    if (data.k.size() == 0) {
        RHS_reduced = -RHS;
    }
    else {
        VectorXd RHS_unknown;
        igl::slice(RHS, data.u, col_colon, RHS_unknown);
        RHS_reduced = -0.5 * data.Auk_plus_AkuT * known_val - RHS_unknown;
    }

    // Solve reduced system
    sol.resize(data.n, RHS.cols());

    VectorXd sol_unknown;
    sol_unknown.resize(data.u.size(), RHS.cols());
    sol_unknown = data.Auu_pref.solve(RHS_reduced);

    sol_unknown = solve_constrained_qp(
        data.Auu, RHS_reduced,
        nonlinearIds(0), -100, -0.1,
        nonlinearIds(1), 1.1, 100);

    igl::slice_into(sol_unknown, data.u, col_colon, sol);
    igl::slice_into(known_val, data.k, col_colon, sol);
}