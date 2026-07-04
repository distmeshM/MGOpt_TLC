#ifndef MQWF_DENSE_H
#define MQWF_DENSE_H

#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/colon.h>
#include <igl/sort.h>
#include <igl/LinSpaced.h>

#include <intersect_ordered.h>

#include <vector>
#include <iostream>
#include <math.h>

#include <mqwf_dense_data.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include <CGAL/Gmpq.h>
#include <mex.h>

// see min_quad_with_fixed in libigl, this is just a dense version for small A
void mqwf_dense_precompute(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXi& known,
	mqwf_dense_data& data);

// template<typename Derived>
// void mqwf_dense_solve(
//   const mqwf_dense_data & data,
//   const Eigen::MatrixBase<Derived> & RHS,
//   const Eigen::MatrixBase<Derived> & known_val,
//   Eigen::MatrixBase<Derived> & sol);

void mqwf_dense_solve(
	const mqwf_dense_data& data,
	const Eigen::VectorXd& RHS,
	const Eigen::VectorXd& known_val,
	Eigen::VectorXd& sol);


void mqwf_constraint_solve(
	const Eigen::MatrixXd& A,
	const Eigen::VectorXd& RHS,
	const Eigen::MatrixXi& known,
	const Eigen::VectorXd& known_val,
	Eigen::VectorXd& sol,
	int nonlinearId, int dir);


void mqwf_nonlinear_precompute(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXi& known,
	mqwf_dense_data& data, int nonlinearId, int dir);


void mqwf_nonlinear_solve(
	const mqwf_dense_data& data,
	const Eigen::VectorXd& RHS,
	const Eigen::VectorXd& known_val,
	Eigen::VectorXd& sol, int dir);


void mqwf_nonlinear_precompute(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXi& known,
	mqwf_dense_data& data, Eigen::VectorXi& nonlinearIds);


void mqwf_nonlinear_solve(
	const mqwf_dense_data& data,
	const Eigen::VectorXd& RHS,
	const Eigen::VectorXd& known_val,
	Eigen::VectorXd& sol, Eigen::VectorXi& nonlinearIds);



#endif