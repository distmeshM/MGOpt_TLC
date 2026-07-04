#ifndef MG_PRECOMPUTE_BLOCK_H
#define MG_PRECOMPUTE_BLOCK_H

#include <vector>
#include <iostream>
#include <math.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
// 如果Eigen未定义VectorXb，手动定义
#if !defined(EIGEN_VECTORXB_H) && !defined(Eigen_VectorXb_H)  // 兼容不同Eigen版本
#define EIGEN_VECTORXB_H
using VectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
#endif


#include <mg_data.h>
#include <get_prolong.h>

void mg_precompute_block(
	const Eigen::MatrixXd & Vf,
	const Eigen::MatrixXi & Ff,
  std::vector<mg_data> & mg,
	VectorXb& constrained);

void mg_precompute_block(
	const Eigen::MatrixXd & Vf,
	const Eigen::MatrixXi & Ff,
	const int & dec_type,
  std::vector<mg_data> & mg,
	VectorXb& constrained);

void mg_precompute_block(
	const Eigen::MatrixXd & Vf,
	const Eigen::MatrixXi & Ff,
	const float & ratio,
	const int & nVCoarsest,
	const int & dec_type,
  std::vector<mg_data> & mg,
	VectorXb& constrained);
#endif