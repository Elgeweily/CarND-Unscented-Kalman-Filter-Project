#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd accum(4);
	accum << 0, 0, 0, 0, 0;

	VectorXd res(5);
	VectorXd ressq(5);
	VectorXd mean(5);
	VectorXd rmse(5);

	if (estimations.size() == 0) {
		cout << "error...no estimations" << endl;
	}

	if (estimations.size() != ground_truth.size()) {
		cout << "error...estimations and ground truth must have same size" << endl;
	}

	// accumulate squared residuals
	for (int i = 0; i < estimations.size(); ++i) {
		res = estimations[i] - ground_truth[i];
		ressq = res.array() * res.array();
		accum = accum + ressq;
	}

	mean = accum.array() / estimations.size();
	rmse = mean.array().sqrt();
	return rmse;
}