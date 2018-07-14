#ifndef CPP2PA_SIMPLEPERCEPTRON_H
#define CPP2PA_SIMPLEPERCEPTRON_H

#include "../../Eigen/Dense"

using namespace Eigen;
using namespace std;

void dllTrainLinearModel(double *, MatrixXd, MatrixXd, double[]);

MatrixXd dllLearningLinearModel(VectorXd, MatrixXd, MatrixXd, double[]);

bool testAlgorithm(MatrixXd, MatrixXd, int result, double[]);

int dllPredictLinearModel(VectorXd, VectorXd, double[]);

void dllTrainLinearModelRl(double *, MatrixXd, MatrixXd, double[]);

double dllPredictLinearModelRl(VectorXd, VectorXd, double[]);

#endif //CPP2PA_SIMPLEPERCEPTRON_H