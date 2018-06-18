//
// Created by Qut on 17/06/2018.
//

#ifndef CPP2PA_SIMPLEPERCEPTRON_H
#define CPP2PA_SIMPLEPERCEPTRON_H


#include "../Eigen/Dense"

using namespace Eigen;
using namespace std;

void dllTrainLinearModel(double *, MatrixXd, MatrixXd, double[], bool);

MatrixXd dllLearningLinearModel(VectorXd, MatrixXd, MatrixXd, double[], bool);

bool testAlgorithm(MatrixXd, MatrixXd, int result);

int dllPredictLinearModel(VectorXd, VectorXd);




#endif //CPP2PA_SIMPLEPERCEPTRON_H
