#ifndef CPP2PA_LIBRARY_H
#define CPP2PA_LIBRARY_H

#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

void dllTrainLinearModel(double *, MatrixXd, MatrixXd, double, bool);

MatrixXd dllLearningLinearModel(VectorXd, MatrixXd, MatrixXd, double, bool);

bool testAlgorithm(MatrixXd, MatrixXd, int result);

int dllPredictLinearModel(VectorXd, VectorXd);


#endif



#endif