//
// Created by Qut on 25/06/2018.
//

#ifndef CPP2PA_MULTILAYERPERCEPTRON_H
#define CPP2PA_MULTILAYERPERCEPTRON_H

#include "../Eigen/Dense"

using namespace Eigen;
using namespace std;


void dllTrainMlpModel(double ***, double **, double **, int[], int, double[]);


double dllPredictMlp(double***, double**, int[], int, double[]);

#endif //CPP2PA_MULTILAYERPERCEPTRON_H
