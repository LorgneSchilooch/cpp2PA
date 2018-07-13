//
// Created by Qut on 07/07/2018.
//

#include "h/RBF.h"

void dllTrainRbfRl(double *w, double input[], double result[], double param[]) {

    double distance;
    Eigen::MatrixXd teta(nbExamples, nbExamples);
    Eigen::MatrixXd Y(nbExamples, 1);
    for (int i = 0; i<nbExamples; i++) {
        Y(i, 0) = result[i];
    }
    for (int i = 0; i < nbExamples; ++i) {
        for (int j = 0; j < nbExamples; ++j) {
            distance = 0;
            for (int k = 0; k<inputSize; k++) {
                distance += (X[j*inputSize + k] - X[i*inputSize + k])*(X[j*inputSize + k] - X[i*inputSize + k]);
            }
            teta(i, j) = exp(-gamma*distance);
        }
    }
    naiveWeights = teta.inverse()*Y;


}