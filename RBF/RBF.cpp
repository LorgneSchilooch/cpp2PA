//
// Created by Qut on 07/07/2018.
//

#include "h/RBF.h"

extern "C" {


////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// RBF  //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

double *createNaiveRbfModel(int size) {
    double *w = new double[size];
    return w;
}

void deleteNaiveRbfModel(double *w) {
    delete w;
}

void trainNaiveRbfModel(double *w, double input[], double result[], double param[]) {
    int sizeTrain = (int) param[6];
    double gamma = (int) param[9];
    int NumberInput = param[3];
    double distance;
    VectorXd wResult;
    Eigen::MatrixXd teta(sizeTrain, sizeTrain);
    Eigen::MatrixXd y(sizeTrain, 1);
    for (int i = 0; i < sizeTrain; i++) {
        y(i, 0) = result[i];
    }
    for (int i = 0; i < sizeTrain; ++i) {
        for (int j = 0; j < sizeTrain; ++j) {
            distance = 0;
            for (int k = 0; k < NumberInput; k++) {
                distance += (input[j * (int) NumberInput + k] - input[i * (int) NumberInput + k]) * (input[j * (int) NumberInput + k] - input[i * (int) NumberInput + k]);
            }
            teta(i, j) = exp(-gamma * distance);
        }
    }
    wResult = teta.inverse() * y;
    for (int i = 0; i < sizeTrain; i++) w[i] = wResult(i);

}

double predictNaiveRbf(double *w, double input[], double inputs[], double param[]) {
    int sizeTrain = (int) param[6];
    double gamma = (int) param[9];
    int NumberInput = param[3];
    bool classification = (param[8] == 1);
    double sum = 0;
    VectorXd wResult((int) sizeTrain);
    for (int i = 0; i < (int) sizeTrain; i++) wResult(i) = w[i];
    double *xX = new double[(int) NumberInput];
    for (int i = 0; i < sizeTrain; i++) {
        for (int j = 0; j < (int) NumberInput; j++) {
            xX[j] = inputs[i * (int) NumberInput + j];
        }
        sum += (wResult)(i, 0) * exp(-gamma * distance(input, xX, NumberInput) * distance(input, xX, NumberInput));
    }
    return (classification == 1) ? ((sum > 0) ? 1 : -1) : sum;
}

}

double distance(double *first, double *second, int inputSize) {
    double distance = 0;
    for (int i = 0; i < inputSize; i++) {
        distance += (second[i] - first[i]) * (second[i] - first[i]);
    }
    return sqrt(distance);
}