#include "library.h"
#include "LinearPerceptron/linearPerceptron.h"
#include "MLP/Neuron.h"
#include "MLP/MLP.h"
#include <iostream>
#include <cmath>
#include <vector>

extern "C" {

double *createLinearModel() {
    double *w = new double[3];
    return w;
}

void deleteLinearModel(double *w) {
    delete w;
}


void trainLinearModel(double *w, double x[], double y[], double result[], double param[], int len, bool rosenblatt) {

    MatrixXd xMat(len, 3);
    MatrixXd resultMat(len, 1);
    for (int i = 0; i < len; i++) {
        xMat.row(i).col(0) << 1;
        xMat.row(i).col(1) << x[i];
        xMat.row(i).col(2) << y[i];
        resultMat.row(i).col(0) << result[i];
    }

    dllTrainLinearModel(w, xMat, resultMat, param, rosenblatt);
}

int predictLinearModel(double *w, double x, double y) {

    VectorXd xp(3);
    VectorXd wp(3);
    xp(0) = 1;
    xp(1) = x;
    xp(2) = y;
    wp(0) = w[0];
    wp(1) = w[1];
    wp(2) = w[2];
    return dllPredictLinearModel(wp, xp);

}

Neuron **createMLPModel(int layer[], double param[], int layerLen, int paramLen) {
    Neuron **neuronArrayByLayer = new Neuron*[layerLen];
    for (int i = 0; i < layerLen; i++) {
        neuronArrayByLayer[i] = new Neuron[layer[i]];
    }



return neuronArrayByLayer;
}


void deleteMLPModel(Neuron *neuronArrayByLayer) {
    delete neuronArrayByLayer;
}


void trainMLPModel(Neuron **neuronArrayByLayer, std::vector<double> pt[], double result[], double param[], int layer[], int layerLen,  int len) {


    dllTrainMLPModel(neuronArrayByLayer, pt[len], layer[layerLen], param[4]);
}

int predictMLPModel(double *w, double x, double y) {

    VectorXd xp(3);
    VectorXd wp(3);
    xp(0) = 1;
    xp(1) = x;
    xp(2) = y;
    wp(0) = w[0];
    wp(1) = w[1];
    wp(2) = w[2];
    return dllPredictLinearModel(wp, xp);

}



}




