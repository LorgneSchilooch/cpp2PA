#include "library.h"
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

}

// ######################### Fonctions internes #########################


void dllTrainLinearModel(double *w, MatrixXd x, MatrixXd result, double param[], bool choice) {

    VectorXd wResult(3);
    wResult(0) = param[2];
    wResult(1) = -0.7; // A mettre en random
    wResult(2) = 0.2; // A mettre en random
    wResult << dllLearningLinearModel(wResult, x, result, param, choice);
    w[0] = wResult(0);
    w[1] = wResult(1);
    w[2] = wResult(2);
}

MatrixXd dllLearningLinearModel(VectorXd w, MatrixXd x, MatrixXd result, double param[], bool choice) {

    VectorXd xt(x.cols());
    bool verif = true;
    bool verif1 = false;
    int max = 0;
    while (verif) {
        for (int i = 0; i < x.rows(); i++) {
            xt = x.row(i);
            if (dllPredictLinearModel(w, xt) != result.coeff(i, 0)) {
                if (choice) {
                    w(0) = w(0) + param[0] * (result.coeff(i, 0) - dllPredictLinearModel(w, xt)) * xt(0);
                    w(1) = w(1) + param[0] * (result.coeff(i, 0) - dllPredictLinearModel(w, xt)) * xt(1);
                    w(2) = w(2) + param[0] * (result.coeff(i, 0) - dllPredictLinearModel(w, xt)) * xt(2);
                } else {
                    w(0) = w(0) + param[0] * result.coeff(i, 0) * xt(0);
                    w(1) = w(1) + param[0] * result.coeff(i, 0) * xt(1);
                    w(2) = w(2) + param[0] * result.coeff(i, 0) * xt(2);
                }
                verif1 = true;
            }
        }
        verif = verif1;
        verif1 = false;
        max++;
        if (max == param[1]) break;
    }
    return w;
}

bool testAlgorithm(VectorXd w, VectorXd x, int result) {

    return dllPredictLinearModel(w, x) == result;

};

int dllPredictLinearModel(VectorXd w, VectorXd x) {
    return (w(0) * x(0) + w(1) * x(1) + w(2) * x(2)) < 0 ? -1 : 1;
};


