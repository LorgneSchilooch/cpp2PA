//
// Created by Qut on 17/06/2018.
//

#include "linearPerceptron.h"
#include <stdlib.h>
#include <time.h>


void dllTrainLinearModel(double *w, MatrixXd x, MatrixXd result, double param[]) {
    double min = -1;
    double max = 1;
    VectorXd wResult((int)param[3] + 1);
    wResult(0) = param[2];
    for (int i = 1; i < (int)param[3] + 1; i++) {
        wResult(i) = min + ((double)rand() / (double)RAND_MAX) * (max - min);
    }
    wResult << dllLearningLinearModel(wResult, x, result, param);
    for (int i = 0; i < (int)param[3] + 1; i ++) {
        w[i] = wResult(i);
    }
}

MatrixXd dllLearningLinearModel(VectorXd w, MatrixXd x, MatrixXd result, double param[]) {

    VectorXd xt(x.cols());
    bool verif = true;
    bool verif1 = false;
    int max = 0;
    while (verif) {
        for (int i = 0; i < x.rows(); i++) {
            xt = x.row(i);
            if (dllPredictLinearModel(w, xt, param) != result.coeff(i, 0)) {
                if (param[5] == 1) {
                    for (int k = 0; k < (int)param[3] + 1; k++) {
                        w(k) = w(k) + param[0] * (result.coeff(i, 0) - dllPredictLinearModel(w, xt, param)) * xt(k);
                    }
                } else {
                    for (int k = 0; k < (int)param[3] + 1; k++) {
                        w(k) = w(k) + param[0] * result.coeff(i, 0) * xt(k);
                    }
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

bool testAlgorithm(VectorXd w, VectorXd x, int result, double param[]) {

    return dllPredictLinearModel(w, x, param) == result;

};

int dllPredictLinearModel(VectorXd w, VectorXd x, double param[]) {
    double tmp = 0.;
    for (int i = 0; i < param[3] + 1; i++) {
        tmp += (w(i) * x(i));
    }
    return (tmp < 0) ? -1 : 1;
};