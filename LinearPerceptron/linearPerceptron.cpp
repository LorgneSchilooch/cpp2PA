#include <stdlib.h>
#include <time.h>
#include "h/linearPerceptron.h"

// Classification
void dllTrainLinearModel(double *w, MatrixXd x, MatrixXd result, double param[]) {
    VectorXd wResult((int) param[3] + 1);
    wResult(0) = param[2];
    for (int i = 1; i < (int) param[3] + 1; i++) {
        wResult(i) = (rand() / (RAND_MAX / (2.0))) - 1.0;
    }
    wResult << dllLearningLinearModel(wResult, x, result, param);
    for (int i = 0; i < (int) param[3] + 1; i++) {
        w[i] = wResult(i);
    }
}

MatrixXd dllLearningLinearModel(VectorXd w, MatrixXd x, MatrixXd result, double param[]) {
    VectorXd xt(x.cols());
    bool verif = true;
    bool verif1 = false;
    for (int v = 0; v < param[1]; v++) {
        for (int i = 0; i < x.rows(); i++) {
            xt = x.row(i);
            if (dllPredictLinearModel(w, xt, param) != result.coeff(i, 0)) {
                if (param[5] == 1) {
                    for (int k = 0; k < (int) param[3] + 1; k++) {
                        w(k) = w(k) + param[0] * (result.coeff(i, 0) - dllPredictLinearModel(w, xt, param)) * xt(k);
                    }
                } else {
                    for (int k = 0; k < (int) param[3] + 1; k++) {
                        w(k) = w(k) + param[0] * result.coeff(i, 0) * xt(k);
                    }
                }
            }
        }
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

void dllTrainLinearModelRl(double *w, MatrixXd input, MatrixXd output, double param[]) {

    MatrixXd we = ((input.transpose() * input).ldlt().solve(input.transpose())) * output; //Merci Pierre

    for (int i = 0; i < (int) param[3] + 1; i++) {
        w[i] = we(i);
    }

}

double dllPredictLinearModelRl(VectorXd w, VectorXd x, double param[]) {
    double tmp = 0.;
    for (int i = 0; i < param[3] + 1; i++) {
        tmp += (w(i) * x(i));
    }
    return tmp;
};