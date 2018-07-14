#include "h/interfaceWithCpp.h"

extern "C" {

////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// Linear perceptron /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


double *createLinearModel(int size) {
    double *w = new double[size];
    return w;
}

void deleteLinearModel(double *w) {
    delete w;
}

// Transform array on matrix
void trainLinearModel(double *w, double input[], double result[], double param[]) {

    MatrixXd xMat((int) param[6], (int) param[3] + 1);
    MatrixXd resultMat((int) param[6], 1);
    int tmp = 0;
    for (int i = 0; i < (int) param[6]; i++) {
        xMat.row(i).col(0) << 1;
        for (int k = 0; k < (int) param[3]; k++) {
            xMat.row(i).col(k + 1) << input[tmp + k];
        }
        resultMat.row(i).col(0) << result[i];
        tmp += (int) param[3];
    }


    dllTrainLinearModel(w, xMat, resultMat, param);
}

// Transform array on matrix
int predictLinearModel(double *w, double input[], double param[]) {

    VectorXd xp((int) param[3] + 1);
    VectorXd wp((int) param[3] + 1);
    xp(0) = 1;
    wp(0) = w[0];
    for (int i = 1; i < (int) param[3] + 1; i++) {
        xp(i) = input[i - 1];
        wp(i) = w[i];
    }


    return dllPredictLinearModel(wp, xp, param);

}

// Transform array on matrix
void trainLinearModelRl(double *w, double input[], double result[], double param[]) {
    MatrixXd xMat((int) param[6], (int) param[3] + 1);
    MatrixXd resultMat((int) param[6], 1);
    int tmp = 0;
    for (int i = 0; i < (int) param[6]; i++) {
        xMat.row(i).col(0) << 1;
        for (int k = 0; k < (int) param[3]; k++) {
            xMat.row(i).col(k + 1) << input[tmp + k];
        }
        resultMat.row(i).col(0) << result[i];
        tmp += (int) param[3];
    }


    dllTrainLinearModelRl(w, xMat, resultMat, param);
}

// Transform array on matrix
double predictLinearModelRl(double *w, double input[], double param[]) {

    VectorXd xp((int) param[3] + 1);
    VectorXd wp((int) param[3] + 1);
    xp(0) = 1;
    wp(0) = w[0];
    for (int i = 1; i < (int) param[3] + 1; i++) {
        xp(i) = input[i - 1];
        wp(i) = w[i];
    }


    return dllPredictLinearModelRl(wp, xp, param);

}

}