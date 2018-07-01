#include "library.h"
#include "LinearPerceptron/linearPerceptron.h"
#include "MultiLayerPerceptron/MultiLayerPerceptron.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <time.h>

extern "C" {

/////////////////////// LP

double *createLinearModel(int size) {
    double *w = new double[size];
    return w;
}

void deleteLinearModel(double *w) {
    delete w;
}


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


////////////////////// MLP

double ***createMlpModel(int maxHeight, int width) {
    double ***w;
    w = new double **[width];
    for (int x = 0; x < width; x++) {
        w[x] = new double *[maxHeight + 1];
        for (int y = 0; y < maxHeight + 1; y++) {
            w[x][y] = new double[maxHeight + 1];
            for (int z = 0; z < maxHeight + 1; z++) {
                w[x][y][z] = (rand() / (RAND_MAX / (2.0))) - 1.0;
            }
        }
    }
    return w;
}

void deleteMlpModel(double ***w, int width) {
    for (int i = 0; i < width; i++) {
        delete[] w[i];
    }
    delete[] w;
}

void trainMlpModel(double ***w, double input[], double result[], int couche[], int width, int maxHeight, double param[]) {

    int count = 0;
    double **x = new double *[(int) param[6]];
    for (int i = 0; i < ((int) param[6]); i++) {
        x[i] = new double[couche[0]];
        for (int j = 0; j < couche[0]; j++) {
            x[i][j] = input[count + j];
        }
        count += couche[0];
    }

    count = 0;
    double **y = new double *[(int) param[6]];
    for (int i = 0; i < ((int) param[6]); i++) {
        y[i] = new double[couche[width - 1]];
        for (int j = 0; j < couche[width - 1]; j++) {
            y[i][j] = result[count + j];
        }
        count += couche[width - 1];
    }

    dllTrainMlpModel(w, x, y, couche, width, param);

    for (int i = 0; i < ((int) param[6]); i++) {
        delete[] x[i];
        delete[] y[i];
    }
    delete[] x;
    delete[] y;


}


double predictMlpModel(double ***w, double input[], int couche[], int width, double param[]) {
    int count = 0;
    double **x = new double *[1];
    x[0] = new double[couche[0]];
    for (int j = 0; j < couche[0]; j++) {
        x[0][j] = input[j];
    }
    double value = dllPredictMlp(w, x, couche, width, param);


    delete[] x[0];
    delete[] x;

    return value;

}

}