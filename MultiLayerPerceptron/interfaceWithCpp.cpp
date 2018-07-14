//
// Created by Qut on 13/07/2018.
//

#include "h/interfaceWithCpp.h"

extern "C" {




////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// Multi layer perceptron  ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////



double ***createMlpModel(int couche[], int width) {
    double ***w;
    w = new double **[width - 1];
    for (int layer = 0; layer < width - 1; layer++) {
        w[layer] = new double *[couche[layer] + 1]; // biais
        for (int ln = 0; ln < couche[layer] + 1; ln++) {
            w[layer][ln] = new double[couche[layer + 1]];
            for (int rn = 0; rn < couche[layer + 1]; ++rn) {
                w[layer][ln][rn] = ((double) rand()) / ((double) RAND_MAX) * 2.0 - 1.0;
            }
        }
    }

    return w;
}

void deleteMlpModel(double ***w, int couche[], int width) {
    for (int i = 0; i < width - 1; i++) {
        for (int j = 0; j < couche[i] + 1; j++) {
            delete[] w[i][j];
        }
        delete[] w[i];
    }
    delete[] w;
}

void trainMlpModel(double ***w, double input[], double result[], int couche[], double param[]) {
    int width = (int)param[7];
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

    dllTrainMlpModel(w, x, y, couche, param);

    for (int i = 0; i < ((int) param[6]); i++) {
        delete[] x[i];
        delete[] y[i];
    }
    delete[] x;
    delete[] y;
}


double predictMlpModel(double ***w, double input[], int couche[], double param[]) {
    double **x = new double *[1];
    x[0] = new double[couche[0]];
    for (int j = 0; j < couche[0]; j++) {
        x[0][j] = input[j];
    }
    double value = dllPredictMlp(w, x, couche,  param);
    delete[] x[0];
    delete[] x;

    return value;

}

}