//
// Created by Manitra Ranaivoharison on 19/06/2018.


#include <iostream>
#include <cmath>
#include <vector>
#include <math.h>
#include "MultiLayerPerceptron.h"
#include <algorithm>


void dllTrainMlpModel(double ***w, double **x, double **y, int couche[], int width, double param[]) {
    for (int v = 0; v < param[1]; v++) { // epoch
        for (int i = 0; i < param[6]; i++) {// parcours train
            // preparation X & delta
            double **xu = new double *[width];
            double **delta = new double *[width];
            for (int j = 0; j < width; j++) {
                xu[j] = new double[couche[j] + 1];
                delta[j] = new double[couche[j] + 1];
                for (int l = 0; l < couche[j] + 1; l++) {
                    xu[j][l] = 0;
                    delta[j][l] = 0;
                }
                xu[j][couche[j]] = 1;
            }

            for (int l = 0; l < couche[0]; ++l) {
                xu[0][l] = x[i][l];
            }
            xu[0][couche[0]] = 1;

            for (int layer = 1; layer < width; ++layer) {
                for (int h = 0; h < couche[layer] + 1; ++h) {
                    double sum = 0;
                    for (int k = 0; k < couche[layer - 1] + 1; ++k) {
                        sum += xu[layer - 1][k] * w[layer - 1][k][h];
                    }
                    if (h == couche[layer]) {
                        xu[layer][h] = 1;
                    } else {
                        xu[layer][h] = tanh(sum);
                    }
                }
            }

            // retropopagation
            for (int j = 0; j < couche[width - 1]; ++j) {
                delta[width - 1][j] = (1 - (xu[width - 1][j] * xu[width - 1][j])) * (xu[width - 1][j] - y[i][j]);
            }

            for (int l = width - 1; l > 0; --l) {
                for (int ln = 0; ln < couche[l - 1]; ++ln) {
                    double sum = 0;
                    for (int a = 0; a < couche[l]; ++a) {
                        sum += w[l - 1][ln][a] * delta[l][a];
                    }
                    delta[l - 1][ln] = (1 - (xu[l - 1][ln] * xu[l - 1][ln])) * sum;
                }
            }

            for (int l = 1; l < width; ++l) {
                for (int ln = 0; ln < couche[l - 1] + 1; ++ln) {
                    for (int rn = 0; rn < couche[l]; ++rn) {
                        w[l - 1][ln][rn] -= param[0] * xu[l - 1][ln] * delta[l][rn];
                    }
                }
            }

            //delete
            for (int z = 0; z < width; z++) {
                delete[] xu[z];
                delete[] delta[z];
            }
            delete[] xu;
            delete[] delta;
        }
    }
}


int dllPredictMlp(double ***w, double **x, int couche[], int width, double param[]) {
    double **xu = new double *[width];
    for (int j = 0; j < width; j++) {
        xu[j] = new double[couche[j] + 1];
        for (int l = 0; l < couche[j] + 1; l++) {
            xu[j][l] = 0;
        }
        xu[j][couche[j]] = 1;
    }
    for (int m = 0; m < couche[0]; ++m) {
        xu[0][m] = x[0][m];
    }
    xu[0][couche[0]] = 1;

    //propagation



    for (int j = 1; j < width; ++j) {
        for (int h = 0; h < couche[j] + 1; h++) {
            double sum = 0;
            for (int k = 0; k < couche[j - 1] + 1; ++k) {
                sum += xu[j - 1][k] * w[j - 1][k][h];
            }
            if (h == couche[j]) {
                xu[j][h] = 1;
            } else {
                xu[j][h] = tanh(sum);
            }
        }
    }

    int index = -999;
    for (int i = 0; i < couche[width - 1]; i++) {

        if (xu[width - 1][i] > 0) {
            index = i;
        }
    }

    //delete
    for (int z = 0; z < width; z++) {
        delete[] xu[z];
    }
    delete[] xu;

    return index;

}

