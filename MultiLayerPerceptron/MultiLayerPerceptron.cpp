#include <cmath>

#include "h/MultiLayerPerceptron.h"

void dllTrainMlpModel(double ***w, double **x, double **y, int couche[], double param[]) {
    int width = (int) param[7];
    int sizeTrain = (int) param[6];
    int iterMax = (int) param[1];
    bool classification = (param[8] == 1);
    double alpha = param[0];
    for (int v = 0; v < iterMax; v++) { // nombre d'iter
        for (int i = 0; i < sizeTrain; i++) { // parcourir tous les inputs
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

            for (int l = 0; l < couche[0]; l++) {
                xu[0][l] = x[i][l];
            }
            xu[0][couche[0]] = 1;

            // propagation
            for (int layer = 1; layer < width; layer++) {
                for (int rHeight = 0; rHeight < couche[layer] + 1; rHeight++) {
                    double sum = 0;
                    for (int lHeight = 0; lHeight < couche[layer - 1] + 1; lHeight++) {
                        sum += xu[layer - 1][lHeight] * w[layer - 1][lHeight][rHeight];
                    }
                    if (rHeight == couche[layer]) {
                        xu[layer][rHeight] = 1;
                    } else {
                        if (classification) xu[layer][rHeight] = tanh(sum);
                        else xu[layer][rHeight] = (layer == width - 1) ? sum : tanh(sum);
                    }
                }
            }

            // retropopagation
            for (int lastHeight = 0; lastHeight < couche[width - 1]; lastHeight++) {
                delta[width - 1][lastHeight] = (classification) ? (1 - pow(xu[width - 1][lastHeight], 2)) * (xu[width - 1][lastHeight] - y[i][lastHeight]) : 1;
                delta[width - 1][lastHeight] *= (xu[width - 1][lastHeight] - y[i][lastHeight]);
            }

            for (int layer = width - 1; layer > 0; layer--) {
                for (int lHeight = 0; lHeight < couche[layer - 1]; lHeight++) {
                    double sum = 0;
                    for (int rHeight = 0; rHeight < couche[layer]; rHeight++) {
                        sum += w[layer - 1][lHeight][rHeight] * delta[layer][rHeight];
                    }
                    delta[layer - 1][lHeight] = (1 - (xu[layer - 1][lHeight] * xu[layer - 1][lHeight])) * sum;
                }
            }

            // new w
            for (int layer = 1; layer < width; layer++) {
                for (int lHeight = 0; lHeight < couche[layer - 1] + 1; lHeight++) {
                    for (int rHeight = 0; rHeight < couche[layer]; rHeight++) {
                        w[layer - 1][lHeight][rHeight] -= alpha * xu[layer - 1][lHeight] * delta[layer][rHeight];
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

double dllPredictMlp(double ***w, double **x, int couche[], double param[]) {
    bool classification = (param[8] == 1);
    int width = (int) param[7];
    double **xu = new double *[width];
    for (int j = 0; j < width; j++) {
        xu[j] = new double[couche[j] + 1];
        for (int l = 0; l < couche[j] + 1; l++) {
            xu[j][l] = 0;
        }
        xu[j][couche[j]] = 1;
    }
    for (int m = 0; m < couche[0]; m++) {
        xu[0][m] = x[0][m];
    }
    xu[0][couche[0]] = 1;

    //propagation
    for (int layer = 1; layer < width; layer++) {
        for (int rHeight = 0; rHeight < couche[layer] + 1; rHeight++) {
            double sum = 0;
            for (int lHeight = 0; lHeight < couche[layer - 1] + 1; lHeight++) {
                sum += xu[layer - 1][lHeight] * w[layer - 1][lHeight][rHeight];
            }
            if (rHeight == couche[layer]) {
                xu[layer][rHeight] = 1;
            } else {
                if (classification) xu[layer][rHeight] = tanh(sum);
                else xu[layer][rHeight] = (layer == width - 1) ? sum : tanh(sum);
            }
        }
    }

    double value = -999999999999;
    if (classification) {
        for (int i = 0; i < couche[width - 1]; i++) {
            if (xu[width - 1][i] > 0) {
                value = i;
            }
        }
    } else value = xu[width - 1][0];

    //delete
    for (int z = 0; z < width; z++) {
        delete[] xu[z];
    }
    delete[] xu;

    return value;
}