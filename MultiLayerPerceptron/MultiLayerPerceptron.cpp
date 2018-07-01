//
// Created by Manitra Ranaivoharison on 19/06/2018.


#include <iostream>
#include <cmath>
#include <vector>
#include <math.h>
#include "MultiLayerPerceptron.h"
#include <algorithm>
#include <fstream>

std::ofstream out;




void dllTrainMlpModel(double ***w, double **x, double **y, int couche[], int width, double param[]) {
    for (int v = 0; v < param[1]; v++) { // epoch
        for (int i = 0; i < param[6]; i++) {// parcours train
            // preparation X & delta
            double **xu = new double *[width];
            double **delta = new double *[width];
            for (int j = 0; j < width; j++) {
                xu[j] = new double[couche[j] + 1];
                delta[j] = new double[couche[j] + 1];
                if (j == 0) {
                    for (int k = 0; k < couche[0] + 1; k++) {
                        xu[0][k] = (k == 0) ? 1. : x[i][k];
                    }
                } else {
                    xu[j][0] = 1.;
                }
            }
            //propagation
            for (int j = 1; j < width; j++) {
                double temp = 0.;
                for (int k = 1; k < (couche[j] + 1); k++) {
                    for (int l = 0; l < (couche[j - 1] + 1); l++) {
                        temp += w[j][k][l] * xu[j - 1][l];
                    }
                    xu[j][k] = tanh(temp);
                }
            }
            // retropopagation)
            for (int j = 1; j < (couche[width - 1] + 1); j++) {
                delta[width - 1][j] = (1. - (xu[width - 1][j] * xu[width - 1][j])) * (xu[width - 1][j] - y[i][j]);
            }
/*            for (int j = width - 2; j < 1; j--) {
                for (int k = 0; k < (couche[j] + 1); k++) {
                    double sum = 0.;
                    for (int l = 1; l < (couche[j + 1] + 1); l++) {
                        sum += delta[j + 1][l] * w[j + 1][l][k];
                    }
                    delta[j][k] = xu[j][k] * (1. - (xu[j][k] * xu[j][k])) * sum;
                }
            }*/
            // calcul des W
            for (int j = 1; j < width; j++) {
                for (int k = 1; k < (couche[j] + 1); k++) {
                    for (int l = 0; l < couche[l - 1] + 1; l++) {
                        w[j][k][l] = w[j][k][l] - (param[0] * xu[j - 1][l] * delta[j][k]);

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


double dllPredictMlp(double ***w, double **x, int couche[], int width, double param[]) {
    // preparation X & delta
    double **xu = new double *[width];
    double **delta = new double *[width];
    for (int j = 0; j < width; j++) {
        xu[j] = new double[couche[j] + 1];
        delta[j] = new double[couche[j] + 1];
        if (j == 0) {
            for (int k = 0; k < couche[0] + 1; k++) {
                xu[0][k] = (k == 0) ? 1 : x[0][k];
            }
        } else {
            xu[j][0] = 1;
        }
    }
    //propagation
    for (int j = 1; j < width; j++) {
        double temp = 0.;
        for (int k = 1; k < (couche[j] + 1); k++) {
            for (int l = 0; l < (couche[j - 1] + 1); l++) {
                temp += w[j][k][l] * xu[j - 1][l];
            }
            xu[j][k] = tanh(temp);
            // std::ios::app is the open mode "append" meaning
// new data will be written to the end of the file.
            out.open("myfile.txt", std::ios::app);

            out << xu[j][k] << '\n';
            out.close();
        }
    }

    double index = xu[width -1][1];
    out.open("myfile1.txt", std::ios::app);

    out << xu[width -1][0] << " --- " << xu[width -1][1] << '\n';
    out.close();

    //delete
    for (int z = 0; z < width; z++) {
        delete[] xu[z];
        delete[] delta[z];
    }
    delete[] xu;
    delete[] delta;

    return index;

}

