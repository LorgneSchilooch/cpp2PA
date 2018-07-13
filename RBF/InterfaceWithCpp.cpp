//
// Created by Qut on 13/07/2018.
//

#include "h/InterfaceWithCpp.h"

extern "C" {


////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// RBF  //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


double *creatRbfNaiveModel(int size) {
    double *w = new double[size];
    return w;
}

void trainRbfRl(double *w, double input[], double result[], double param[]) {
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


    //dllTrainLinearModelRl(w, xMat, resultMat, param);
}

}