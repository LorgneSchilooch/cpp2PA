#include "h/RBF.h"
#include <cstdlib>
#include <vector>
#include <cfloat>
#include <fstream>

extern "C" {
double *createNaiveRbfModel(int size) {
    double *w = new double[size];
    return w;
}

void deleteNaiveRbfModel(double *w, double *center) {
    delete w;
    delete center;
}

void trainNaiveRbfModel(double *w, double input[], double result[], double param[]) {
    int sizeTrain = (int) param[6];
    double gamma = param[9];
    int NumberInput = (int) param[3];
    double distance;
    VectorXd wResult;
    MatrixXd teta(sizeTrain, sizeTrain);
    MatrixXd y(sizeTrain, 1);
    for (int i = 0; i < sizeTrain; i++) {
        y(i, 0) = result[i];
    }
    for (int i = 0; i < sizeTrain; ++i) {
        for (int j = 0; j < sizeTrain; ++j) {
            distance = 0;
            for (int k = 0; k < NumberInput; k++) {
                distance += (input[j * NumberInput + k] - input[i * NumberInput + k]) * (input[j * NumberInput + k] - input[i * NumberInput + k]);
            }
            teta(i, j) = exp(-gamma * distance);
        }
    }
    wResult = teta.inverse() * y;
    for (int i = 0; i < sizeTrain; i++) w[i] = wResult(i);

}

double predictNaiveRbf(double *w, double input[], double inputs[], double param[]) {
    int sizeTrain = (int) param[6];
    double gamma = param[9];
    int NumberInput = (int) param[3];
    bool classification = (param[8] == 1);
    double sum = 0;
    VectorXd wResult((int) sizeTrain);
    for (int i = 0; i < (int) sizeTrain; i++) wResult(i) = w[i];
    double *xX = new double[(int) NumberInput];

    for (int i = 0; i < sizeTrain; i++) {
        for (int j = 0; j < (int) NumberInput; j++) {
            xX[j] = inputs[i * (int) NumberInput + j];
        }
        sum += (wResult)(i, 0) * exp(-gamma * pow(distance(xX, input, NumberInput), 2));
    }
    delete[] xX;
    return (classification == 1) ? ((sum > 0) ? 1 : 0) : sum;
}

double *createRbfModel(int size) {
    double *w = new double[size];
    return w;
}

void deleteRbfModel(double *w) {
    delete w;
}

double *trainRbfModel(double *w, double inputs[], double result[], double param[]) {

    int select = (int) param[10];
    int sizeTrain = (int) param[6];
    double gamma = param[9];
    int numberInput = (int) param[3];
    MatrixXd wResult(numberInput, 1);
    MatrixXd mInputs(sizeTrain, numberInput);
    MatrixXd selected(select, numberInput);

    for (int i = 0; i < sizeTrain; i++) {
        for (int j = 0; j < numberInput; j++) {
            mInputs.row(i).col(j) << inputs[i * numberInput + j];
        }
    }
    for (int i = 0; i < select; i++) {
        for (int j = 0; j < numberInput; j++) {
            selected.row(i).col(j) << (double) rand() / RAND_MAX;
        }
    }
    double lloyd[sizeTrain][2];
    for (int i = 0; i < select; i++) {
    }

//lloyd
    int iter = 0;
    VectorXd rSelected(selected.cols());
    VectorXd rInputs(mInputs.cols());
    while (iter < param[1]) {
        // Calcul distance
        for (int k = 0; k < sizeTrain; k++) {
            lloyd[k][1] = DBL_MAX;
        }
        for (int i = 0; i < select; i++) {
            rSelected = selected.row(i);
            for (int j = 0; j < sizeTrain; j++) {
                rInputs = mInputs.row(j);
                double distance = (rSelected - rInputs).cwiseAbs().sum();
                if (lloyd[j][1] > distance) {
                    lloyd[j][1] = distance;
                    lloyd[j][0] = i;
                }

            }
        }
        // Mettre à jour l'emplacement des select
        for (int i = 0; i < select; i++) {
            int count = 0;
            VectorXd vCount(mInputs.cols());
            for (int j = 0; j < sizeTrain; j++) {
                if (lloyd[j][0] == i) {
                    VectorXd rr = mInputs.row(j);
                    vCount += rr;
                    count++;
                }
            }
            if (count != 0) selected.row(i) = vCount / count;
        }
        iter++;
    }
    Eigen::MatrixXd teta(sizeTrain, select);
    Eigen::MatrixXd y(sizeTrain, 1);
    for (int i = 0; i < sizeTrain; i++) {
        y(i, 0) = result[i];
    }
    for (int i = 0; i < sizeTrain; i++) {
        rInputs = mInputs.row(i);
        for (int j = 0; j < select; j++) {
            rSelected = selected.row(j);
            double distance = pow((rInputs - rSelected).cwiseAbs().sum(), 2);
            teta(i, j) = exp(-gamma * distance);
        }
    }
    MatrixXd tetaT = teta.transpose();
    wResult = (tetaT * teta).inverse() * tetaT * y;
    for (int i = 0; i < select; i++) {
        VectorXd wR = wResult.row(i);
        w[i] = wR(0);
    }
    double *center = new double[selected.cols() * selected.rows()];
    for (int i = 0; i < selected.rows(); i++) {
        for (int j = 0; j < selected.cols(); j++) {
            VectorXd vselect = selected.row(i);
            center[i * selected.cols() + j] = vselect(j);
        }
    }
    return center;
}

double predictRbfModel(double *w, double *center, double inputs[], double param[]) {
    int select = (int) param[10];
    double gamma = param[9];
    int numberInput = (int) param[3];
    bool classification = (param[8] == 1);
    double sum = 0;

    VectorXd mInputs(numberInput);
    MatrixXd selected(select, numberInput);
    VectorXd wResult(select);


    for (int j = 0; j < numberInput; j++) mInputs(j) = inputs[j];

    for (int i = 0; i < select; i++) {
        for (int j = 0; j < numberInput; j++) {
            selected.row(i).col(j) << center[i * numberInput + j];
        }
    }

    for (int i = 0; i < select; i++) wResult(i) = w[i];
    for (int i = 0; i < select; i++) {
        VectorXd vS = selected.row(i);
        double distance = (mInputs - vS).cwiseAbs().sum();
        sum += wResult(i) * exp(-gamma * pow(distance, 2));

    }

    return (classification == 1) ? ((sum > 0) ? 1 : 0) : sum;

}


}

double distance(double *first, double *second, int inputSize) {
    double distance = 0;
    for (int i = 0; i < inputSize; i++) {
        distance += (second[i] - first[i]) * (second[i] - first[i]);
    }
    return sqrt(distance);
}