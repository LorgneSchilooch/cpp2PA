//
// Created by Qut on 17/06/2018.
//

#include<vector>
#include "Neuron.h"

#ifndef CPP2PA_MLP_H
#define CPP2PA_MLP_H

void dllTrainMLPModel(Neuron **neuronArrayByLayer, std::vector<double> pt[], double layer[], int layerLen, double Result[], int len, double param[]);
void dllSpiderMan(Neuron **neuronArrayByLayer, double layer[], int layerLen);


#endif //CPP2PA_MLP_H
