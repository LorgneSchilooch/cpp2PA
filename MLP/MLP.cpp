//
// Created by Qut on 17/06/2018.
//

#include "MLP.h"

void dllTrainMLPModel(Neuron neuronArrayByLayer, std::vector<double> pt[], double *layer, int layerLen, double *Result, int len, double *param) {


}


void dllSpiderMan(Neuron **neuronArrayByLayer, double *layer, int layerLen) {
    double **blop = new double *[3];
    blop[0] = &neuronArrayByLayer[1]->output;
    neuronArrayByLayer[2]->setEntries(*blop);


}
