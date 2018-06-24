//
// Created by Qut on 17/06/2018.
//

#include "Neuron.h"



double Neuron::getBias() const {
    return bias;
}

void Neuron::setBias(double bias) {
    Neuron::bias = bias;
}

double Neuron::getEntries(int i) {
    return entries[i];
}

void Neuron::setEntries(double *entries) {
    Neuron::entries = entries;
}

double Neuron::getWeigth(int i) {
    return weigth[i];
}

void Neuron::setWeigth(double *weigth) {
    Neuron::weigth = weigth;
}

double Neuron::getOutput() const {
    return output;
}

Neuron::Neuron() = default;
