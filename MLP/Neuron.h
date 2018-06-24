//
// Created by Qut on 17/06/2018.
//

#ifndef CPP2PA_NEURON_H
#define CPP2PA_NEURON_H


class Neuron {

public:
    double *entries;

    Neuron();


    double getEntries(int i);

    void setEntries(double *entries);

    double getWeigth(int i);

    void setWeigth(double *weigth);

    double getBias() const;

    void setBias(double bias);

    double *weigth;
    double bias;
    double output;

    double getOutput() const;


};


#endif //CPP2PA_NEURON_H
