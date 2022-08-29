#ifndef CONNECTION_H
#define CONNECTION_H

#include "neuron.h"
class Neuron;

struct Connection {
    Neuron *source;
    double weight;
};

#endif