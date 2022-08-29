#ifndef NEURON_H
#define NEURON_H

#include "connection.h"
#include <iostream>
#include <vector>

class Neuron {
    private:
        double bias;
        double cached_value;
        std::vector<Connection> inputs;
        bool has_to_be_updated;
        bool frozen;
    public:
        Neuron();
        double activation();
        void set_bias(double bias);
        void set_weight(int index, double weight);
        void create_connections(std::vector<Neuron *> upstream_neurons);
        void freeze();
        void unfreeze();
        void fix_value(double value);
};

#endif