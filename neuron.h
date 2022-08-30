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
        double error_rate;
    public:
        Neuron();
        double activation(bool perform_relu);
        void set_bias(double bias);
        void set_weight(int index, double weight);
        void create_connections(std::vector<Neuron *> upstream_neurons);
        void freeze();
        void unfreeze();
        void fix_value(double value);
        void back_prop_error_rate(); 
        void add_error_rate(double error_rate);
        void init_error_rate(double value);
        void update_weights();
};

#endif