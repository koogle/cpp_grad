#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "neuron.h"

enum LayerType {
    relu,
    softmax 
};

class Layer {
    private:
        int size;
        LayerType type; 
        std::vector<Neuron *> neurons;
    public:
        Layer(int size, LayerType type);
        void fully_connect(Layer layer);
        std::vector<double> activate_all_neurons();
        std::vector<Neuron *> get_neurons();
        void set_values(std::vector<double> values);
};

#endif