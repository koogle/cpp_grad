#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include "layer.h"

Layer::Layer(int size, LayerType type) 
{
    this->type = type;
    for (int index = 0; index < size; index++)
    {
        Neuron *n = new Neuron();
        n->set_bias(0.0);
        this->neurons.push_back(n);
    }
}

std::vector<Neuron *> Layer::get_neurons()
{
    return this->neurons;
}

void Layer::fully_connect(Layer previous)
{
    for (auto neuron: this->neurons)
    {
        neuron->create_connections(previous.get_neurons());
    }
}

std::vector<double> Layer::activate_all_neurons()
{
    std::vector<double> result;
    for (auto neuron: this->neurons)
    {
        result.push_back(neuron->activation(this->type == relu));
    }
    
    if (this->type == softmax) {
        double sum = 0.0;
        double max_value = 0.0;
        for (auto value:result) {
            max_value = std::max(value, 0.0);
        }

        for (auto value: result) {
            sum += exp(value/max_value);
        }

        for (int index = 0; index < result.size(); index++) {
            result[index] = exp(result.at(index)/max_value) / sum;
        }
    }

    return result;
}

void Layer::set_values(std::vector<double> values)
{
    for (int index = 0; index < values.size(); index++)
    {
        this->neurons.at(index)->fix_value(values.at(index));
    }
}
