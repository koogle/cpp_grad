#include <vector>
#include "layer.h"

Layer::Layer(int size, LayerType type)
{
    this->type = type;
    for (int index = 0; index < size; index++)
    {
        Neuron *n = new Neuron();
        n->set_bias(1.0);
        this->neurons.push_back(n);
    }
}

std::vector<Neuron *> Layer::get_neurons()
{
    return this->neurons;
}

void Layer::fully_connect(Layer previous)
{
    for (auto neuron : this->neurons)
    {
        neuron->create_connections(previous.get_neurons());
    }
}

std::vector<double> Layer::activate_all_neurons()
{
    std::vector<double> result;
    for (auto neuron : this->neurons)
    {
        result.push_back(neuron->activation());
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