#include "neuron.h"
#include "connection.h"
#include <algorithm>

Neuron::Neuron()
{
    has_to_be_updated = true;
}

double Neuron::activation()
{
    if (!this->has_to_be_updated)
    {
        return this->cached_value;
    }

    // Just use relu here
    double sum = 0;
    for (const auto &connection: this->inputs)
    {
        sum += connection.source->activation() * connection.weight;
    }
    double activation = std::max(sum - this->bias, 0.0);
    this->cached_value = activation;
    this->has_to_be_updated = false;

    return activation;
}

void Neuron::set_bias(double bias)
{
    this->bias = bias;
}

void Neuron::set_weight(int index, double weight)
{
    this->inputs.at(index).weight = weight;
}

void Neuron::create_connections(std::vector<Neuron *> upstream_neurons)
{
    this->inputs.clear();
    for (auto upstream: upstream_neurons)
    {
        Connection new_connection = Connection();
        new_connection.source = upstream;
        new_connection.weight = 1.0;
        this->inputs.push_back(new_connection);
    }
}

void Neuron::freeze()
{
    this->frozen = true;
}

void Neuron::unfreeze()
{
    this->frozen = false;
}

void Neuron::fix_value(double value)
{
    this->freeze();
    this->cached_value = value;
    this->has_to_be_updated = false;
}
