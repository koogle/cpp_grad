#include "neuron.h"
#include "connection.h"
#include <algorithm>
#include <iostream>
#include "random.cc"

Neuron::Neuron()
{
    this->has_to_be_updated = true;
}

double Neuron::activation(bool perform_relu = true)
{
    if (!this->has_to_be_updated)
    {
        return this->cached_value;
    }

    // Just use relu here
    double sum = 0;
    for (const auto &connection : this->inputs)
    {
        sum += connection.source->activation() * connection.weight;
    }
    double activation = perform_relu ? std::max(sum - this->bias, 0.0) : sum - this->bias;
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
    int index = 0;
    for (auto upstream : upstream_neurons)
    {
        Connection new_connection = Connection();
        new_connection.source = upstream;
        // Init weights as random
        new_connection.weight = get_next_random_weight();
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

void Neuron::back_prop_error_rate()
{
    for (auto &connection : this->inputs)
    {
        // Relu derivate
        connection.source->add_error_rate(this->activation() > 0 ? this->error_rate * connection.weight : 0.0);
    }
}

void Neuron::init_error_rate(double value = 0.0)
{
    if (this->frozen)
    {
        return;
    }

    this->error_rate = value;
}

void Neuron::add_error_rate(double error_rate)
{
    if (this->frozen)
    {
        return;
    }

    this->error_rate += error_rate;
}

void Neuron::update_weights()
{
    if (this->frozen)
    {
        return;
    }

    double const learning_rate = 0.1;
    for (auto &connection : this->inputs)
    {
        connection.weight = connection.weight - (this->error_rate * learning_rate * connection.source->activation());
    }
    this->bias = this->bias - (this->error_rate * learning_rate);
    this->has_to_be_updated = true;
}