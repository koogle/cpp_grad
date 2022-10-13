#ifndef LAYER_CC
#define LAYER_CC

#include <vector>
#include "value.cc"
#include "neuron.cc"
#include "parameter.cc"

class Layer : BaseParameterClass
{
private:
public:
    std::vector<Neuron> neurons;
    Layer(size_t n_neurons, size_t n_inputs)
    {
        neurons = std::vector<Neuron>();
        std::generate_n(std::back_inserter(neurons), n_neurons, [n_inputs]
                        { return Neuron(n_inputs); });
    }

    std::vector<ValuePtr> run(std::vector<ValuePtr> inputs)
    {
        auto results = std::vector<ValuePtr>();
        results.reserve(this->neurons.size());
        for (auto neuron : this->neurons)
        {
            results.push_back(neuron.run(inputs));
        }
        return results;
    }

    void zero_grad() override
    {
        for (auto neuron : this->neurons)
        {
            neuron.zero_grad();
        }
    }

    void update_params(double alpha) override
    {
        for (auto neuron : this->neurons)
        {
            neuron.update_params(alpha);
        }
    }
};

#endif