#ifndef MLP_CC
#define MLP_CC

#include <vector>
#include "value.cc"
#include "parameter.cc"
#include "layer.cc"

class MLP : BaseParameterClass
{
private:
public:
    std::vector<Layer> layers;
    MLP(std::vector<size_t> layer_sizes)
    {
        layers = std::vector<LayerV2>();
        layers.reserve(layer_sizes.size());

        for (auto index = 0; index < layer_sizes.size(); index++)
        {
            // inputs are previous layers number of nodes
            layers.push_back(LayerV2(layer_sizes[index], layer_sizes[std::max(index - 1, 0)]));
        }
    }

    std::vector<ValuePtr> run(std::vector<double> inputs)
    {
        auto values = std::vector<ValuePtr>();
        values.reserve(inputs.size());
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(values), [](auto double_value)
                       { return ValuePtr(new Value(double_value)); });

        for (auto layer : layers)
        {
            values = layer.run(values);
        }
        return values;
    }

    void zero_grad() override
    {
        for (auto layer : layers)
        {
            layer.zero_grad();
        }
    }

    void update_params(double alpha) override
    {
        for (auto layer : this->layers)
        {
            layer.update_params(alpha);
        }
    }
};

#endif