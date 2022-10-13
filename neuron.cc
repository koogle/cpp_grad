#ifndef NEURON_CC
#define NEURON_CC

#include <vector>
#include "value.cc"
#include "parameter.cc"
#include "random.cc"

class Neuron : BaseParameterClass
{
private:
    bool linear = true;

public:
    std::vector<ValuePtr> params;

    Neuron(size_t n_inputs)
    {
        params = std::vector<ValuePtr>();
        std::generate_n(std::back_inserter(params), n_inputs + 1, []
                        { return ValuePtr(new Value{get_next_random_weight()}); });
        for (const auto param : params)
        {
            param->needs_grad = true;
        }
    }

    ValuePtr run(std::vector<ValuePtr> inputs)
    {
        // adding one term for the bias
        inputs.push_back(ValuePtr(new Value(1)));
        if (inputs.size() != params.size())
        {
            std::cerr << "can't process inputs that don't match number of weights" << std::endl;
        }
        auto act = ValuePtr(new Value(0.0));
        for (auto index = 0; index < inputs.size(); index++)
        {
            act = act->add(params[index]->mul(inputs[index]));
        }
        if (!linear)
        {
            return act->relu();
        }
        else
        {
            return act;
        }
    }

    void zero_grad() override
    {
        for (auto param : this->params)
        {
            param->grad = 0;
        }
    }

    void update_params(double alpha) override
    {
        for (auto param : this->params)
        {
            param->value -= param->grad * alpha;
        }
    }
};

#endif