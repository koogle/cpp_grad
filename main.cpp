#include <iostream>
#include <vector>
#include <cmath>
#include <set>
#include <array>
#include <algorithm>
#include <ctime>
#include "printing.cc"
#include "data_loading.cc"
#include "neuron.cc"
#include "layer.cc"
#include "random.cc"

/*std::vector<double> calculate_error(int label, std::vector<double> output)
{
    std::vector<double> error_vector;
    for (int index = 0; index < output.size(); index++)
    {
        error_vector.push_back(index == label ? output[index] - 1 : output[index]);
    }
    return error_vector;
}*/

enum ValueOperation
{
    NONE,
    ADD,
    MUL,
    RELU
};

class Value;

using ValuePtr = std::shared_ptr<Value>;

class Value : public std::enable_shared_from_this<Value>
{
private:
    ValueOperation operation = NONE;
    std::array<ValuePtr, 2> previous = {nullptr, nullptr};

public:
    double grad = 0.0;
    double value;
    bool needs_grad = false;

    Value(double initValue)
    {
        this->value = initValue;
    }

    Value(double initValue, ValueOperation operation, ValuePtr first, ValuePtr second)
    {
        this->value = initValue;
        this->operation = operation;
        this->previous = {first, second};
        this->needs_grad = true;
    }

    ValuePtr mul(double other)
    {
        return this->mul(ValuePtr(new Value{other}));
    }

    ValuePtr mul(ValuePtr other)
    {
        return ValuePtr(new Value(this->value * other->value, MUL, shared_from_this(), other));
    }

    ValuePtr add(double other)
    {
        return this->add(ValuePtr(new Value{other}));
    }

    ValuePtr add(ValuePtr other)
    {
        return ValuePtr(new Value(this->value + other->value, ADD, shared_from_this(), other));
    }

    ValuePtr relu()
    {
        return ValuePtr(new Value(this->value < 0 ? 0 : this->value, RELU, shared_from_this(), nullptr));
    }

    void backward_step()
    {
        const auto first = this->previous[0];
        const auto second = this->previous[1];
        switch (this->operation)
        {
        case ADD:
            if (first->needs_grad)
            {
                first->grad += this->grad;
            }
            if (second->needs_grad)
            {
                second->grad += this->grad;
            }
            break;
        case MUL:
            if (first->needs_grad)
            {
                first->grad += second->value * this->grad;
            }
            if (second->needs_grad)
            {
                second->grad += first->value * this->grad;
            }
            break;
        case RELU:
            if (second != nullptr)
            {
                std::cerr << "relu backward called with two parent nodes" << std::endl;
            }
            if (first->needs_grad)
            {
                first->grad += (this->value > 0) * this->grad;
            }

            break;
        case NONE:
            break;
        }
    }

    void search_previous(std::vector<ValuePtr> &order, std::set<ValuePtr> &visited)
    {
        for (const auto previous : this->previous)
        {
            if (previous == nullptr)
            {
                continue;
            }
            if (visited.find(previous) == visited.end())
            {
                visited.insert(previous);
                previous->search_previous(order, visited);
                order.push_back(previous);
            }
        }
    }

    void backward()
    {
        auto order = std::vector<ValuePtr>();
        auto visited = std::set<ValuePtr>();
        this->grad = 1.0;
        this->search_previous(order, visited);
        std::reverse(order.begin(), order.end());
        this->backward_step();
        for (const auto node : order)
        {
            node->backward_step();
        }
    }
};

class BaseParameterClass
{
private:
public:
    virtual void zero_grad(){};
    virtual void update_params(double alpha){};
};

class NeuronV2 : BaseParameterClass
{
private:
    bool linear = true;

public:
    std::vector<ValuePtr> params;

    NeuronV2(size_t n_inputs)
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

class LayerV2 : BaseParameterClass
{
private:
public:
    std::vector<NeuronV2> neurons;
    LayerV2(size_t n_neurons, size_t n_inputs)
    {
        neurons = std::vector<NeuronV2>();
        std::generate_n(std::back_inserter(neurons), n_neurons, [n_inputs]
                        { return NeuronV2(n_inputs); });
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

class MLP : BaseParameterClass
{
private:
public:
    std::vector<LayerV2> layers;
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

int main(int _argc, char **_argv)
{
    std::cout << "Booting up..." << std::endl;
    const std::vector<std::vector<double>> inputs = {
        {1.0, -2.0, -1.0, 0.0},
        {.75, -1.75, -1.0, .25},
        {.5, -1.5, -1.0, .5},
        {.25, -1.25, -1.0, .75},
        {0.0, -1.0, -1.0, 1.0},
    };

    auto mlp = MLP({4, 5, 2, 6, 4});
    mlp.zero_grad();

    const auto total_runs = 200; // 50000;
    const auto start_time = std::time(NULL);
    std::cout << start_time << std::endl;
    for (auto i = 0; i <= total_runs; i++)
    {
        auto input = inputs[i%5];
        auto v = mlp.run(input);

        /*const auto neurons = mlp.layers[0].neurons;
        for (auto index = 0; index < neurons.size(); index++)
        {
            //            std::cout << "\nNeuron " << index + 1 << ":\n";
            for (auto weight : neurons[index].params)
            {
                //                std::cout << "v: " << weight->value << "\tgrad:" << weight->grad << std::endl;
            }
        }*/

        // calculate loss and update weights
        auto loss = ValuePtr(new Value(0.0));
        for (auto index = 0; index < input.size(); index++)
        {
            auto single_loss = v[index]->add(ValuePtr(new Value(input[index]))->mul(-1.0));
            loss = loss->add(single_loss->mul(single_loss));
        }
        
        mlp.zero_grad();
        loss->backward();
        
        //for (auto neuron: (--mlp.layers.end())->neurons) {
        //    std::cout << "neuron:" << std::endl;
        //
        //    for (auto param : neuron.params) {
        //        std::cout << "p: " << param->grad << std::endl;
        //    }
        //}

        mlp.update_params(0.01);

        if (i == total_runs)
        {
            
            const auto end_time = std::time(NULL);
            std::cout << "input" << std::endl;
            for (auto in: input) {
                std::cout << in << std::endl;
            }

            std::cout << "output" << std::endl;
            for (auto output : v)
            {
                std::cout << output->value << "\t" << output->grad << std::endl;
            }
            std::cout << "final loss: " << loss->value << "\ttotal time: " << end_time - start_time << std::endl;
        }
    }
    

    // v->backward();
    // for (auto param : n.params)
    //{
    //     std::cout << param->value << "\t" << param->grad << std::endl;
    // }

    /*auto x = ValuePtr(new Value{-4.0});
    x->needs_grad = true;
    auto z = x->mul(2)->add(2)->add(x);
    auto q = z->relu()->add(z->mul(x));
    auto h = z->mul(z)->relu();
    auto y = h->add(q)->add(q->mul(x));
    y->backward();
    std::cout << "y.data " << y->value << "\n" << "x.grad" << x->grad << std::endl;
    */

    return 0;
    /*
    // hack as I have to refactor the count of files loading
    const int total_count = 6000;
    std::uniform_int_distribution<int> image_distribution(0, total_count);

    const int image_index = image_distribution(generator);

    uint8_t *training_labels = read_training_labels();
    uint8_t **training_images = read_training_images();
    print_image(training_labels, training_images, image_index);

    // Create input layer of the network
    auto first_layer = Layer(28 * 28, relu);
    auto second_layer = Layer(15, relu);
    auto third_layer = Layer(15, relu);
    auto output_layer = Layer(10, softmax);

    second_layer.fully_connect(first_layer);
    third_layer.fully_connect(second_layer);
    output_layer.fully_connect(third_layer);

    uint8_t *test_values = training_images[image_index];
    std::vector<double> values;
    for (int index = 0; index < 28 * 28; index++)
    {
        // Scale values to 255
        values.push_back(test_values[index] / 255);
    }
    first_layer.set_values(values);

    uint8_t test_label = training_labels[image_index];
    for (int i = 0; i < 10; i++)
    {

        std::vector<double> output = output_layer.activate_all_neurons();
        std::cout << "round: " << i << std::endl;
        for (int index = 0; index < output.size(); index++)
        {
            std::cout << index << ": " << output.at(index) << std::endl;
        }
        std::vector<Neuron *> output_neurons = output_layer.get_neurons();
        std::vector<double> errors = calculate_error(test_label, output);
        for (int index = 0; index < output_neurons.size(); index++)
        {
            output_neurons[index]->init_error_rate(errors[index]);
        }
        for (auto neuron : third_layer.get_neurons())
        {
            neuron->init_error_rate();
        }
        for (auto neuron : second_layer.get_neurons())
        {
            neuron->init_error_rate();
        }
        for (auto neuron : output_layer.get_neurons())
        {
            neuron->back_prop_error_rate();
        }
        for (auto neuron : third_layer.get_neurons())
        {
            neuron->back_prop_error_rate();
        }
        for (auto neuron : second_layer.get_neurons())
        {
            neuron->update_weights();
        }
        for (auto neuron : third_layer.get_neurons())
        {
            neuron->update_weights();
        }
        for (auto neuron : output_layer.get_neurons())
        {
            neuron->update_weights();
        }
    }

    free(training_labels);
    for (int index = 0; index < total_count; index++)
    {
        free(training_images[index]);
    }
    free(training_images);
    return 0;
    */
}
