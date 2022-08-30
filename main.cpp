#include <iostream>
#include <vector>
#include <cmath>
#include "printing.cc"
#include "data_loading.cc"
#include "neuron.cc"
#include "layer.cc"
#include "random.cc"

std::vector<double> calculate_error(int label, std::vector<double> output)
{
    std::vector<double> error_vector;
    for (int index = 0; index < output.size(); index++)
    {
        error_vector.push_back(index == label ? output[index] - 1 : output[index]);
    }
    return error_vector;
}

int main(int _argc, char **_argv)
{
    std::cout << "Booting up..." << std::endl;

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
}
