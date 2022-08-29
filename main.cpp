#include <iostream>
#include <vector>
#include "printing.cc"
#include "data_loading.cc"
#include "neuron.cc"

std::vector<Neuron *> create_layer(int size)
{
    std::vector<Neuron *> layer;
    for (int index = 0; index < size; index++)
    {
        Neuron *n = new Neuron();
        n->set_bias(1.0);
        layer.push_back(n);
    }
    return layer;
}

void fully_connect(std::vector<Neuron *> left, std::vector<Neuron *> right)
{
    for (auto neuron: right)
    {
        neuron->create_connections(left);
    }
}

int main(int _argc, char **_argv)
{
    std::cout << "Booting up..." << std::endl;
    // hack as I have to refactor the count of files loading
    int total_count = 6000;

    uint8_t *training_labels = read_training_labels();
    uint8_t **training_images = read_training_images();
    print_random_image(total_count, training_labels, training_images);

    // Create input layer of the network
    auto first_layer = create_layer(28 * 28);
    auto second_layer = create_layer(15);
    auto third_layer = create_layer(15);
    auto output_layer = create_layer(10);

    fully_connect(first_layer, second_layer);
    fully_connect(second_layer, third_layer);
    fully_connect(third_layer, output_layer);

    uint8_t *test_values = training_images[20];
    for (int index = 0; index < 28 * 28; index++)
    {
        std::cout << "fixing " << (double) test_values[index] << std::endl;
        first_layer.at(index)->fix_value(test_values[index]);
    }

    for (int index = 0; index < 10; index++)
    {
        std::cout << index << ": " << output_layer.at(index)->activation() << std::endl;
    }

    free(training_labels);
    for (int index = 0; index < total_count; index++)
    {
        free(training_images[index]);
    }
    free(training_images);
    return 0;
}