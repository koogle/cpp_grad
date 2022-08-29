#include <iostream>
#include <vector>
#include "printing.cc"
#include "data_loading.cc"
#include "neuron.cc"
#include "layer.cc"

int main(int _argc, char **_argv)
{
    std::cout << "Booting up..." << std::endl;
    // hack as I have to refactor the count of files loading
    int total_count = 6000;

    uint8_t *training_labels = read_training_labels();
    uint8_t **training_images = read_training_images();
    print_random_image(total_count, training_labels, training_images);

    // Create input layer of the network
    auto first_layer = Layer(28 * 28, relu);
    auto second_layer = Layer(15, relu);
    auto third_layer = Layer(15, relu);
    auto output_layer = Layer(12, relu);

    second_layer.fully_connect(first_layer);
    third_layer.fully_connect(second_layer);
    output_layer.fully_connect(third_layer);

    uint8_t *test_values = training_images[20];
    std::vector<double> values;
    for (int index = 0; index < 28 * 28; index++)
    {
        std::cout << "fixing " << (double)test_values[index] << std::endl;
        values.push_back(test_values[index]);
    }
    first_layer.set_values(values);

    std::vector<double> activation = output_layer.activate_all_neurons();

    for (int index = 0; index < activation.size(); index++)
    {
        std::cout << index << ": " << activation.at(index) << std::endl;
    }

    free(training_labels);
    for (int index = 0; index < total_count; index++)
    {
        free(training_images[index]);
    }
    free(training_images);
    return 0;
}