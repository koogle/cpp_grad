#include <string>
#include <iostream>
#include <random>

std::string convert_to_char(int value)
{
    if (value > 191)
    {
        return "F";
    }
    else if (value > 127)
    {
        return "A";
    }
    else if (value > 64)
    {
        return "-";
    }
    else
    {
        return " ";
    }
}

void print_image(uint8_t *training_labels, uint8_t **training_images, int index)
{
    for (int row_index = 0; row_index < 28; row_index++)
    {
        std::string line;
        for (int column_index = 0; column_index < 28; column_index++)
        {
            line.append(convert_to_char((int)training_images[index][row_index * 28 + column_index]));
        }
        std::cout << line << std::endl;
    }
    std::cout << "Label: " << (int)training_labels[index] << std::endl;
}

void print_random_image(int total_count, uint8_t *training_labels, uint8_t **training_images)
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int> distr(0, total_count);

    print_image(training_labels, training_images, distr(generator));
}