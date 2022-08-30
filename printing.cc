#include <string>
#include <iostream>

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
