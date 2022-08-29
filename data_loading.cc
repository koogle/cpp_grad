#include <fstream>
#include <iostream>

unsigned int read_int32_t(std::ifstream *input)
{
    void *buffer = malloc(4);
    input->read((char *)buffer, 4);
    unsigned int value = __builtin_bswap32(*((int32_t *)buffer));
    free(buffer);
    return value;
}

uint8_t **read_training_images()
{
    std::ifstream input("data/train-images-idx3-ubyte", std::ios::binary);
    unsigned int magic_number = read_int32_t(&input);
    unsigned int count = read_int32_t(&input);
    unsigned int rows = read_int32_t(&input);
    unsigned int columns = read_int32_t(&input);
    assert(magic_number == 2051);

    uint8_t **image_data = (uint8_t **)malloc(count * sizeof(uint8_t *));
    for (int index = 0; index < count; index++)
    {
        image_data[index] = (uint8_t *)malloc(rows * columns * sizeof(u_int8_t));
        input.read((char *)image_data[index], rows * columns);
    }

    // read file end sign
    char *eof = (char *)malloc(1);
    input.read(eof, 1);
    assert(input.eof());

    std::cout << "done reading test image data " << *eof << std::endl;
    return image_data;
}

uint8_t *read_training_labels()
{
    std::ifstream input("data/train-labels-idx1-ubyte", std::ios::binary);
    unsigned int magic_number = read_int32_t(&input);
    unsigned int count = read_int32_t(&input);
    assert(magic_number == 2049);

    uint8_t *label_data = (uint8_t *)malloc(count * sizeof(uint8_t));
    input.read((char *)label_data, count);

    // read file end sign
    char *eof = (char *)malloc(1);
    input.read(eof, 1);
    assert(input.eof());

    std::cout << "done reading test label data " << *eof << std::endl;
    return label_data;
}
