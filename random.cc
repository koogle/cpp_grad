#ifndef RANDOM_CC
#define RANDOM_CC

#include <random>
std::random_device rand_dev;
std::mt19937 generator(rand_dev());

std::uniform_real_distribution<double> weight_distribution(0, 1);

double get_next_random_weight()
{
    return weight_distribution(generator);
}

#endif