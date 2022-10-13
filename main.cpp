#include <iostream>
#include <vector>
#include <cmath>
#include <set>
#include <array>
#include <algorithm>
#include <ctime>
#include "mlp.cc"


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

        // calculate loss and update weights
        auto loss = ValuePtr(new Value(0.0));
        for (auto index = 0; index < input.size(); index++)
        {
            auto single_loss = v[index]->add(ValuePtr(new Value(input[index]))->mul(-1.0));
            loss = loss->add(single_loss->mul(single_loss));
        }
        
        mlp.zero_grad();
        loss->backward();
        
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
    
    return 0;
}
