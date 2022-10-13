# CPP Grad

A simple MLP written in CPP with shared pointers and implements ADD, MUL and RELU operations in under 500 lines.

Inspiration has been
- https://github.com/geohot/tinygrad
- https://github.com/karpathy/micrograd


```cpp
const std::vector<std::vector<double>> inputs = {
    {1.0, -2.0, -1.0, 0.0},
    {.75, -1.75, -1.0, .25},
    {.5, -1.5, -1.0, .5},
    {.25, -1.25, -1.0, .75},
    {0.0, -1.0, -1.0, 1.0},
};

auto mlp = MLP({4, 5, 2, 6, 4});
mlp.zero_grad();

const auto total_runs = 100;

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
        std::cout << "final loss: " << loss->value << std::endl;
    }
}
```
