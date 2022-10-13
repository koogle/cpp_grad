#ifndef VALUE_CC
#define VALUE_CC

#include <memory>
#include <set>
#include <vector>
#include <array>
#include <iostream>

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

#endif