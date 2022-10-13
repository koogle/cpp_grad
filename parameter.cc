#ifndef PARAMETER_CC
#define PARAMETER_CC

class BaseParameterClass
{
private:
public:
    virtual void zero_grad(){};
    virtual void update_params(double alpha){};
};

#endif