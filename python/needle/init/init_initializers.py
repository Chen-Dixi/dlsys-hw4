import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = math.sqrt(2) * math.sqrt(3 / fan_in) # gain is sqrt(2)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs) if shape is None else rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION