from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxZ = Z.max(axis=self.axes, keepdims=True)
        expSum = array_api.sum(array_api.exp(Z - maxZ), axis=self.axes)
        return array_api.log(expSum) + Z.max(axis=self.axes, keepdims=False)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        input_shape = z.shape
        tmp_reshape = list(input_shape)

        if self.axes is None:
            return out_grad * exp(z - node)
        elif isinstance(self.axes, tuple):
            for summed_axe in self.axes:
                tmp_reshape[summed_axe] = 1
        elif isinstance(self.axes, int):
            tmp_reshape[self.axes] = 1
        
        node_new = reshape(node, tuple(tmp_reshape))
        grad_new = reshape(out_grad, tuple(tmp_reshape))

        return grad_new * exp(z - node_new)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

