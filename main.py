import sys
sys.path.append('./python')
import itertools
import numpy as np
import needle as ndl
from needle import backend_ndarray as nd

device = nd.cuda()
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]

np.random.seed(1)    

if __name__ == '__main__':
    # test_tanh_backward()
    # cifar10_dataset()
    # cifar10_dataloader()
    out_channels = 3,
    in_channels = 2,
    bias_bound = 1
    dtype="float"
    a = ndl.init.rand(
                    *(out_channels,),
                    low=-bias_bound, high=bias_bound,
                    dtype=dtype,
                    device=device,
                    requires_grad=True)