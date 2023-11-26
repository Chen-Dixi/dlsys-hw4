import sys
sys.path.append('./python')
import numpy as np
import needle as ndl
from needle import backend_ndarray as nd

device = nd.cuda()
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]

np.random.seed(1)

def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    # print("backward grad:{}".format(backward_grad))
    # print("numerical grad:{}".format(numerical_grad))
    assert error < 4.2e-1, "error: {}".format(error)
    return [g.numpy() for g in backward_grad]

def test_tanh():
    for shape in GENERAL_SHAPES:
        _A = np.random.randn(*shape).astype(np.float32)
        A = ndl.Tensor(nd.array(_A), device=device)
        ndl_tanh_res = ndl.tanh(A).numpy()
        np.testing.assert_allclose(np.tanh(_A), ndl_tanh_res, atol=1e-5, rtol=1e-5)
        print("=======")
        print(ndl_tanh_res)

def test_tanh_backward():
    shape = (4, 5, 6)
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.tanh, A)

if __name__ == '__main__':
    test_tanh_backward()