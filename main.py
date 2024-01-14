import sys
sys.path.append('./python')
import itertools
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

def cifar10_dataset():
    train = False
    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=train)
    if train:
        assert len(dataset) == 50000
    else:
        assert len(dataset) == 10000
    example = dataset[np.random.randint(len(dataset))]
    assert(isinstance(example, tuple))
    X, y = example
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 32, 32)

def cifar10_dataloader():
    train = False
    batch_size = 2
    cifar10_test_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=train)
    train_loader = ndl.data.DataLoader(cifar10_test_dataset, batch_size)
    for (X, y) in train_loader:
        break
    assert isinstance(X.cached_data, nd.NDArray)
    assert isinstance(X, ndl.Tensor)
    assert isinstance(y, ndl.Tensor)
    assert X.dtype == 'float32'
    

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