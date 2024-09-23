# type: ignore

import numpy as np
import torch
import pytest

from micrograd.core import Tensor, OpNode, TensorData, ShapeError
import micrograd.nn as nn

def test_ops():
    a = Tensor(np.array([1,1,1]), name="a")
    b = Tensor(np.array([1,1,1]), name="b")
    y = (a + b - a + a).log().exp().sum()
    assert y.data == 6
    assert a.sum().data == 3 
    assert np.allclose(a.log().data, np.log(a.data))
    assert np.allclose(a.exp().data, np.exp(a.data))
    y = -(a * b) * 2
    assert np.allclose(y.data, -2*np.ones_like(a.data))

    x = Tensor(np.array([1, -1, 0, -3.4]))
    y = x.abs()
    y.sum().backward()
    assert np.allclose(y.data, np.abs(x.data))
    assert np.allclose(x.grad.data, np.sign(x.data))

    y = x / 2.0
    assert np.allclose(y.data, x.data / 2.0)

def test_sum():
    x = Tensor(np.ones((2, 3, 4)))
    y = x.sum(axis=1)
    assert y.data.shape == (2, 4)
    assert np.allclose(y.data, 3)
    y = x.sum(axis=(0, 2))
    assert y.data.shape == (3,)
    assert np.allclose(y.data, 8)

    y = x.sum()
    assert y.data.shape == ()
    assert np.allclose(y.data, 24)
    y.backward()
    assert np.allclose(x.grad, np.ones_like(x.data))

def test_mean():
    x = Tensor(np.ones((2, 3, 4)))
    y = x.mean(axis=1)
    assert y.data.shape == (2, 4)
    print(y.data)
    assert np.allclose(y.data, 1)
    y = x.mean(axis=(0, 2))
    assert y.data.shape == (3,)
    assert np.allclose(y.data, 1)

    y = x.mean()
    assert y.data.shape == ()
    assert np.allclose(y.data, 1)
    y.backward()
    assert np.allclose(x.grad, np.ones_like(x.data) * 1/24)

def test_matmul():
    x = Tensor(np.ones((2, 3)))
    y = Tensor(np.ones((3, 4)))
    z = x @ y
    assert z.data.shape == (2, 4)
    assert np.allclose(z.data, 3)
    z.sum().backward()
    assert np.allclose(x.grad, z.grad @ y.data.T)
    assert np.allclose(y.grad, x.data.T @ z.grad)

def test_batchmm():
    x = Tensor(np.ones((10, 3)))
    y = Tensor(np.ones((5, 3)))
    z = x.batchmm(y)
    z.sum().backward()

    xt = torch.tensor(np.ones((10, 3)), requires_grad=True)
    yt = torch.tensor(np.ones((5, 3)), requires_grad=True)
    zt = xt @ yt.T
    zt.sum().backward()

    assert z.data.shape == zt.data.shape == (10, 5)
    assert np.allclose(z.data, zt.data)
    assert np.allclose(x.grad, xt.grad.numpy())

def test_dotprod():
    x = Tensor(np.ones(10))
    y = Tensor(np.ones(10)*3)
    z = x @ y 
    z.sum().backward()
    assert np.allclose(x.grad, y.data)
    assert np.allclose(y.grad, x.data)

def test_matmul_fail():
    with pytest.raises(ShapeError):
        z = Tensor(np.ones((2, 3))) @ Tensor(np.ones((2, 3)))
    with pytest.raises(ShapeError):
        z = Tensor(np.array(1)) @ Tensor(np.array(1))
    with pytest.raises(ShapeError):
        z = Tensor(np.array(1)) @ Tensor(np.array([1,2,3]))
    with pytest.raises(ShapeError):
        z = Tensor(np.array([1])) @ Tensor(np.array([[1,2,3]]))

def test_broadcast1():
    x = Tensor(np.ones((3,1)))
    y = x.broadcast((5, 10, 3, 7))
    assert y.data.shape == (5, 10, 3, 7)
    y.sum().backward()
    assert x.grad.shape == x.shape
    assert np.allclose(x.grad, np.ones_like(x.data) * 5*10*7)

def test_select():
    x = Tensor(np.arange(12).reshape(3,4))
    y = x[:, 1]
    assert y.data.shape == (3,)
    assert np.allclose(y.data, x.data[:, 1])
    y.sum().backward()
    g = np.zeros_like(x.data); g[:, 1] = 1
    assert np.allclose(x.grad, g)

    x = Tensor(np.ones((3,3,5,10)))
    y = x[1,2,1,3]
    assert y.data.shape == ()
    assert np.allclose(y.data, x.data[1,2,1,3])
    y.backward()
    print(x.grad)
    g = np.zeros_like(x.data); g[1, 2, 1, 3] = 1
    assert np.allclose(x.grad, g)

def test_softmax():
    xt = torch.tensor([[1,2,3], [1,2,3]], dtype=torch.float32, requires_grad=True)
    ft = torch.nn.LogSoftmax(dim=0)
    yt = ft(xt)
    yt.sum().backward()
    print(xt.grad)

    x = Tensor(np.array([[1,2,3], [1,2,3]]))
    f = nn.LogSoftmax(axis=0)
    y = f(x)
    y.sum().backward()
    print(x.grad)

    assert np.allclose(y.data, yt.data.numpy())
    assert np.allclose(x.grad, xt.grad.numpy())

if __name__ == "__main__":
    test_mean()