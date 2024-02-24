# type: ignore

import numpy as np
import torch 

from micrograd.core import Tensor, OpNode, TensorData
from micrograd.nn import Module, Linear, Sequential, ReLU, LogSoftmax, NLLLoss, CrossEntropyLoss, Conv2D

def test_linear():
    x = Tensor(np.ones((64, 784)))
    m = Linear(784, 10)
    w, b = m.weight, m.bias 
    assert w.shape == (10, 784)
    assert b.shape == (1, 10)
    y = m(x)
    assert y.shape == (64, 10)
    assert np.allclose(y.data, ((w @ x.T).T + b).data, atol=1e-4)
    assert np.allclose(y.data, (x @ w.T + b).data, atol=1e-4)

    y.sum().backward()

def init_weights(m):
        if isinstance(m, Linear):
            m.weight.data = np.ones_like(m.weight.data)
            m.bias.data = np.ones_like(m.bias.data)

def init_weights_torch(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data = torch.ones_like(m.weight.data)
        m.bias.data = torch.ones_like(m.bias.data)

def test_sequential():
    x = Tensor(np.ones((64, 784)))
    m = Sequential(Linear(784, 10), 
                   ReLU(), 
                   Linear(10, 5), 
                   LogSoftmax(axis=-1))
    m.apply(init_weights)
    y = m(x)
    y.sum().backward()
    
    xt = torch.tensor(np.ones((64, 784)), dtype=torch.float32, requires_grad=True)
    mt = torch.nn.Sequential(torch.nn.Linear(784, 10), 
                             torch.nn.ReLU(), 
                             torch.nn.Linear(10, 5), 
                             torch.nn.LogSoftmax(dim=-1))
    mt.apply(init_weights_torch)
    yt = mt(xt)
    yt.sum().backward()

    assert yt.shape == y.shape
    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(m.layers[0].weight.grad, mt[0].weight.grad.numpy(), atol=1e-4)

def test_nll_loss():
    x = Tensor(np.arange(10)).broadcast((5, 10))
    y = x.logsoftmax(axis=-1)
    yhat = Tensor(np.array([0,1,2,3,4]))
    loss = NLLLoss()(y, yhat)
    loss.backward()

    xt = torch.tensor(np.broadcast_to(np.arange(10), (5,10)), dtype=torch.float32, requires_grad=True)
    yt = torch.nn.LogSoftmax(dim=1)(xt)
    yhatt = torch.tensor([0,1,2,3,4])
    losst = torch.nn.NLLLoss()(yt, yhatt)
    losst.backward()

    assert np.allclose(loss.data, losst.data.numpy())
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-4)

def test_ce_loss():
    x = Tensor(np.arange(10)).broadcast((5, 10))
    yhat = Tensor(np.array([0,1,2,3,4]))
    loss = CrossEntropyLoss()(x, yhat)
    loss.backward()

    xt = torch.tensor(np.broadcast_to(np.arange(10), (5,10)), dtype=torch.float32, requires_grad=True)
    yhatt = torch.tensor([0,1,2,3,4])
    losst = torch.nn.CrossEntropyLoss()(xt, yhatt)
    losst.backward()

    assert np.allclose(loss.data, losst.data.numpy())
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-4)

# TODO: make parametric
def test_conv2d():
    c = 3           # in channels
    b = 10          # batches
    h, w = 28, 28   # height, width (for 1D set h=channels, w=length)
    ph, pw = 1,1   # padding       (for 1D set ph=0, pw=padding)
    cout = 3        # out channels
    kh, kw = 3, 3   # filter size   (for 1D set kh=channels)

    x = Tensor(np.random.randn(b, c, h, w))
    conv = Conv2D(c, cout, (kh, kw), (ph, pw))
    y = conv(x)
    y.sum().backward()

    xt = torch.tensor(x.data, requires_grad=True)
    convt = torch.nn.Conv2d(c, cout, (kh, kw), 1, (ph, pw), bias=False)
    convt.weight.data = torch.tensor(conv.w.data)
    yt = convt(xt)
    yt.sum().backward()

    assert np.allclose(y.data, yt.data.numpy(), atol=1e-5)
    assert np.allclose(x.grad.data, xt.grad.data.numpy(), atol=1e-4)
    assert np.allclose(conv.w.grad.data, convt.weight.grad.data.numpy(), atol=1e-4)

if __name__ == "__main__":
    test_nll_loss()