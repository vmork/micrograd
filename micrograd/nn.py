from abc import ABC, abstractmethod
import numpy as np

import micrograd.ops as ops 
from micrograd.core import TensorData, Tensor, Op

# TODO: look into other weight inits
def kaiming_init(k: int, shape) -> Tensor:
    sk = np.sqrt(1/k)
    return Tensor(np.random.uniform(-sk, sk, shape))

# -- Layers --

class Module(ABC):
    def __init__(self):
        pass 

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    def parameters(self) -> list[Tensor]:
        return []

    def __call__(self, *inputs: Tensor) -> Tensor:
        return self.forward(*inputs)
    
    def apply(self, fn):
        fn(self)

class Linear(Module):
    def __init__(self, insize: int, outsize: int):
        self.insize = insize 
        self.outsize = outsize

        self.weight = kaiming_init(insize, (outsize, insize))
        self.bias   = kaiming_init(insize, (1, outsize))
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.data.ndim == 2 
        xb, xn = x.shape 
        assert xn == self.insize

        y = x.batchmm(self.weight) + self.bias
        
        assert y.shape == (xb, self.outsize)
        return y

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]

class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = layers 
    
    def forward(self, x: Tensor) -> Tensor:
        for m in self.layers:
            x = m(x)
        return x

    def parameters(self) -> list[Tensor]:
        return [p for m in self.layers for p in m.parameters()]
    
    def apply(self, fn):
        for m in self.layers:
            m.apply(fn)

class Conv2D(Module):
    def __init__(self, cin: int, cout: int, kersize: tuple[int,int], padding: tuple[int,int]=(0,0)):
        kh, kw = kersize
        self.w = kaiming_init(cin*kh*kw, (cout, cin, kh, kw))
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.data.ndim == 4, f"input has shape {x.shape}"
        return ops.Conv2D.apply(x, self.w, padding=self.padding)

class Flatten(Module):
    def __init__(self, start_dim: int=1, end_dim: int=-1):
        self.start = start_dim
        self.end = end_dim 
    def forward(self, x: Tensor) -> Tensor:
        return x.view((*x.shape[:self.start], -1, *x.shape[self.end:][1:]))

# -- Activation functions --
                
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.ReLU.apply(x)

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.Sigmoid.apply(x)

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.Tanh.apply(x)

class LogSoftmax(Module):
    def __init__(self, axis: int=-1):
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return ops.LogSoftmax.apply(x, axis=self.axis)

# -- Loss functions --

class NLLLoss(Module):
    """Negative log-likelihood loss. Input assumed to be logsoftmaxed."""
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.data.ndim == 2
        assert y.data.squeeze().ndim == 1
        y = y.data.astype(int).squeeze()
        assert x.shape[0] == y.size

        return -x[range(x.shape[0]), y].mean()

class CrossEntropyLoss(Module):
    """Cross entropy loss. Input assumed to be unnormalized logits."""
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return NLLLoss()(LogSoftmax()(x), y)

class MSELoss(Module):
    """Mean squared error loss."""
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return ((x - y)**2).mean()