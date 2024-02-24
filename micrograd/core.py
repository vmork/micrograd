from __future__ import annotations
from typing import Callable, List, Any
from abc import ABC, abstractmethod
from functools import wraps
import numpy as np

TensorData = np.ndarray 
Shape = tuple[int] | int | None

def do_broadcast(*inputs: Tensor) -> tuple[Tensor,...]:
    """
    Broadcasts the inputs to the same shape if necessary, returning new nodes that can be backpropagated through correctly.
    """
    shapes = [x.shape for x in inputs]
    bshape = np.broadcast_shapes(*shapes)
    return tuple(x.broadcast(bshape) if x.shape != bshape else x for x in inputs)

def astuple(x: Any):
    return x if isinstance(x, tuple) else (x,)

class Op(ABC):
    """
    Bundles a function f (forward) together with its derivative (backward). 
    See ops.py for implementations.
    """

    broadcastable = False

    @staticmethod
    @abstractmethod
    def forward(ctx: dict, *inputs: TensorData, **kwargs) -> TensorData|tuple[TensorData,...]:
        """
        Computes the operation and stores information required for the backward pass in the ctx-dict.
        """
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx: dict, *outgrads: TensorData) -> TensorData|tuple[TensorData,...]:
        """
        Takes outgrads, the gradients of the scalar loss l wrt each of the outputs, ie dl/do for each o,
        and returns tuple of gradients wrt the inputs, ie dl/dx for each x.
        """
        pass

    @classmethod
    def apply(cls, *inputs: Tensor, **kwargs):
        """
        Apply the forward method and build up the graph.
        """
        ctx = {}

        if cls.broadcastable: inputs = do_broadcast(*inputs)

        raw_inputs = [t.data for t in inputs]
        raw_outputs = cls.forward(ctx, *raw_inputs, **kwargs)

        if not isinstance(raw_outputs, tuple): raw_outputs = (raw_outputs, )
        opnode = OpNode(cls, ctx, *inputs)
        outputs = tuple(Tensor(y, opnode) for y in raw_outputs)
        opnode.outputs = outputs 
        return outputs[0] if len(outputs) == 1 else outputs

class OpNode:
    """
    Operation node in the computation graph, multiple inputs and multiple outputs.
    Corresponds to Function in pytorch.
    """
    def __init__(self, op: type[Op], ctx: dict, *inputs: Tensor):
        self.op = op 
        self.ctx = ctx
        self.inputs = inputs 
        self.outputs: tuple[Tensor,...]|None = None 
        self.name = op.__name__
    
    def __repr__(self):
        return f"{self.name}({''.join(str(x) for x in self.inputs)})"
    
    def _backward(self): 
        """Update the grads for the input tensors to this opnode"""
        gys = tuple(y.grad for y in self.outputs) # type: ignore
        gxs = self.op.backward(self.ctx, *gys)
        gxs = (gxs,) if type(gxs) != tuple else gxs
        for x, gx in zip(self.inputs, gxs):
            assert x.grad.shape == gx.shape, f"{self.name}._backward: {x.grad.shape} != {gx.shape}" #type: ignore
            x.grad += gx 
    
import micrograd.ops as ops # moved bc circular import issue

def wrap_numeric(f: Callable) -> Callable:
    @wraps(f)
    def _wrapper(self, other):
        other = Tensor(other) if isinstance(other, (int, float)) else other
        return f(self, other)
    return _wrapper

class Tensor:
    def __init__(self, data: TensorData|int|float, source: OpNode|None=None, name:str|None=None):
        self.data: TensorData = data if isinstance(data, np.ndarray) else np.array(data)
        # self.data = self.data.astype(np.float32)
        self.grad = np.zeros_like(data, dtype=np.float32)
        self.source = source
        self.name = name or "."
    
    def _topo_order(self) -> reversed[OpNode]:
        t = []
        visited = set()
        def _dfs(n: Any):
            children = n.inputs if type(n) == OpNode else (n.source,)
            for c in children:
                if c in visited or c is None: continue 
                visited.add(c)
                _dfs(c)
            if type(n) == OpNode: t.append(n)
        _dfs(self)
        return reversed(t)
    
    def backward(self):
        if self.data.squeeze().ndim != 0:
            raise ValueError("Can only start backprop from scalar node")
        self.grad = np.ones_like(self.data) # shape could be like (1,1,1,...)
        for opnode in self._topo_order():
            opnode._backward()
    
    @property
    def shape(self):
        return self.data.shape
    
    @property 
    def ndim(self):
        return self.data.ndim
    
    @property 
    def size(self):
        return self.data.size
    
    @property 
    def T(self):
        return self.transpose()
    
    @property
    def item(self):
        return self.data.item()
    
    # -- Operators
            
    @wrap_numeric
    def __add__(self, other) -> Tensor:
        return ops.Add.apply(self, other) 
    
    @wrap_numeric
    def __mul__(self, other) -> Tensor:
        return ops.Mul.apply(self, other)
    
    @wrap_numeric
    def __pow__(self, other) -> Tensor:
        return ops.Pow.apply(self, other) 

    @wrap_numeric
    def __sub__(self, other):
        return ops.Sub.apply(self, other) 
    
    @wrap_numeric
    def __truediv__(self, other):
        return ops.Div.apply(self, other)
    
    def __neg__(self):
        return ops.Neg.apply(self) 
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other) -> Tensor:
        return self.matmul(other)
    
    def __getitem__(self, idx: tuple[int|slice]) -> Tensor:
        return ops.Select.apply(self, idx=idx)
    
    def log(self) -> Tensor:
        return ops.Log.apply(self) 

    def exp(self) -> Tensor:
        return ops.Exp.apply(self) 

    def abs(self) -> Tensor:
        return ops.Abs.apply(self)
    
    def sum(self, axis:Shape=None) -> Tensor: 
        return ops.Sum.apply(self, axis=axis) 
    
    def mean(self, axis:Shape=None) -> Tensor:
        ax = axis if axis is None else np.atleast_1d(axis)
        n = np.array(self.data.shape)[ax].prod()
        return self.sum(axis=axis) / float(n)

    def transpose(self) -> Tensor:
        return ops.Transpose.apply(self)
    
    def matmul(self, other) -> Tensor:
        return ops.Matmul.apply(self, other) 
    
    def batchmm(self, other) -> Tensor:
        return ops.BatchMM.apply(self, other)

    def sigmoid(self) -> Tensor:
        return ops.Sigmoid.apply(self)
    
    def logsoftmax(self, axis: int=-1) -> Tensor:
        return ops.LogSoftmax.apply(self, axis=axis)
    
    def softmax(self, axis: int=-1) -> Tensor:
        return self.logsoftmax(axis=axis).exp()
    
    def view(self, shape: tuple[int]) -> Tensor:
        return ops.View.apply(self, shape=shape) 
    
    def broadcast(self, shape: tuple[int]) -> Tensor:
        return ops.Broadcast.apply(self, shape=shape)
    
    def __repr__(self):
        return f"T({self.name}, d={self.data}, g={self.grad})"

if __name__ == "__main__":
    x = Tensor(np.ones((5,1)))
    a = Tensor(np.ones((5,)), name="a")
    print(a - x)
    # y = a + x
    # y.sum().backward()