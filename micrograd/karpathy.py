from __future__ import annotations
import numpy as np
from typing import Callable
from functools import wraps

def wrap_numeric(f: Callable) -> Callable:
    @wraps(f)
    def _wrapper(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return f(self, other)
    return _wrapper

class Scalar:
    def __init__(self, data, _children=(), _opname='', name='_'):
        self.data = data 
        self.grad = 0
        self.name = name
        self._opname = _opname
        self._children = _children 
        self._backward = lambda: None

    @wrap_numeric
    def __add__(self, other):
        out = Scalar(self.data + other.data, (self, other), '+')

        # y = out, x1 = self, x2 = other
        # y = x1 + x2, L = f(y) => dL/dx1 = dL/dx2 = dL/dy * 1 = out.grad
        # we do self.grad += dL/dx1 since this node might contribute to L through other paths aswell (diamond)
        def _backward():
            self.grad += 1 * out.grad # parent grad needs to be computed first 
            other.grad += 1 * out.grad
        out._backward = _backward

        return out 

    @wrap_numeric
    def __mul__(self, other):
        out = Scalar(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
        out._backward = _backward
        return out 

    @wrap_numeric
    def __pow__(self, other):
        out = Scalar(self.data ** other.data, (self, other), '*')
        # y = s**o => dy/ds = o*s**(o-i)
        #          => y = e^(ln(s)*o) => dy/do = ln(s) * s**o
        def _backward():
            x, p = self.data, other.data
            self.grad  += (p * x**(p-1)) * out.grad
            other.grad += (np.log(x) * x**p) * out.grad
        out._backward = _backward
        return out 

    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other): # so that int + Scalar works
        return self + other 
    
    def __rsub__(self, other):
        return self - other 
    
    def __rmul__(self, other):
        return self * other

    def log(self):
        out = Scalar(np.log(self.data), (self,), 'ln')
        def _backward():
            self.grad += (1/self.data) * out.grad 
        out._backward = _backward 
        return out

    def exp(self):
        out = Scalar(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += np.exp(self.data) * out.grad 
        out._backward = _backward 
        return out

    def _toposort(self) -> reversed[Scalar]:
        t = [] 
        visited = set()

        def _dfs(v):
            for c in v._children:
                if c in visited: continue
                visited.add(c)
                _dfs(c)
            t.append(v)

        _dfs(self)
        return reversed(t)

    def backward(self):
        self.grad = 1
        for s in self._toposort():
            print(s.name, s.grad)
            s._backward()

    def __repr__(self):
        return f"({self.name or '_'}: d={self.data}, g={self.grad})"

def main():
    x = Scalar(3, name='x')
    y1 = x*Scalar(2, name="c"); y1.name = "y1"
    y2 = y1
    L = y2 + y2

    L.backward()
    print(list(L._toposort()))
    print(x.grad, y1.grad, L.grad)

if __name__ == "__main__":
    main()