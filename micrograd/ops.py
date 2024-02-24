import numpy as np

from micrograd.core import Op
from micrograd.conv import convolve2D, pad2D

TensorData = np.ndarray

class Add(Op):
    broadcastable = True
    @staticmethod
    def forward(ctx: dict, a: TensorData, b: TensorData):
        return a + b
    
    @staticmethod
    def backward(ctx: dict, gy: TensorData):
        # L = f(y) = f(a + b) 
        # => dL/da = dL/db = dL/dy * dy/da = dL/dy * 1
        return gy, gy

class Mul(Op):
    broadcastable = True
    @staticmethod
    def forward(ctx: dict, a: TensorData, b: TensorData):
        ctx['a'] = a; ctx['b'] = b 
        return a * b 
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        a, b = ctx['a'], ctx['b']
        return b*gy, a*gy

class Neg(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData):
        return -x
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        return -gy

class Sub(Op):
    broadcastable = True
    @staticmethod
    def forward(ctx: dict, a: TensorData, b: TensorData):
        ctx['a'] = a; ctx['b'] = b 
        return a - b 
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        return gy, -gy

class Div(Op):
    broadcastable = True
    @staticmethod 
    def forward(ctx: dict, a: TensorData, b: TensorData):
        ctx['a'] = a; ctx['b'] = b 
        return a / b 
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        a, b = ctx['a'], ctx['b']
        return gy / b, -a * gy / b**2

class Pow(Op):
    broadcastable = True
    @staticmethod 
    def forward(ctx: dict, a: TensorData, b: TensorData):
        ctx['a'] = a; ctx['b'] = b
        return a ** b

    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        a, b = ctx['a'], ctx['b']
        return gy * b * a**(b-1), gy * a**b * np.log(a)

class Log(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData):
        ctx['x'] = x 
        return np.log(x)
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        x = ctx['x']
        return gy / x

class Exp(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData):
        ctx['expx'] = np.exp(x)
        return ctx['expx']
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        expx = ctx['expx']
        return gy * expx

class Abs(Op):
    @staticmethod
    def forward(ctx: dict, x: TensorData):
        ctx['x'] = x 
        return np.abs(x)
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        x = ctx['x']
        return np.sign(x)
    
class Sum(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData, *, axis:int|tuple[int]|None=None):
        ctx['shape'] = x.shape
        y = x.sum(axis=axis, keepdims=True)
        ctx['bshape'] = y.shape # the broadcasting shape, has ones in the `axis` positions
        return y.squeeze()
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        # expand the correct dimensions (ie add ones to the shape) and broadcast (ie repeat gy)
        shape, bshape = ctx['shape'], ctx['bshape']
        return np.broadcast_to(gy.reshape(bshape), shape)

class Matmul(Op):
    @staticmethod
    def forward(ctx: dict, a: TensorData, b: TensorData):
        ctx['a'] = a; ctx['b'] = b 
        return a @ b

    @staticmethod
    def backward(ctx: dict, gy: TensorData):
        a, b = ctx['a'], ctx['b']
        return gy @ b.T, a.T @ gy

class BatchMM(Op):
    # row i of the input matrix X is the vector x_i
    # row i of the reulting matrix y equals A @ x_i
    # ie: BatchMM(A, X) = (A @ X.T).T = X @ A.T
    @staticmethod 
    def forward(ctx: dict, a: TensorData, x: TensorData):
        ctx['a'] = a; ctx['x'] = x 
        return x @ a.T

    # can be derived from matmul derivative rules
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        a, x = ctx['a'], ctx['x']
        return gy.T @ x, gy @ a

class Id(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData):
        return x
    
    @staticmethod
    def backward(ctx: dict, gy: TensorData):
        return gy

class ReLU(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData):
        ctx['x'] = x
        return np.maximum(0, x)
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        x = ctx['x']
        return gy * (x > 0)

class Sigmoid(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData):
        ctx['sigx'] = 1 / (1 + np.exp(-x))
        return ctx['sigx']
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        sigx = ctx['sigx']
        return gy * sigx * (1 - sigx)

class Tanh(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData):
        ctx['tanhx'] = np.tanh(x)
        return ctx['tanhx']
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        tanhx = ctx['tanhx']
        return gy * (1 - tanhx**2)

class LogSoftmax(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData, *, axis: int=-1):
        x = x - x.max(axis=axis, keepdims=True) # numerical trick to avoid over/underflow
        ctx['axis'] = axis
        ctx['e'] = np.exp(x)
        ctx['s'] = ctx['e'].sum(axis=axis, keepdims=True)
        return x - np.log(ctx['s'])
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        # probably not the best way to calculate this
        e, s, axis = ctx['e'], ctx['s'], ctx['axis']
        return gy - (e / s) * gy.sum(axis=axis, keepdims=True)

class Transpose(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData):
        return x.T
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        return gy.T

class View(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData, *, shape: tuple[int]):
        ctx['shape'] = x.shape
        return x.reshape(shape)
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        shape = ctx['shape']
        return gy.reshape(shape)

class Broadcast(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData, *, shape: tuple[int]):
        ctx['sx'] = x.shape; ctx["sy"] = shape
        return np.broadcast_to(x, shape)
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        sx, sy = ctx['sx'], ctx['sy']
        nx, ny = len(sx), len(sy)
        expdim = tuple(i for i in range(ny) 
                       if ny-i > nx               # added new dimension to the right
                       or sy[i] != sx[i-(ny-nx)]) # or expanded dimension of size 1
        # sum gradient along the expanded dimensions
        return gy.sum(axis=expdim).reshape(sx)

class Select(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData, idx: tuple[int|slice]):
        ctx['idx'] = idx; ctx['shape'] = x.shape
        return x[idx]

    @staticmethod
    def backward(ctx: dict, gy: TensorData):
        idx = ctx['idx']
        gx = np.zeros(ctx['shape'], dtype=gy.dtype)
        gx[idx] = gy
        return gx

class Conv2D(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData, w: TensorData, *, padding:tuple[int,int]=(0,0)):
        ctx['x'] = x; ctx['w'] = w; ctx['padding'] = padding
        return convolve2D(x, w, padding)
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        x, w, padding = ctx['x'], ctx['w'], ctx['padding']

        cout, cin, kh, kw = w.shape
        ph, pw = padding
        
        pyh = kh - ph - 1
        pyw = kw - pw - 1
        w_rev = np.flip(w, (-1, -2)).swapaxes(0, 1)
        gx = convolve2D(gy, w_rev, (pyh, pyw))

        gw = convolve2D(pad2D(x, ph, pw).swapaxes(0,1), gy.swapaxes(0,1)).swapaxes(0,1)

        return gx, gw

class Dropout(Op):
    @staticmethod 
    def forward(ctx: dict, x: TensorData, *, p: float=0.5):
        ctx['p'] = p
        if p == 0: return x
        mask = np.random.binomial(1, 1-p, size=x.shape) / (1-p)
        ctx['mask'] = mask
        return x * mask
    
    @staticmethod 
    def backward(ctx: dict, gy: TensorData):
        mask = ctx['mask']
        return gy * mask

