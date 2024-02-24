import numpy as np 
from numpy.lib.stride_tricks import sliding_window_view

def pad2D(x: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """Pads if p > 0, crops if p < 0. `x` is assumed to have shape (b, c, h, w).

    Both padding and cropping is symmetric, so output will have shape (b, c, h+2*ph, w+2*pw)"""
    assert x.ndim == 4 
    if ph < 0: x = x[:, :, -ph:ph, :]
    if pw < 0: x = x[:, :, :, -pw:pw]
    return np.pad(x, ((0, 0), (0, 0), (max(0,ph), max(0,ph)), (max(0,pw), max(0,pw))))

# like 10x slower than torch.nn.Conv2D :(
def convolve2D(inp: np.ndarray, filtr: np.ndarray, padding: tuple[int,int] = (0, 0)) -> np.ndarray:
    assert inp.ndim == filtr.ndim == 4, (inp.ndim, filtr.ndim)
    (b, c, h, w) = inp.shape 
    (cout, cprime, kh, kw) = filtr.shape 
    assert cprime == c, (inp.shape, filtr.shape)

    ph, pw = padding
    x = np.pad(inp, ((0, 0), (0, 0), (ph, ph), (pw, pw)))

    wout = w + 2*pw - kw + 1
    hout = h + 2*ph - kh + 1
    # print("out: ", hout, wout)

    u = sliding_window_view(x, (b, c, kh, kw)) # get all subarrays (essentially free operation, only uses stride tricks)
    u = u.reshape(hout*wout, b, c*kh*kw) # flatten each subarray, except along batch dim (TODO: this is slow for some reason)
    u = u.swapaxes(0, 1) # put batch dim as first dim, for matmul later. 2nd dim contains each subarray 

    w = filtr.reshape((cout, c*kh*kw)).T

    y = u @ w # do the actual computation
    assert y.shape == (b, hout*wout, cout)

    return y.swapaxes(1, 2).reshape(b, cout, hout, wout)