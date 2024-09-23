import numpy as np
from micrograd.core import Tensor
from math import ceil

class DataLoader:
  def __init__(self, x: Tensor, y: Tensor, batch_size=32, shuffle=True):
    assert len(x) == len(y), f"Length mismatch: {len(x)} != {len(y)}"
    self.x = x 
    self.y = y
    self.batch_size = batch_size
    self.shuffle = shuffle
    self._reset()
  
  def _reset(self):
    self.idx = 0
    if self.shuffle: 
      ix = np.random.permutation(len(self.x))
      self.x = self.x[ix]
      self.y = self.y[ix]

  def __iter__(self):
    return self

  def __len__(self):
    return ceil(len(self.x) / self.batch_size)
  
  def __next__(self):
    if self.idx >= len(self.x):
      self._reset()
      raise StopIteration

    batch = (self.x[self.idx:self.idx+self.batch_size], 
             self.y[self.idx:self.idx+self.batch_size])
    self.idx += self.batch_size
    return batch

def main():
  x = Tensor(np.arange(0, 700, 1).reshape(100, 7))
  y = Tensor(np.ones(100))
  dl = DataLoader(x, y, batch_size=32, shuffle=False)

  for (x,y) in dl:
    print(x.shape, y.shape, x[0].data, y[0].data)
  for (x,y) in dl:
    print(x.shape, y.shape, x[0].data, y[0].data)
  
  print(len(dl))

if __name__ == '__main__':
  main()