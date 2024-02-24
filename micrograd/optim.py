# Resources: - https://d2l.ai/chapter_optimization/sgd.html
#            - https://distill.pub/2017/momentum/
#            - https://optimization.cbe.cornell.edu/index.php?title=Momentum
#            - https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

import numpy as np 

from micrograd.core import Tensor

class Optimizer:
    def __init__(self, parameters: list[Tensor]):
        self.parameters = parameters

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)

    def step(self):
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, parameters: list[Tensor], 
                 lr: float=1e-2, momentum: float=0, dampening: float=0):
        super().__init__(parameters)
        self.lr = lr 
        self.mu = momentum
        self.tau = dampening
        # exponentially moving average of past gradients
        self.v = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        for i, p in enumerate(self.parameters):
            self.v[i] = self.mu*self.v[i] + (1-self.tau)*p.grad 
            p.data -= self.lr * self.v[i]

class Adagrad(Optimizer):
    def __init__(self, parameters: list[Tensor], lr: float=1e-2, eps=1e-6):
        super().__init__(parameters)
        self.lr = lr 
        self.eps = eps
        # s_i is estimate of of many times we have observed feature corresponding parameter i
        # we want to take smaller learning rates for common features, where we are confident in the right parameter
        self.s = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        for i, p in enumerate(self.parameters):
            self.s[i] += p.grad**2
            lr = self.lr / (np.sqrt(self.s[i] + self.eps))
            p.data -= lr * p.grad

class RMSprop(Optimizer):
    # In Adagrad the value of s keeps growing linearly without bound, slowing down the lr very quickly,
    # since s is just the sum of the squares of all past gradients.
    # In rmsprop we instead take s to be an exponentially moving average of the square of the gradients,
    # ie recent gradients are valued more.
    # Otherwise the two algorithms are identical.
    def __init__(self, parameters: list[Tensor], lr: float=1e-2, alpha=0.99, eps=1e-6):
        super().__init__(parameters)
        self.lr = lr 
        self.alpha = alpha # higher values makes s less sensitive to new data
        self.eps = eps 
        self.s = [np.zeros_like(p.data) for p in self.parameters]   
    
    def step(self):
        for i, p in enumerate(self.parameters):
            self.s[i] = self.alpha*self.s[i] + (1-self.alpha)*p.grad**2
            lr = self.lr / (np.sqrt(self.s[i] + self.eps))
            p.data -= lr * p.grad

class Adadelta(Optimizer):
    # No fixed learning rate, instead use estimate deltax of the per step change in the model parameters.
    # Otherwise same as RMSprop.
    def __init__(self, parameters: list[Tensor], rho=0.99, eps=1e-6):
        super().__init__(parameters)
        self.rho = rho # same as alpha in RMSprop
        self.eps = eps 
        self.s = [np.zeros_like(p.data) for p in self.parameters]   
        self.deltax = [np.zeros_like(p.data) for p in self.parameters]   
    
    def step(self):
        for i, p in enumerate(self.parameters):
            s, dx, eps, rho = self.s[i], self.deltax[i], self.eps, self.rho

            self.s[i] = rho*s + (1-rho)*p.grad**2
            lr = np.sqrt(dx + eps) / (np.sqrt(self.s[i] + eps))
            self.deltax[i] = rho*dx + (1-rho)*(lr*p.grad)**2
            p.data -= lr * p.grad

class Adam(Optimizer):
    # Combines a lot of features from previous optimizers
    def __init__(self, parameters: list[Tensor], lr: float=1e-3, beta1: float=0.9, beta2: float=0.999, eps=1e-6):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1 
        self.beta2 = beta2
        self.eps = eps 
        self.v = [np.zeros_like(p.data) for p in self.parameters] # running average of gradient    (beta1)
        self.s = [np.zeros_like(p.data) for p in self.parameters] # running 2nd moment of gradient (beta2)
        self.t = 1 # timestep
    
    def step(self):
        for i, p in enumerate(self.parameters):
            beta1, beta2, eps = self.beta1, self.beta2, self.eps

            self.v[i] = beta1*self.v[i] + (1-beta1)*p.grad 
            self.s[i] = beta2*self.s[i] + (1-beta2)*p.grad**2

            vhat = self.v[i] / (1 - beta1**self.t)
            shat = self.s[i] / (1 - beta2**self.t)

            p.data -= self.lr * vhat / (np.sqrt(shat) + eps)

        self.t += 1