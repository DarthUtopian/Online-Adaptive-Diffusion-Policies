import torch
import numpy as np
from abc import ABC, abstractclassmethod
# DPS implementation

class ConditionalMethod(ABC):
    def __init__(self, operator, noiser:str, **kwargs):
        self.operator = operator
        self.name_noiser = noiser
        
    def project(self, data, nosiy_measure, **kwargs):
        pass
        
    def grad_n_value(self, x_prev, x_0_hat, measure, **kwargs):
        n_node = x_0_hat.shape[0]
        if self.name_noiser == "gaussian":
            norm = torch.linalg.norm(measure - self.operator.apply(x_0_hat)) # useful
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_0_hat)[0]
        elif self.name_noiser == "poisson":
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        return norm_grad, norm
    
    @abstractclassmethod
    def conditional(self, x_t, x_0_hat, **kwargs):
        pass
    

class PosteriorSampling(ConditionalMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)
        
    def conditional(self, x_t, x_0_hat, measure, param_t=1.0, **kwargs):
        norm_grad, norm = self.grad_n_value(x_t, x_0_hat, measure, **kwargs)
        x_t = x_t - self.scale * param_t * norm_grad
        return x_t, norm
    
    
class Operator(ABC):
    def __init__(self, **kwargs):
        pass
    
    @abstractclassmethod
    def apply(self, data, **kwargs):
        pass
    
    def project(self, data, measure, **kwargs):
        pass