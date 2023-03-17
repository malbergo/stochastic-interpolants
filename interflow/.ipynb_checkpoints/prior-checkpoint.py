import numpy as np
import torch

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.multivariate_normal import  MultivariateNormal




class Prior(torch.nn.Module):
    """
    Abstract class for prior distributions of normalizing flows. The interface
    is similar to `torch.distributions.distribution.Distribution`, but batching
    is treated differently. Parameters passed to constructors are never batched,
    but are aware of the target (single) sample shape. The `forward` method then
    accepts just the batch size and produces a batch of samples of the known
    shape.
    """
    def forward(self, batch_size):
        raise NotImplementedError()
    def log_prob(self, x):
        raise NotImplementedError()
    def draw(self, batch_size):
        """Alias of `forward` to allow explicit calls."""
        return self.forward(batch_size)
    
    
    

class SimpleNormal(Prior):
    def __init__(self, loc, var, requires_grad = False):
        super().__init__()
        
        if requires_grad:
            loc.requires_grad_()
            var.requires_grad_()
        self.loc = loc
        self.var = var
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(self.loc), torch.flatten(self.var))
        self.shape = loc.shape
        
            
    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)
    def forward(self, batch_size):
        x = self.dist.sample((batch_size,))
        return torch.reshape(x, (-1,) + self.shape)
    
    def rsample(self, batch_size):
        x = self.dist.rsample((batch_size,))
        return torch.reshape(x, (-1,) + self.shape)
    
    
class SimpleUniform(Prior):
    def __init__(self, low, high, requires_grad = False):
        super().__init__()
        
        if requires_grad:
            loc.requires_grad_()
            var.requires_grad_()
        self.loc = loc
        self.var = var
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(self.loc), torch.flatten(self.var))
        # self.shape = loc.shape
        
            
    def log_prob(self,x):
        raise NotImplementedError()
    def forward(self, shape):
        # x = self.dist.sample((batch_size,))
        # return torch.reshape(x, (-1,) + self.shape)
        return (self.low - self.high)*torch.rand((shape,)) + self.high
    
    def rsample(self, batch_size):
        x = self.dist.rsample((batch_size,))
        return torch.reshape(x, (-1,) + self.shape)
    
    
class MultivariateNormal(Prior):
    def __init__(self, loc, cov, requires_grad = False):
        super().__init__()
        
        if requires_grad:
            loc.requires_grad_()
            cov.requires_grad_()
        self.loc = loc
        self.cov = cov
        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(self.loc, self.cov)
        # self.dist = torch.distributions.normal.Normal(
        #     torch.flatten(self.loc), torch.flatten(self.var))
        self.shape = loc.shape
        
            
    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return logp
    def forward(self, batch_size):
        x = self.dist.sample((batch_size,))
        return torch.reshape(x, (-1,) + self.shape)
    
    def rsample(self, batch_size):
        x = self.dist.rsample((batch_size,))
        return torch.reshape(x, (-1,) + self.shape)
    
    
    
    
class GMM(Prior):
    def __init__(self, loc=None, var=None, scale = 1.0, ndim = None, nmix= None, device='cpu', requires_grad=False):
        super().__init__()
        
        self.device = device
        self.scale = scale       ### only specify if loc is None
        def _compute_mu(ndim):
                return self.scale*torch.randn((1,ndim))
                        
        if loc is None:
            self.nmix = nmix
            self.ndim = ndim 
            loc = torch.cat([_compute_mu(ndim) for i in range(1, self.nmix + 1)], dim=0)
            var = torch.stack([1.0*torch.ones((ndim,)) for i in range(nmix)])
        else:
            self.nmix = loc.shape[0]
            self.ndim = loc.shape[1] ### locs should have shape [n_mix, ndim]
            
        self.loc = loc   ### locs should have shape [n_mix, ndim]
        self.var = var   ### should have shape [n_mix, ndim]
        
        if requires_grad:
            self.loc.requires_grad_()
            self.var.requires_grad_()
        
        mix = Categorical(torch.ones(self.nmix,))
        comp = Independent(Normal(
                     self.loc, self.var), 1)
        self.dist = MixtureSameFamily(mix, comp)
        
    def log_prob(self, x):
        logp = self.dist.log_prob(x)
        return logp
    
    
    def forward(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x
    
    def rsample(self, batch_size):
        x = self.dist.rsample((batch_size,))
        return x