import numpy as np
import torch
from . import util
import math
import hashlib
import os
from typing import Callable



class InputWrapper(torch.nn.Module):
    """
    Helper function that makes it so a velocity field parameterized by
    a neural network can be evaluated like v(x,t) e.g. having two inputs
    rather than stacking x [ndim] and t [1] to a [ndim + 1] shaped single
    input.
    """
    def __init__(self, v: torch.nn.Module):
        super(InputWrapper, self).__init__()
        self.v = v
        
    def net_inp(
        self,
        t: torch.tensor,  # [1]
        x: torch.tensor   # [batch x dim]
    ) -> torch.tensor:    # [batch x (1 + dim)]
        """Concatenate time over the batch dimension."""
        inp = torch.cat((t.repeat(x.shape[0]).unsqueeze(1), x), dim = 1)
        return inp
    
    def forward(self, x, t):
        tx = self.net_inp(t,x)
        return self.v(tx)


def make_fc_net(
    hidden_sizes: int, 
    in_size: int, 
    out_size: int, 
    inner_act: str, 
    final_act: str, 
    **config
):
    """Construct a fully-connected network."""
    sizes = [in_size] + hidden_sizes + [out_size]
    net = []
    for i in range(len(sizes) - 1):
        net.append(torch.nn.Linear(
            sizes[i], sizes[i+1]))
        if i != len(sizes) - 2:
            net.append(make_activation(inner_act))
            continue
        else:
            if make_activation(final_act):
                net.append(make_activation(final_act))
                
    net = torch.nn.Sequential(*net)
    return InputWrapper(net)


def make_It(
    path: str = 'linear', 
    gamma: Callable = None
):
    """gamma function must be specified if using the trigonometric interpolant"""
    if path == 'linear':
        It   = lambda t, x0, x1: (1 - t)*x0 + t*x1
        dtIt = lambda _, x0, x1: x1 - x0
        
    elif path == 'trig':
        if gamma == None:
            raise TypeError("Gamma function must be provided for trigonometric interpolant!")
        a    = lambda t: torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t)
        b    = lambda t: torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t)
        adot = lambda t: -self.gg_dot(t)/torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t) \
                                - 0.5*math.pi*torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t)
        bdot = lambda t: -self.gg_dot(t)/torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t) \
                                + 0.5*math.pi*torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t)

        It   = lambda t, x0, x1: self.a(t)*x0 + self.b(t)*x1
        dtIt = lambda t, x0, x1: self.adot(t)*x0 + self.bdot(t)*x1
        
    elif path == 'encoding-decoding':
        def I_fn(t, x0, x1):
                if t <= torch.tensor(1/2):
                    return (torch.cos(  math.pi * t)**2)*x0
                elif t >= torch.tensor(1/2):
                    return (torch.cos(  math.pi * t)**2)*x1

        It  = I_fn

        def dtI_fn(t,x0,x1):
            if t < torch.tensor(1/2):
                return -(1/2)* torch.sin(  math.pi * t) * torch.cos(  math.pi * t)*x0
            else:
                return -(1/2)* torch.sin(  math.pi * t) * torch.cos( math.pi * t)*x1

        dtIt = dtI_fn

    elif path == 'one-sided':
        It   = lambda t, x0, x1: (1-t)*x0 + torch.sqrt(t)*torch.randn(x0.shape)
        dtIt = lambda t, x0, x1: -x0 + 1/(2*torch.sqrt(t))*torch.randn(x0.shape)

    elif path == 'custom':
        return None, None

    else:
        raise NotImplementedError("The interpolant you specified is not implemented.")


    return It, dtIt


def make_gamma(
    gamma_type: str = 'brownian'
):
    """
    returns callable functions for gamma, gamma_dot,
    and gamma(t)*gamma_dot(t) to avoid numerical divide by 0s,
    e.g. if one is using the brownian (default) gamma.
    """
    if gamma_type == 'brownian':
        gamma = lambda t: torch.sqrt(t*(1-t))
        gamma_dot = lambda t: (1/(2*torch.sqrt(t*(1-t)))) * (1 -2*t)
        gg_dot = lambda t: (1/2)*(1-2*t)
        
    elif gamma_type == 'zero':
        gamma = gamma_dot = gg_dot = lambda t: torch.zeros_like(t)

    elif gamma_type == 'bsquared':
        gamma = lambda t: t*(1-t)
        gamma_dot = lambda t: 1 -2*t
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    elif gamma_type == 'sinesquared':
        gamma = lambda t: torch.sin(math.pi * t)**2
        gamma_dot = lambda t: 2*math.pi*torch.sin(math.pi * t)*torch.cos(math.pi*t)
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    elif gamma_type == 'sigmoid':
        f = torch.tensor(10.0)
        gamma = lambda t: torch.sigmoid(f*(t-(1/2)) + 1) - torch.sigmoid(f*(t-(1/2)) - 1) - torch.sigmoid((-f/2) + 1) + torch.sigmoid((-f/2) - 1)
        gamma_dot = lambda t: (-f)*( 1 - torch.sigmoid(-1 + f*(t - (1/2))) )*torch.sigmoid(-1 + f*(t - (1/2)))  + f*(1 - torch.sigmoid(1 + f*(t - (1/2)))  )*torch.sigmoid(1 + f*(t - (1/2)))
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    else:
        raise NotImplementedError("The gamma you specified is not implemented.")
        
    return gamma, gamma_dot, gg_dot


def make_activation(
    act: str
):
    if act == 'elu':
        return torch.nn.ELU()
    if act == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif act == 'elu':
        return torch.nn.ELU()
    elif act == 'relu':
        return torch.nn.ReLU()
    elif act == 'tanh':
        return torch.nn.Tanh()
    elif act =='sigmoid':
        return torch.nn.Sigmoid()
    elif act == 'softplus':
        return torch.nn.Softplus()
    elif act == 'silu':
        return torch.nn.SiLU()
    elif act == 'Sigmoid2Pi':
        class Sigmoid2Pi(torch.nn.Sigmoid):
            def forward(self, input):
                return 2*np.pi*super().forward(input) - np.pi
        return Sigmoid2Pi()
    elif act == 'none' or act is None:
        return None
    else:
        raise NotImplementedError(f'Unknown activation function {act}')


def make_optimizer(model, opt_type, opt_cfg, model_prefix):
    opt = getattr(torch.optim, opt_type)(model.parameters(), **opt_cfg)
    maybe_load_optimizer(opt, model_prefix=model_prefix, opt_cfg=opt_cfg)
    return opt


def maybe_load_optimizer(optimizer, *, model_prefix, opt_cfg):
    if model_prefix is None:
        return

    opt_path = '{}.opt.pt'.format(model_prefix)
    if not os.path.exists(opt_path):
        print(f"No optimizer state found at {opt_path}")
        return

    state_dict = torch.load(opt_path)
    # Ovverride lr for loaded optimizer
    if opt_cfg.get('lr', None) is not None:
        for param_group in state_dict['param_groups']:
            param_group['lr'] = opt_cfg['lr']
    try:
        optimizer.load_state_dict(state_dict)
    except ValueError as e:
        print("WARNING: Could not load optimizer state", e)


def make_scheduler(*, optimizer, sched_type, **sched_config):
    if sched_type == 'Adal':
        return getattr(scheduler, sched_type)(optimizer, **sched_config)
    else:
        class Mixing(getattr(torch.optim.lr_scheduler, sched_type)):
            def __init__(self, *, N, lr_m, **kwargs):
                super().__init__(**kwargs)
            def step(self, *args, **kwargs):
                super().step()

        return Mixing(optimizer=optimizer, **sched_config)

    
def load_model(model, *, model_prefix, device, _run=None):
    assert model_prefix is not None
    return maybe_load_model(model, model_prefix=model_prefix, device=device, _run=_run)


def maybe_load_model(model, *, model_prefix, strict=False):
    if model_prefix is None:
        print('model_prefix is None, skipping model load')
        return
    # if hvd.rank() == 0:
    model_path = '{}.pt'.format(model_prefix)
    print("Loading model %s" % (model_path, ))
    # if _run is not None:
        # _run.open_resource(model_path) # log as a resource
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=strict)
    # model.to(device)
    print("---> after loading model hash:", util.hash_model_parameters(model).hex())
