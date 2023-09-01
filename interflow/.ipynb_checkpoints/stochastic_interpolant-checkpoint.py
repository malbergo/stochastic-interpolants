import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable, Any, Tuple
from torchdiffeq import odeint_adjoint as odeint
from torch import vmap
import math
from . import fabrics


Time     = torch.tensor
Sample   = torch.tensor
Velocity = torch.nn.Module
Score    = torch.nn.Module



def compute_div(
    f: Callable[[Time, Sample], torch.tensor],
    x: torch.tensor,
    t: torch.tensor  # [batch x dim]
) -> torch.tensor:
    """Compute the divergence of f(x,t) with respect to x, assuming that x is batched. Assumes data is [bs, d]"""
    bs = x.shape[0]
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        t.requires_grad_(True)
        f_val = f(x, t)
        divergence = 0.0
        for i in range(x.shape[1]):
            divergence += \
                    torch.autograd.grad(
                            f_val[:, i].sum(), x, create_graph=True
                        )[0][:, i]

    return divergence.view(bs)


class SFromEta(torch.nn.Module):
    """Class for turning a noise model into a score model."""
    def __init__(
        self,
        eta: Callable[[Sample, Time], torch.tensor],
        gamma: Callable[[Time], torch.tensor],
    ) -> None:
        super(SFromEta, self).__init__()
        self.eta = eta
        self.gamma = gamma
        
    def forward(self, x, t):
        val = (self.eta(x,t) / self.gamma(t))
        return val
    
    


    
class BFromVS(torch.nn.Module):
    
    """
    Class for turning a velocity model $v$ and a score model $s$ into a drift model $b$.
    If one-sided interpolation, gg_dot should be replaced with alpha*alpha_dot.
    """
    def __init__(
        self,
        v: Callable[[Sample, Time], torch.tensor],
        s: Callable[[Sample, Time], torch.tensor],
        gg_dot: Callable[[Time], torch.tensor],
    ) -> None:
        super(BFromVS, self).__init__()
        self.v = v
        self.s = s
        self.gg_dot = gg_dot

        
    def forward(self, x, t):
        return self.v(x, t) - self.gg_dot(t)*self.s(x, t)



class Interpolant(torch.nn.Module):
    """
    Class for all things interpoalnt $x_t = I_t(x_0, x_1) + \gamma(t)z.
    If path is one-sided, then interpolant constructs x_t = a(t) x_0 + b(t) x_1 with x_0 ~ N(0,1).
    
    path: str,    what type of interpolant to use, e.g. 'linear' for linear interpolant. see fabrics for options
    gamma_type:   what type of gamma function to use, e.g. 'brownian' for $\gamma(t) = \sqrt{t(1-t)}
    """
    def __init__(
        self, 
        path: str,
        gamma_type: str,
        gamma: Callable[[Time], torch.tensor]          = None,
        gamma_dot: Callable[[Time], torch.tensor]      = None,
        gg_dot: Callable[[Time], torch.tensor]         = None,
        It: Callable[[Time, Sample, Sample], Sample]   = None, 
        dtIt: Callable[[Time, Sample, Sample], Sample] = None
    ) -> None:
        super(Interpolant, self).__init__()
        

        self.path = path
        if gamma == None:
            if self.path == 'one-sided-linear' or self.path == 'one-sided-trig': gamma_type = None
            self.gamma, self.gamma_dot, self.gg_dot = fabrics.make_gamma(gamma_type=gamma_type)
        else:
            self.gamma, self.gamma_dot, self.gg_dot = gamma, gamma_dot, gg_dot
        if self.path == 'custom':
            print('Assuming interpolant was passed in directly...')
            self.It = It
            self.dtIt = dtIt
            assert self.It != None
            assert self.dtIt != None
 

        self.It, self.dtIt, ab = fabrics.make_It(path, self.gamma, self.gamma_dot, self.gg_dot)
        self.a, self.adot, self.b, self.bdot = ab[0], ab[1], ab[2], ab[3]
        

    def calc_xt(self, t: Time, x0: Sample, x1: Sample):
        if self.path=='one-sided-linear' or self.path == 'mirror' or self.path=='one-sided-trig':
            return self.It(t, x0, x1)
        else:
            z = torch.randn(x0.shape).to(t)
            return self.It(t, x0, x1) + self.gamma(t)*z, z


    def calc_antithetic_xts(self, t: Time, x0: Sample, x1: Sample):
        """
        Used if estimating the score and not the noise (eta). 
        """
        if self.path=='one-sided-linear' or self.path == 'one-sided-trig':
            It_p = self.b(t)*x1 + self.a(t)*x0
            It_m = self.b(t)*x1 - self.a(t)*x0
            return It_p, It_m, x0
        else:
            z   = torch.randn(x0.shape).to(t)
            gam = self.gamma(t)
            It  = self.It(t, x0, x1)
            return It + gam*z, It - gam*z, z


    def forward(self, _):
        raise NotImplementedError("No forward pass for interpolant.")



class PFlowRHS(torch.nn.Module):
    def __init__(self, b: Velocity, interpolant: Interpolant, sample_only=False):
        super(PFlowRHS, self).__init__()
        self.b = b
        self.interpolant = interpolant
        self.sample_only = sample_only


    def setup_rhs(self):
        def rhs(x: torch.tensor, t: torch.tensor):
            self.b.to(x)

            t = t.unsqueeze(0)
            return self.b(x,t)

        self.rhs = rhs


    def forward(self, t: torch.tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (self.rhs(x, t), -compute_div(self.rhs, x, t))
    
    def reverse(self, t: torch.tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (-self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (-self.rhs(x, t), compute_div(self.rhs, x, t))
        
        

class MirrorPFlowRHS(torch.nn.Module):
    def __init__(self, s: Velocity, interpolant: Interpolant, sample_only=False):
        super(MirrorPFlowRHS, self).__init__()
        self.s = s
        self.interpolant = interpolant
        self.sample_only = sample_only


    def setup_rhs(self):
        def rhs(x: torch.tensor, t: torch.tensor):
            # tx = net_inp(t, x)
            self.s.to(x)

            t = t.unsqueeze(0)
            return self.interpolant.gg_dot(t)*self.s(x,t)

        self.rhs = rhs


    def forward(self, t: torch.tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (self.rhs(x, t), -compute_div(self.rhs, x, t))
    
    def reverse(self, t: torch.tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (-self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (-self.rhs(x, t), compute_div(self.rhs, x, t))





@dataclass
class PFlowIntegrator:
    b: Velocity
    method: str
    interpolant: Interpolant
    start_end: tuple = (0.0, 1.0)
    n_step: int = 5
    atol: torch.tensor = 1e-5
    rtol: torch.tensor = 1e-5
    sample_only: bool  = False
    mirror:      bool  = False


    def __post_init__(self) -> None:
        if self.mirror:
            self.rhs = MirrorPFlowRHS(s=self.b, interpolant=self.interpolant, sample_only=self.sample_only)
        else:
            self.rhs = PFlowRHS(b=self.b, interpolant=self.interpolant, sample_only=self.sample_only)
        self.rhs.setup_rhs()
        
        self.start, self.end = self.start_end[0], self.start_end[1]


    def rollout(self, x0: Sample, reverse=False):
        if reverse:
            integration_times = torch.linspace(self.end, self.start, self.n_step).to(x0)
        else:
            integration_times = torch.linspace(self.start, self.end, self.n_step).to(x0)
        dlogp = torch.zeros(x0.shape[0]).to(x0)

        state = odeint(
            self.rhs,
            (x0, dlogp),
            integration_times,
            method=self.method,
            atol=[self.atol, self.atol],
            rtol=[self.rtol, self.rtol]
        )

        x, dlogp = state
        return x, dlogp
    
    
    



@dataclass
class SDEIntegrator:
    b: Velocity
    s: Score
    eps: torch.tensor
    interpolant: Interpolant
    n_save: int = 4
    start_end: tuple = (0, 1)
    n_step: int = 100
    n_likelihood: int = 1

    
    def __post_init__(self) -> None:
        """Initialize forward dynamics, reverse dynamics, and likelihood."""
        
        def bf(x: torch.tensor, t: torch.tensor):
            """Forward drift. Assume x is batched but t is not."""
            self.b.to(x)
            self.s.to(x) ### needed to make lightning work. arises because using __post_init__
            return self.b(x,t) + self.eps*self.s(x,t)


        def br(x: torch.tensor, t: torch.tensor):
            """Backwards drift. Assume x is batched but t is not."""
            self.b.to(x)
            self.s.to(x) ### needed to make lightning work. arises because using __post_init__
            with torch.no_grad():
                return self.b(x,t) - self.eps*self.s(x,t)


        def dt_logp(x: torch.tensor, t: torch.tensor):
            """Time derivative of the log-likelihood, assumed integrating from 1 to 0.
            Assume x is batched but t is not.
            """
            score  = self.s(x,t)
            s_norm = torch.linalg.norm(score, axis=-1)**2
            return -(compute_div(self.bf, x, t) + self.eps*s_norm)

        
        self.bf = bf
        self.br = br
        self.dt_logp = dt_logp
        self.start, self.end = self.start_end[0], self.start_end[1]
        self.ts = torch.linspace(self.start, self.end, self.n_step)
        self.dt = (self.ts[1] - self.ts[0])

        
        
    def step_forward_heun(self, x: Sample, t: torch.tensor) -> Sample:
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW   = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        xhat = x + torch.sqrt(2*self.eps)*dW
        K1   = self.bf(xhat, t + self.dt)
        xp   = xhat + self.dt*K1
        K2   = self.bf(xp, t + self.dt)
        return xhat + 0.5*self.dt*(K1 + K2)


    def step_forward(self, x: Sample, t: torch.tensor) -> Sample:
        """Euler-Maruyama."""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        return x + self.bf(x, t)*self.dt + torch.sqrt(2*self.eps)*dW


    def step_reverse(self, x: Sample, t: torch.tensor) -> Sample:
        """Euler-Maruyama."""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        return x - self.br(x, t)*self.dt + torch.sqrt(2*self.eps)*dW
    
    
    def step_reverse_heun(self, x: Sample, t: torch.tensor) -> Sample:
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW   = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        xhat = x + torch.sqrt(2*self.eps)*dW
        K1   = self.br(xhat, t - self.dt)
        xp   = xhat - self.dt*K1
        K2   = self.br(xp, t - self.dt)
        return xhat - 0.5*self.dt*(K1 + K2)


    def step_likelihood(self, like: torch.tensor, x: Sample, t: torch.tensor) -> Sample:
        """Forward-Euler."""
        return like - self.dt_logp(x, t)*self.dt


    def rollout_likelihood(
        self, 
        init: Sample # [batch x dim]
    ) -> torch.tensor:
        """Solve the reverse-time SDE to generate a likelihood estimate."""
        bs, d  = init.shape
        likes  = torch.zeros((self.n_likelihood, bs)).to(init)
        xs     = torch.zeros((self.n_likelihood, bs, d)).to(init)



        # TODO: for more general dimensions, need to replace these 1's by something else.
        x    = init.repeat((self.n_likelihood, 1, 1)).reshape((self.n_likelihood*bs, d))
        like = torch.zeros(self.n_likelihood*bs).to(x)
        save_counter = 0

        for ii,t in enumerate(self.ts[:-1]):
            t = self.end - t.to(x)
            x    = self.step_reverse_heun(x, t)
            like = self.step_likelihood(like, x, t-self.dt) # semi-implicit discretization?
                             
        xs, likes = x.reshape((self.n_likelihood, bs, d)), like.reshape((self.n_likelihood, bs))

        
        # only output mean
        return xs, torch.mean(likes, axis=0)



    def rollout_forward(
        self, 
        init: Sample, # [batch x dim]
        method: str = 'heun'
    ) -> torch.tensor:
        """Solve the forward-time SDE to generate a batch of samples."""
        save_every = int(self.n_step/self.n_save)
        xs         = torch.zeros((self.n_save, *init.shape)).to(init)
        x          = init
        self.dt = self.dt.to(x)


        save_counter = 0
        
        for ii, t in enumerate(self.ts[:-1]):
            t = t.to(x)
            t = t.unsqueeze(0)
            if method == 'heun':
                x = self.step_forward_heun(x, t)
            else:
                x = self.step_forward(x,t)

            if ((ii+1) % save_every) == 0:
                xs[save_counter] = x
                save_counter += 1
            
        xs[save_counter] = x

        return xs
    
    
    
    
    
@dataclass
class MirrorSDEIntegrator:
    s: Score
    eps: torch.tensor
    interpolant: Interpolant
    n_save: int = 4
    start_end: tuple = (0, 1)
    n_step: int = 100
    n_likelihood: int = 1

    
    def __post_init__(self) -> None:
        """Initialize forward dynamics, reverse dynamics, and likelihood."""
        
        def bf(x: torch.tensor, t: torch.tensor):
            """Forward drift. Assume x is batched but t is not."""
            self.s.to(x) ### needed to make lightning work. arises because using __post_init__
    
            return -self.interpolant.gg_dot(t)*self.s(x,t) + self.eps*self.s(x,t)

        def br(x: torch.tensor, t: torch.tensor):
            """Backwards drift. Assume x is batched but t is not."""
            self.s.to(x) ### needed to make lightning work. arises because using __post_init__
            return (-self.interpolant.gg_dot(t) - self.eps)*self.s(x,t)


        def dt_logp(x: torch.tensor, t: torch.tensor):
            """Time derivative of the log-likelihood, assumed integrating from 1 to 0.
            Assume x is batched but t is not.
            """
            # tx     = net_inp(t, x)
            score  = self.s(x,t)
            s_norm = torch.linalg.norm(score, axis=-1)**2
            return -(compute_div(self.bf, x, t) + self.eps*s_norm)
        
        def eps_fn(eps0: torch.tensor, t: torch.tensor):
            # return eps0*torch.sqrt((1-t))
            # return torch.sqrt(eps0*t*(1-t))
            # return 4*eps0*(t-1/2)**2
            return eps0
        
        self.bf = bf
        self.br = br
        self.eps_fn = eps_fn
        self.dt_logp = dt_logp
        self.start, self.end = self.start_end[0], self.start_end[1]
        self.ts = torch.linspace(self.start, self.end, self.n_step)
        self.dt = (self.ts[1] - self.ts[0])
        
        
    def step_forward_heun(self, x: Sample, t: torch.tensor) -> Sample:
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW   = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        xhat = x + torch.sqrt(2*self.eps)*dW
        K1   = self.bf(xhat, t + self.dt)
        xp   = xhat + self.dt*K1
        K2   = self.bf(xp, t + self.dt)
        return xhat + 0.5*self.dt*(K1 + K2)


    def step_forward(self, x: Sample, t: torch.tensor) -> Sample:
        """Euler-Maruyama."""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        return x + self.bf(x, t)*self.dt + torch.sqrt(2*self.eps)*dW


    def step_reverse(self, x: Sample, t: torch.tensor) -> Sample:
        """Euler-Maruyama."""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        return x - self.br(x, t)*self.dt + torch.sqrt(2*self.eps)*dW
    
    
    def step_reverse_heun(self, x: Sample, t: torch.tensor) -> Sample:
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW   = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        xhat = x + torch.sqrt(2*self.eps)*dW
        K1   = self.br(xhat, t - self.dt)
        xp   = xhat - self.dt*K1
        K2   = self.br(xp, t - self.dt)
        return xhat - 0.5*self.dt*(K1 + K2)


    def step_likelihood(self, like: torch.tensor, x: Sample, t: torch.tensor) -> Sample:
        """Forward-Euler."""
        return like - self.dt_logp(x, t)*self.dt

    def rollout_likelihood(
        self, 
        init: Sample # [batch x dim]
    ) -> torch.tensor:
        """Solve the reverse-time SDE to generate a likelihood estimate."""
        bs, d  = init.shape
        likes  = torch.zeros((self.n_likelihood, bs)).to(init)
        xs     = torch.zeros((self.n_likelihood, bs, d)).to(init)



        # TODO: for more general dimensions, need to replace these 1's by something else.
        x    = init.repeat((self.n_likelihood, 1, 1)).reshape((self.n_likelihood*bs, d))
        like = torch.zeros(self.n_likelihood*bs).to(x)
        save_counter = 0

        for ii,t in enumerate(self.ts[:-1]):
            t = self.end - t.to(x)
            x    = self.step_reverse_heun(x, t)
            like = self.step_likelihood(like, x, t-self.dt) # semi-implicit discretization?
                             
        xs, likes = x.reshape((self.n_likelihood, bs, d)), like.reshape((self.n_likelihood, bs))

        
        # only output mean
        return xs, torch.mean(likes, axis=0)


    def rollout_forward(
        self, 
        init: Sample, # [batch x dim]
        method: str = 'heun'
    ) -> torch.tensor:
        """Solve the forward-time SDE to generate a batch of samples."""
        save_every = int(self.n_step/self.n_save)
        xs         = torch.zeros((self.n_save, *init.shape)).to(init)
        x          = init
        self.dt = self.dt.to(x)


        save_counter = 0
        for ii, t in enumerate(self.ts[:-1]):
            t = t.to(x)
            t = t.unsqueeze(0)
            if method == 'heun':
                x = self.step_forward_heun(x, t)
            else:
                x = self.step_forward(x,t)

            if ((ii+1) % save_every) == 0:
                xs[save_counter] = x
                save_counter += 1
            
        xs[save_counter] = x

        return xs
    
    
#### here ye we define all the possible losses! For b, v, s, eta

def loss_per_sample_b(
    b: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    dtIt        = interpolant.dtIt(t, x0, x1)
    gamma_dot   = interpolant.gamma_dot(t)
    btp         = b(xtp, t)
    btm         = b(xtm, t)
    loss        = 0.5*torch.sum(btp**2) - torch.sum((dtIt + gamma_dot*z) * btp)
    loss       += 0.5*torch.sum(btm**2) - torch.sum((dtIt - gamma_dot*z) * btm)
    
    return loss
    
def loss_per_sample_s(
    s: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    stp         = s(xtp, t)
    stm         = s(xtm, t)
    loss      = 0.5*torch.sum(stp**2) + (1 / interpolant.gamma(t))*torch.sum(stp*z)
    loss     += 0.5*torch.sum(stm**2) - (1 / interpolant.gamma(t))*torch.sum(stm*z)
    
    return loss


def loss_per_sample_eta(
    eta: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample via antithetic sampling."""
    xt, z   = interpolant.calc_xt(t, x0, x1)
    xt, t   = xt.unsqueeze(0), t.unsqueeze(0)
    eta_val = eta(xt, t)
    return 0.5*torch.sum(eta_val**2) + torch.sum(eta_val*z) 
    

def loss_per_sample_v(
    v: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample via antithetic sampling."""
    xt, z = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt  = interpolant.dtIt(t, x0, x1)
    v_val = v(xt, t)
    
    return 0.5*torch.sum(v_val**2) - torch.sum(dtIt * v_val)



def loss_per_sample_one_sided_b(
    b: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample."""
    xt  = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt        = interpolant.dtIt(t, x0, x1)
    # gamma_dot   = interpolant.gamma_dot(t)
    bt          = b(xt, t)
    loss        = 0.5*torch.sum(bt**2) - torch.sum((dtIt) * bt)
    
    return loss

def loss_per_sample_one_sided_v(
    v: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample."""
    xt    = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt  = interpolant.dtIt(t, x0, x1)
    vt = v(xt, t)
    loss  = 0.5*torch.sum(vt**2) - torch.sum((dtIt) * vt)
    
    return loss



def loss_per_sample_one_sided_s(
    s: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample via antithetic samples for x_t = sqrt(1-t)z + sqrt(t) x1 where z=x0.
    """
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    stp         = s(xtp, t)
    stm         = s(xtm, t)
    alpha       = interpolant.a(t)
    
    loss      = 0.5*torch.sum(stp**2) + (1 / (alpha))*torch.sum(stp*x0)
    loss     += 0.5*torch.sum(stm**2) - (1 / (alpha))*torch.sum(stm*x0)
    
    return loss


def loss_per_sample_one_sided_eta(
    eta: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample via samples for x_t = alpha(t)z + beta(t) x1 where z=x0.
    """
    xt         = interpolant.calc_xt(t, x0, x1)
    xt, t      = xt.unsqueeze(0), t.unsqueeze(0)
    etat         = eta(xt, t)
    loss      = 0.5*torch.sum(etat**2) + torch.sum(etat*x0)
    
    return loss


def loss_per_sample_mirror(
    s: Score,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample via antithetic sampling."""
    xt        = interpolant.calc_xt(t, x0, x1)
    xt, t     = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt      = interpolant.dtIt(t, x0, x1)
    st        = s(xt, t)

    loss      = 0.5*torch.sum(st**2) + (1 / interpolant.gamma(t))*torch.sum(st*x0)
    
    return loss


def make_batch_loss(loss_per_sample: Callable, method: str ='shared') -> Callable:
    """Convert a sample loss into a batched loss."""
    if method == 'shared':
        ## Share the batch dimension i for x0, x1, t
        in_dims_set = (None, 0, 0, 0, None)
        batched_loss = vmap(loss_per_sample, in_dims=in_dims_set, randomness='different')
        return batched_loss
    
    
### global variable for the available losses
losses = {'b':loss_per_sample_b, 's':loss_per_sample_s, 'eta':loss_per_sample_eta, 
          'v':loss_per_sample_v, 'one-sided-b':loss_per_sample_one_sided_b, 'one-sided-s':loss_per_sample_one_sided_s, 
          'one-sided-eta':loss_per_sample_one_sided_eta, 'one-sided-v':loss_per_sample_one_sided_v,
          'mirror': loss_per_sample_mirror}

    
def make_loss(
    method: str, 
    interpolant: Interpolant,
    loss_type: str
) -> Callable:
    
    loss_fn_unbatched = losses[loss_type]
    
    
    ## batchify the loss
    def loss(
        bvseta: Velocity, 
        x0s: torch.tensor, 
        x1s: torch.tensor,
        ts: torch.tensor, 
        interpolant: Interpolant,
    ) -> torch. tensor:
        
        loss_fn = make_batch_loss(loss_fn_unbatched, method)
        loss_val = loss_fn(bvseta, x0s, x1s, ts, interpolant)
        loss_val = loss_val.mean()
        return loss_val
    
    return loss



