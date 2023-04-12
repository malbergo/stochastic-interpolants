import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable, Any, Tuple
from torchdiffeq import odeint_adjoint as odeint
from functorch import jacfwd, vmap
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
    """Compute the divergence of f(x,t) with respect to x, assuming that x is batched."""
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



class Interpolant(torch.nn.Module):
    """
    Class for all things interpoalnt $x_t = I_t(x_0, x_1) + \gamma(t)z.
    
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
        
        
        self.gamma, self.gamma_dot, self.gg_dot = fabrics.make_gamma(gamma_type=gamma_type)
        if path == 'custom':
            print('Assuming interpolant was passed in directly...')
            self.It = It
            self.dtIt = dtIt
            assert self.It != None
            assert self.dtIt != None
        else:
            self.It, self.dtIt = fabrics.make_It(path, self.gamma)
        

    def calc_xt(self, t: Time, x0: Sample, x1: Sample):
        z = torch.randn(x0.shape).to(t)
        return self.It(t, x0, x1) + self.gamma(t)*z, z


    def calc_antithetic_xts(self, t: Time, x0: Sample, x1: Sample):
        z   = torch.randn(x0.shape).to(t)
        gam = self.gamma(t)
        It  = self.It(t, x0, x1)
        return It + gam*z, It - gam*z, z


    def forward(self, _):
        raise NotImplementedError("No forward pass for interpolant.")



class PFlowRHS(torch.nn.Module):
    def __init__(self, v: Velocity, s: Score, interpolant: Interpolant, sample_only=False):
        super(PFlowRHS, self).__init__()
        self.v = v
        self.s = s
        self.interpolant = interpolant
        self.sample_only = sample_only


    def setup_rhs(self):
        def rhs(x: torch.tensor, t: torch.tensor):
            # tx = net_inp(t, x)
            self.v.to(x)
            self.s.to(x)

            t = t.unsqueeze(0)
            return self.v(x,t) - self.interpolant.gg_dot(t)*self.s(x,t)

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
    v: Velocity
    s: Score
    method: str
    interpolant: Interpolant
    n_step: int
    atol: torch.tensor = 5e-4
    rtol: torch.tensor = 5e-4
    sample_only: bool = False


    def __post_init__(self) -> None:
        self.rhs = PFlowRHS(v=self.v, s=self.s, interpolant=self.interpolant, sample_only=self.sample_only)
        self.rhs.setup_rhs()


    def rollout(self, x0: Sample, reverse=False):
        if reverse:
            integration_times = torch.linspace(1.0, 0.0, self.n_step).to(x0)
        else:
            integration_times = torch.linspace(0.0, 1.0, self.n_step).to(x0)
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
    v: Velocity
    s: Score
    dt: float
    eps: torch.tensor
    interpolant: Interpolant
    n_save: int
    n_likelihood: int = 1

    
    def __post_init__(self) -> None:
        """Initialize forward dynamics, reverse dynamics, and likelihood."""
        
        
        def b(x: torch.tensor, t: torch.tensor):
            self.v.to(x)
            self.s.to(x) ### needed to make lightning work. arises because using __post_init__
            return (self.v(x,t) - self.interpolant.gg_dot(t)*self.s(x,t))
        
        
        def bf(x: torch.tensor, t: torch.tensor):
            """Forward drift. Assume x is batched but t is not."""
            self.v.to(x)
            self.s.to(x) ### needed to make lightning work. arises because using __post_init__
            return self.b(x,t) + self.eps*self.s(x,t)


        def br(x: torch.tensor, t: torch.tensor):
            """Backwards drift. Assume x is batched but t is not."""
            self.v.to(x)
            self.s.to(x) ### needed to make lightning work. arises because using __post_init__
            with torch.no_grad():
                return self.b(x,t) - self.eps*self.s(x,t)


        def dt_logp(x: torch.tensor, t: torch.tensor):
            """Time derivative of the log-likelihood, assumed integrating from 1 to 0.
            Assume x is batched but t is not.
            """
            # tx     = net_inp(t, x)
            score  = self.s(x,t)
            s_norm = torch.linalg.norm(score, axis=-1)**2
            return -(compute_div(self.bf, x, t) + self.eps*s_norm)

        
        self.b  = b
        self.bf = bf
        self.br = br
        self.dt_logp = dt_logp
        
        
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
        n_step = int(torch.ceil(1.0/self.dt))
        bs, d  = init.shape
        likes  = torch.zeros((self.n_likelihood, bs)).to(init)
        xs     = torch.zeros((self.n_likelihood, bs, d)).to(init)


        # ensure we integrate to exactly t=1
        assert (n_step % self.n_save) == 0
        assert n_step*self.dt == 1.0


        # TODO: for more general dimensions, need to replace these 1's by something else.
        x    = init.repeat((self.n_likelihood, 1, 1)).reshape((self.n_likelihood*bs, d))
        like = torch.zeros(self.n_likelihood*bs).to(x)
        save_counter = 0

        for ii in range(n_step):
            t    = 1 - torch.tensor(ii*self.dt).to(x)
            # print("X SHAPE:", x.shape)
            # print("T SHAPE:", t.shape)
            x    = self.step_reverse_heun(x, t)
            like = self.step_likelihood(like, x, t-self.dt) # semi-implicit discretization?
                             
        xs, likes = x.reshape((self.n_likelihood, bs, d)), like.reshape((self.n_likelihood, bs))


        # only output mean
        return xs, torch.mean(likes, axis=0)

        # output all predictions
        # return xs, likes


    def rollout_forward(
        self, 
        init: Sample, # [batch x dim]
        method: str = 'heun'
    ) -> torch.tensor:
        """Solve the forward-time SDE to generate a batch of samples."""
        n_step     = int(torch.ceil(1.0/self.dt))
        save_every = int(n_step/self.n_save)
        xs         = torch.zeros((self.n_save, *init.shape)).to(init)
        x          = init


        # ensure we integrate to exactly t=1
        assert (n_step % self.n_save) == 0
        assert n_step*self.dt == 1.0

        save_counter = 0
        for ii in range(n_step):
            t = torch.tensor(ii*self.dt).to(x)
            t = t.unsqueeze(0)
            if method == 'heun':
                x = self.step_forward_heun(x, t)
            else:
                x = self.step_forward(x,t)

            if ((ii+1) % save_every) == 0:
                xs[save_counter] = x
                save_counter += 1


        return xs


def loss_per_sample_sv(
    v: Velocity,
    s: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant,
    loss_fac: float
) -> torch.tensor:
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    dtIt        = interpolant.dtIt(t, x0, x1)
    dt_gam      = interpolant.gamma_dot(t)
    vtp         = v(xtp, t)
    vtm         = v(xtm, t)
    loss_v      = 0.5*torch.sum(vtp**2) - torch.sum((dtIt) * vtp)
    loss_v      += 0.5*torch.sum(vtm**2) - torch.sum((dtIt) * vtm)
    
    stp         = s(xtp, t)
    stm         = s(xtm, t)
    loss_s      = 0.5*torch.sum(stp**2) + (1 / interpolant.gamma(t))*torch.sum(stp*z)
    loss_s      += 0.5*torch.sum(stm**2) - (1 / interpolant.gamma(t))*torch.sum(stm*z)
    
    return (loss_v, loss_fac * loss_s)

    
x0_batch_loss_sv = vmap(loss_per_sample_sv, in_dims=(None, None,    0, None, None, None, None), randomness='different')
x1_batch_loss_sv = vmap(x0_batch_loss_sv,   in_dims=(None, None, None,    0, None, None, None), randomness='different')
t_batch_loss_sv  = vmap(x1_batch_loss_sv,   in_dims=(None, None, None, None,    0, None, None), randomness='different')


def loss_sv(
    v: Velocity, 
    s: Score,
    x0s: torch.tensor,
    x1s: torch.tensor, 
    ts: torch.tensor, 
    interpolant: Interpolant,
    loss_fac: torch.tensor
) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:

    losses_v, losses_s = t_batch_loss_sv(v, s, x0s, x1s, ts, interpolant, loss_fac)

    loss_v = losses_v.mean()
    loss_s = losses_s.mean()
    return loss_v + loss_s, (loss_v, loss_s)
