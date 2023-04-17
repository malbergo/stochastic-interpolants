import torch
import torch.distributions as D
from dataclasses import dataclass
from typing import Callable, Tuple
from functorch import vmap
from math import pi
from interflow import fabrics as fabrics
from interflow import stochastic_interpolant as stochastic_interpolant


Weight   = torch.tensor
Weights  = torch.tensor
Point    = torch.tensor
Time     = torch.tensor
Cov      = torch.tensor
Covs     = torch.tensor
Mean     = torch.tensor
Means    = torch.tensor
Velocity = torch.tensor


@dataclass
class GMMInterpolant:
    p0s: Weights # [N0]
    p1s: Weights # [N1]
    mu0s: Means  # [N0, d]
    mu1s: Means  # [N1, d]
    C0s: Covs    # [N0, d, d]
    C1s: Covs    # [N1, d, d]
    path: str
    gamma_type: str
    device: torch.cuda.device
    use_preconditioner: bool = False


    def __post_init__(self):
        # extract dimensions
        self.N0 = self.p0s.shape[0]
        self.N1 = self.p1s.shape[0]
        self.d  = self.C0s.shape[1]


        # ensure actual probabilities
        assert(torch.sum(self.p0s) == 1)
        assert(torch.sum(self.p1s) == 1)


        # setup interpolant
        self.gamma, self.gamma_dot, self.gg_dot = \
            fabrics.make_gamma(self.gamma_type)
        
        if self.path == 'linear':
            self.a    = lambda t: (1-t)
            self.b    = lambda t: t
            self.adot = lambda t: -1
            self.bdot = lambda t: 1
            
        if self.path == 'trig':
            self.a    = lambda t: torch.sqrt(1 - self.gamma(t)**2)*torch.cos(0.5*pi*t)
            self.b    = lambda t: torch.sqrt(1 - self.gamma(t)**2)*torch.sin(0.5*pi*t)
            self.adot = lambda t: -self.gg_dot(t)/torch.sqrt(1 - self.gamma(t)**2)*torch.cos(0.5*pi*t) \
                                    - 0.5*pi*torch.sqrt(1 - self.gamma(t)**2)*torch.sin(0.5*pi*t)
            self.bdot = lambda t: -self.gg_dot(t)/torch.sqrt(1 - self.gamma(t)**2)*torch.sin(0.5*pi*t) \
                                    + 0.5*pi*torch.sqrt(1 - self.gamma(t)**2)*torch.cos(0.5*pi*t)
            
        elif self.path == 'encoding-decoding':
            self.a    = lambda t: torch.where(t <= 0.5, torch.cos(pi*t)**2, 0.)
            self.b    = lambda t: torch.where(t > 0.5,  torch.cos(pi*t)**2, 0.)
            self.adot = lambda t: torch.where(t <= 0.5, -2*pi*torch.cos(pi*t)*torch.sin(pi*t), 0.)
            self.bdot = lambda t: torch.where(t > 0.5,  -2*pi*torch.cos(pi*t)*torch.sin(pi*t), 0.)
            
        elif self.path == 'one_sided':
            self.a    = lambda t: (1-t)
            self.b    = lambda t: 0
            self.adot = lambda t: -1
            self.bdot = lambda t: 0


        # set up distributions for sampling
        self.rho0 = D.MixtureSameFamily(
            D.Categorical(self.p0s),
            D.MultivariateNormal(self.mu0s, self.C0s)
        )
        
        self.rho1 = D.MixtureSameFamily(
            D.Categorical(self.p1s),
            D.MultivariateNormal(self.mu1s, self.C1s)
        )
        
        
        # batch s and v for input to integrators
        if self.use_preconditioner:
            # divide through by common, large factors in high-d
            # factor out most likely mode in the mixture
            self.v = lambda txs: self.batch_velocity(txs, compute_score=False)
            self.s = lambda txs: self.batch_velocity(txs, compute_score=True)
        else:
            self.v = vmap(self.calc_v)
            self.s = vmap(self.calc_s)
        
        
    def get_velocities(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Convert v and s into torch modules for compatibility with Lightning."""
        class Velocity(torch.nn.Module):
            def __init__(self, velocity: Callable[[torch.tensor], torch.tensor]):
                super().__init__()
                self.velocity = velocity
                
            def forward(self, tx: torch.tensor):
                return self.velocity(tx)
            
        return Velocity(self.v), Velocity(self.s)
    
    
    def log_rho0(self, samples: torch.tensor):
        return self.rho0.log_prob(samples)
    
    
    def log_rho1(self, samples: torch.tensor):
        return self.rho1.log_prob(samples)

    
    def sample_rho0(self, nsamples: int):
        return self.rho0.sample((nsamples,))


    def sample_rho1(self, nsamples: int):
        return self.rho1.sample((nsamples,))
    
    
    def calc_mij(
        self,
        mu0_i: Mean,
        mu1_j: Mean,
        t: Time
    ) -> Mean:
        return self.a(t)*mu0_i + self.b(t)*mu1_j


    def calc_Cij(
        self,
        C0_i: Cov,
        C1_j: Cov,
        t: Time
    ) -> Cov:
        return self.a(t)**2 * C0_i + self.b(t)**2 * C1_j
    
    
    def calc_Cij_eps(
        self, 
        C0_i: Cov, 
        C1_j: Cov, 
        t: Time
    ) -> Cov:
        return self.calc_Cij(C0_i, C1_j, t) + self.gamma(t)**2 * torch.eye(self.d, device=self.device)


    def eval_normal_ij(
        self,
        mu0_i: Mean,
        mu1_j: Mean,
        C0_i: Cov,
        C1_j: Cov,
        t: Time,
        x: Point,
        include_twopi: bool = True
    ) -> float:
        mij     = self.calc_mij(mu0_i, mu1_j, t)
        Cij_eps = self.calc_Cij_eps(C0_i, C1_j, t)

        if include_twopi:
            Z = (2*pi)**(self.d/2) * torch.exp(0.5*torch.slogdet(Cij_eps)[1])
        else:
            Z = torch.exp(0.5*torch.slogdet(Cij_eps)[1])

        inv_vec = torch.linalg.solve(Cij_eps, x-mij)
        exp_arg = -0.5*(x - mij) @ inv_vec
        exp_val = torch.exp(exp_arg)
        return  exp_val / Z


    def calc_mij_dot(
        self, 
        mu0_i: Mean, 
        mu1_j: Mean, 
        t: Time
    ) -> Mean:
        return self.adot(t)*mu0_i + self.bdot(t)*mu1_j


    def calc_Cij_dot(
        self, 
        C0_i: Cov, 
        C1_j: Cov, 
        t: Time
    ) -> Cov:
        return 2*(self.a(t)*self.adot(t)*C0_i + self.b(t)*self.bdot(t)*C1_j)
    
    
    def eval_distribution(
        self, 
        t: Time,
        x: Point,
        as_tensor: bool = False,
        include_twopi: bool = True
    ) -> float:
        map_func = lambda mu0_i, mu1_j, C0_i, C1_j: self.eval_normal_ij(
                mu0_i, mu1_j, C0_i, C1_j, t, x, include_twopi
            )
        mat = vmap(
            vmap(
                map_func, in_dims=(0, None, 0, None)
            ), in_dims = (None, 0, None, 0), out_dims=1
        )(self.mu0s, self.mu1s, self.C0s, self.C1s)

        if as_tensor:
            return self.p0s[:, None] * mat * self.p1s[None, :]
        else:
            return self.p0s @ mat @ self.p1s


    def calc_vij(
        self,
        p0_i: Weight,
        p1_j: Weight,
        mu0_i: Mean,
        mu1_j: Mean,
        C0_i: Cov,
        C1_j: Cov,
        t: Time,
        x: Point
    ) -> Velocity:
        mij_dot = self.calc_mij_dot(mu0_i, mu1_j, t)
        mij     = self.calc_mij(mu0_i, mu1_j, t)
        Cij_dot = self.calc_Cij_dot(C0_i, C1_j, t)
        Cij_eps = self.calc_Cij_eps(C0_i, C1_j, t)
        Nij     = self.eval_normal_ij(mu0_i, mu1_j, C0_i, 
                                      C1_j, t, x, 
                                      include_twopi=False)
        inv_vec = torch.linalg.solve(Cij_eps, x-mij)
        
        return p0_i*p1_j*(mij_dot + 0.5*Cij_dot @ inv_vec)*Nij


    def calc_v(
        self, 
        tx: torch.tensor, 
    ) -> Velocity:
        t, x = tx[0], tx[1:]
        
        vijs = vmap(
            vmap(
                lambda p0_i, p1_j, mu0_i, mu1_j, C0_i, C1_j: \
                    self.calc_vij(p0_i, p1_j, mu0_i, mu1_j, C0_i, C1_j, t, x), in_dims=(0, None, 0, None, 0, None)
            ), in_dims=(None, 0, None, 0, None, 0), out_dims=1
        )(self.p0s, self.p1s, self.mu0s, self.mu1s, self.C0s, self.C1s)
        
        return torch.sum(vijs.reshape(-1, self.d), axis=0) / self.eval_distribution(t, x, include_twopi=False)


    def calc_sij(
        self,
        p0_i: Weight,
        p1_j: Weight,
        mu0_i: Mean,
        mu1_j: Mean,
        C0_i: Cov,
        C1_j: Cov,
        t: Time,
        x: Point
    ) -> Velocity:
        mij     = self.calc_mij(mu0_i, mu1_j, t)
        Cij_eps = torch.inverse(self.calc_Cij_eps(C0_i, C1_j, t))
        Nij     = self.eval_normal_ij(mu0_i, mu1_j, C0_i, C1_j, t, x, include_twopi=False)
        inv_vec = torch.linalg.solve(Cij_eps, x-mij)

        return -p0_i*p1_j*inv_vec*Nij


    def calc_s(
        self, 
        tx: torch.tensor
    ) -> Velocity:
        t, x = tx[0], tx[1:]
        sijs = vmap(
            vmap(
                lambda p0_i, p1_j, mu0_i, mu1_j, C0_i, C1_j: \
                    self.calc_sij(p0_i, p1_j, mu0_i, mu1_j, C0_i, C1_j, t, x), in_dims=(0, None, 0, None, 0, None)
            ), in_dims=(None, 0, None, 0, None, 0), out_dims=1
        )(self.p0s, self.p1s, self.mu0s, self.mu1s, self.C0s, self.C1s)

        return torch.sum(sijs.reshape(-1, self.d), axis=0) / self.eval_distribution(t, x, include_twopi=False)


    def calc_velocities(
        self,
        t: Time,
        x: Point,
        compute_score: bool
    ) -> torch.tensor:
        func = self.calc_sij if compute_score else self.calc_vij

        return vmap(
            vmap(
                lambda p0_i, p1_j, mu0_i, mu1_j, C0_i, C1_j: \
                    func(p0_i, p1_j, mu0_i, mu1_j, C0_i, C1_j, t, x), in_dims=(0, None, 0, None, 0, None)
            ), in_dims=(None, 0, None, 0, None, 0), out_dims=1
        )(self.p0s, self.p1s, self.mu0s, self.mu1s, self.C0s, self.C1s)


    def batch_velocity(
        self, 
        txs: torch.tensor, # [bs, d+1]
        compute_score: bool
    ) -> Velocity:
        # get access to t and x individually
        ts = txs[:, 0]    # [bs]
        xs = txs[:, 1:]   # [bs, d]
        bs = ts.shape[0]


        # identify the most probable distribution over each element in the batch
        dists     = vmap(lambda t, x: self.eval_distribution(
            t, x, as_tensor=True, include_twopi=False
        ))(ts, xs) # [bs, N0, N1]
        dists     = dists.reshape((bs, self.N0*self.N1))                            # [bs, N0*N1]
        max_inds  = torch.argmax(dists, dim=1, keepdims=True)                       # [bs]
        max_dists = vmap(lambda tens, kk: tens[kk])(dists, max_inds)                # [bs]


        # compute the velocities or scores
        vijs = vmap(lambda t, x: self.calc_velocities(t, x, compute_score))(ts, xs) # [bs, N0, N1, d]
        vijs = vijs.reshape(bs, self.N0*self.N1, self.d)                            # [bs, N0*N1, d]


        # factor out the most probable distribution
        # TODO: check if we need to "plug in" the correct answers...
        dists /= max_dists
        vijs /= max_dists[:, :, None]
        
        return torch.sum(vijs, axis=1) / torch.sum(dists, axis=1)[:, None]
