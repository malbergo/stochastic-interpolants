import torch
import torch.distributions as D
from dataclasses import dataclass
from typing import Callable, Tuple
from functorch import vmap
from math import pi
from interflow import fabrics as fabrics


Weight   = torch.tensor
Weights  = torch.tensor
Point    = torch.tensor
Time     = torch.tensor
Cov      = torch.tensor
Covs     = torch.tensor
Mean     = torch.tensor
Means    = torch.tensor
Velocity = torch.tensor


# torch.set_default_tensor_type(torch.cuda.DoubleTensor)

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


    def __post_init__(self):
        # extract dimensions
        self.N0 = self.p0s.shape[0]
        self.N1 = self.p1s.shape[0]
        self.d  = self.C0s.shape[1]


        # ensure actual probabilities
        assert(torch.sum(self.p0s) == 1)
        assert(torch.sum(self.p1s) == 1)


        # setup interpolant
        self.gamma, self.gamma_dot, self.gg_dot = fabrics.make_gamma(self.gamma_type)
        
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
    
    
    def calc_Cij_gam(
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
        x: Point
    ) -> float:
        mij     = self.calc_mij(mu0_i, mu1_j, t).double()
        Cij_gam = self.calc_Cij_gam(C0_i, C1_j, t).double()
        Z       = (2*pi)**(self.d/2) * torch.exp(0.5*torch.slogdet(Cij_gam)[1])
        x = x.double()
        # print("Z:", Z)
        
        
        # print("NIJ:", -0.5*(x - mij) @ torch.inverse(Cij_gam) @ (x - mij) - torch.log(Z) )
        return torch.exp(-0.5*(x - mij) @ torch.inverse(Cij_gam) @ (x - mij)) / Z
    
    def eval_log_normal_ij(
        self,
        mu0_i: Mean,
        mu1_j: Mean,
        C0_i: Cov,
        C1_j: Cov,
        t: Time,
        x: Point
    ) -> float:
        mij     = self.calc_mij(mu0_i, mu1_j, t).double()
        Cij_gam = self.calc_Cij_gam(C0_i, C1_j, t).double()
   
        mahal_dist = torch.einsum('ki,ij,kj->k', (x-mij), torch.linalg.inv(Cij_gam), (x-mij))
        n_ij = -0.5*(self.d*torch.log(torch.tensor(2)*torch.pi) \
                + torch.linalg.slogdet(Cij_gam)[-1] + mahal_dist)

        return n_ij
    
    def eval_normal_ij_new(
        self,
        mu0_i: Mean,
        mu1_j: Mean,
        C0_i: Cov,
        C1_j: Cov,
        t: Time,
        x: Point
    ) -> float:
        mij     = self.calc_mij(mu0_i, mu1_j, t)
        Cij_gam = self.calc_Cij_gam(C0_i, C1_j, t)


        mahal_dist = torch.einsum('ki,ij,kj->k', (x-mij), torch.linalg.inv(Cij_gam), (x-mij))
        
        n_ij = -0.5*(self.d*torch.log(torch.tensor(2)*torch.pi) \
                + torch.linalg.slogdet(Cij_gam)[-1] + mahal_dist)
        
        
        return torch.exp(n_ij.double())

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
        x: Point
    ) -> float:
        mat = vmap(
            vmap(
                lambda mu0_i, mu1_j, C0_i, C1_j: self.eval_normal_ij(mu0_i, mu1_j, C0_i, C1_j, t, x), in_dims=(0, None, 0, None)
            ), in_dims = (None, 0, None, 0), out_dims=1
        )(self.mu0s, self.mu1s, self.C0s, self.C1s)
        
        return self.p0s.double() @ mat @ self.p1s.double()
    


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
        mij_dot     = self.calc_mij_dot(mu0_i, mu1_j, t).double()
        mij         = self.calc_mij(mu0_i, mu1_j, t).double()
        Cij_dot     = self.calc_Cij_dot(C0_i, C1_j, t).double()
        Cij_gam_inv = torch.inverse(self.calc_Cij_gam(C0_i, C1_j, t)).double()
        Nij         = self.eval_normal_ij(mu0_i, mu1_j, C0_i, C1_j, t, x).double()
        

        return (p0_i*p1_j*(mij_dot + 0.5*Cij_dot @ Cij_gam_inv @ (x - mij))).double()*Nij


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
        

        return torch.sum(vijs.reshape(-1, self.d), axis=0) / self.eval_distribution(t, x)


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
        mij         = self.calc_mij(mu0_i, mu1_j, t).double()
        Cij_gam_inv = torch.inverse(self.calc_Cij_gam(C0_i, C1_j, t)).double()
        Nij         = self.eval_normal_ij(mu0_i, mu1_j, C0_i, C1_j, t, x).double()

        return (-p0_i*p1_j*Cij_gam_inv @ (x - mij).double()) * Nij


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

        return torch.sum(sijs.reshape(-1, self.d), axis=0) / self.eval_distribution(t, x)
    
    
    
    
    
@dataclass
class GMMInterpolant_Simple:
    p0s: Weights # [N0]
    p1s: Weights # [N1]
    mu0s: Means  # [N0, d]
    mu1s: Means  # [N1, d]
    C0s: Covs    # [N0, d, d]
    C1s: Covs    # [N1, d, d]
    path: str
    gamma_type: str
    device: torch.cuda.device


    def __post_init__(self):
        # extract dimensions
        self.N0 = self.p0s.shape[0]
        self.N1 = self.p1s.shape[0]
        self.d  = self.C0s.shape[1]


        # ensure actual probabilities
        assert(torch.sum(self.p0s) == 1)
        assert(torch.sum(self.p1s) == 1)


        # setup interpolant
        self.gamma, self.gamma_dot, self.gg_dot = fabrics.make_gamma(self.gamma_type)
        
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
        
        
    
    def eval_distribution_loop(
        self, 
        t: Time,
        x: Point
    ) -> float:
        """
        Only for \rho0 = standard gaussian, just one mixture term
        """
        log_pdfs = torch.empty((self.N1))
        for j in range(self.N1):
            mu0 = self.mu0s[0]
            mu1_j = self.mu1s[j]
            C0 = self.C0s[0]
            C1_j = self.C1s[j]
            log_pdfs[j] = self.p1s[j] + self.eval_log_normal_ij(mu0, mu1_j, C0, C1_j, t, x)
            
        print("DID THIS WORK:", torch.logsumexp(log_pdfs, dim=0))
        return torch.exp(torch.logsumexp(log_pdfs, dim=0).double())
        
        # print("MAT?:", mat, mat.shape)
        # return self.p0s.double() @ mat @ self.p1s.double()
        
        
    def eval_log_normal_ij(
        self,
        mu0_i: Mean,
        mu1_j: Mean,
        C0_i: Cov,
        C1_j: Cov,
        t: Time,
        x: Point
    ) -> float:
        mij     = self.calc_mij(mu0_i, mu1_j, t).double()
        Cij_gam = self.calc_Cij_gam(C0_i, C1_j, t).double()
        # Z       = (2*pi)**(self.d/2) * torch.exp(0.5*torch.slogdet(Cij_gam)[1])
        # print("Z:", Z)
        
        # return torch.exp(-0.5*(x - mij) @ torch.inverse(Cij_gam) @ (x - mij)) / Z
        print("SHAPES:", x.shape, mij.shape, Cij_gam.shape)
        # mahal_dist = ((x-mij).T @ torch.linalg.inv(Cij_gam) @ (x-mij))
        mahal_dist = torch.einsum('ki,ij,kj->k', (x-mij), torch.linalg.inv(Cij_gam), (x-mij))
        
        n_ij = -0.5*(self.d*torch.log(torch.tensor(2)*torch.pi) \
                + torch.linalg.slogdet(Cij_gam)[-1] + mahal_dist)
        # print("N_IJ:", n_ij)
        
        return n_ij
    
    
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
    
    
    def calc_Cij_gam(
        self, 
        C0_i: Cov, 
        C1_j: Cov, 
        t: Time
    ) -> Cov:
        return self.calc_Cij(C0_i, C1_j, t) + self.gamma(t)**2 * torch.eye(self.d, device=self.device)