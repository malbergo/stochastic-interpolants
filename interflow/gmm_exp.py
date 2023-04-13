import torch
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any, Callable


import interflow as itf
import interflow.prior as prior
import interflow.fabrics
import interflow.stochastic_interpolant as stochastic_interpolant
import interflow.gmm as gmm
from interflow.util import grab
import time


# TODO: install latex on pytorch singularity image?
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid']  = True
mpl.rcParams['axes.grid.which']  = 'both'
mpl.rcParams['xtick.minor.visible']  = True
mpl.rcParams['ytick.minor.visible']  = True
mpl.rcParams['xtick.minor.visible']  = True
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['grid.color'] = '0.8'
mpl.rcParams['grid.alpha'] = '0.5'
mpl.rcParams['figure.figsize'] = (8, 4)
mpl.rcParams['figure.titlesize'] = 12.5
mpl.rcParams['font.size'] = 12.5
mpl.rcParams['legend.fontsize'] = 12.5
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = False


@dataclass
class GMMExp:
    """Base class for GMM experiment."""
    ## experiment parameters
    learn_b: bool
    learn_eta: bool

    ## gmm parameters
    ndim: int
    N0: int
    N1: int
    scale: float
    scale_fac: float
    gamma_type: str
    path: str
    p0s: torch.tensor
    p1s: torch.tensor
    mu0s: torch.tensor
    mu1s: torch.tensor
    C0s: torch.tensor
    C1s: torch.tensor
    device: str

    ## network parameters
    base_lr: float
    hidden_sizes: list
    in_size: int
    out_side: int
    inner_act: str
    final_act: str

    ## optimization parameters
    N_era: int
    N_epoch: int
    N_t: int
    metrics_freq: int
    plot_freq: int
    n_save: int
    loss_fac: float
    clip: float

    ## misc parameters
    n_likelihood: int
    plot_bs: int
    prior_bs: int
    target_bs: int
    likelihood_bs: int
    dt: torch.tensor

    ## sampling parameters
    eps: torch.tensor

    ## logging parameters
    wandb_name: str


    def __post_init__(self) -> None:
        """Initialize the simulation."""

        ## set up the exact interpolant, and the samplers
        self.exact_interpolant = gmm.GMMInterpolant(
            self.p0s, self.p1s, self.mu0s, self.mu1s, self.C0s,
            self.C1s, self.path, self.gamma_type, device=self.device
        )
        self.target   = self.exact_interpolant.sample_rho1
        self.base     = self.exact_interpolant.sample_rho0
        self.log_rho0 = self.exact_interpolant.log_rho0

        ## set up the networks
        self.interpolant = stochastic_interpolant.Interpolant(
                path=self.path, gamma_type=self.gamma_type
            )

        # TODO: right now just v and s. should be generalized to b and s.
        self.v = itf.fabrics.make_fc_net(
                hidden_sizes=self.hidden_sizes, in_size=self.in_size,
                out_size=self.out_size, inner_act=self.inner_act,
                final_act=self.final_act
            )

        self.s = itf.fabrics.make_fc_net(
                hidden_sizes=self.hidden_sizes, in_size=self.in_size,
                out_size=self.out_size, inner_act=self.inner_act,
                final_act=self.final_act
            )

        self.opt = torch.optim.Adam(
                [*self.v.parameters(), *self.s.parameters()], lr=self.base_lr
            )

        self.sched = torch.optim.lr_scheduler.StepLR(
                optimizer=self.opt, step_size=1500, gamma=0.4
            )


        ## set up storage
        self.data_dict = {
            'losses': [],
            'v_losses': [],
            's_losses': [],
            'v_grads': [],
            's_grads': [],
            'times': [],
            'logps_pflow': [],
            'logps_sdeflow': [],
            'kl_pflow': [],
            'kl_sdeflow': [],
            'v_vhat_diff': [],
            's_shat_diff': []
        }


    def compute_likelihoods(self):
        """Draw samples from the probability flow and SDE 
        models, and compute corresponding likelihoods."""

        sde_flow = stochastic_interpolant.SDEIntegrator(
            v=self.v, s=self.s, dt=torch.tensor(1e-2), 
            eps=self.eps, interpolant=self.interpolant, 
            n_save=self.n_save, n_likelihood=self.n_likelihood
        )

        pflow = stochastic_interpolant.PFlowIntegrator(
            v=self.v, s=self.s, method='dopri5', 
            interpolant=self.interpolant, n_step=3
        )

        with torch.no_grad():
            x0_tests = self.base(self.bs)
            xfs_sde = sde_flow.rollout_forward(x0_tests) # [n_save, bs, dim]
            xf_sde = grab(xfs_sde[-1].squeeze())         # [bs, dim]

            # ([n_likelihood, bs, dim], [bs])
            x0s_sdeflow, dlogps_sdeflow = sde_flow.rollout_likelihood(xfs_sde[-1])
            log_p0s = torch.reshape(
                self.log_rho0(x0s_sdeflow.reshape((self.n_likelihood*self.bs, self.ndim))),
                (self.n_likelihood, self.bs)
            )
            logpx_sdeflow = torch.mean(log_p0s, axis=0) - dlogps_sdeflow


        logp0 = self.log_rho0(x0_tests) # [bs]
        xfs_pflow, dlogp_pflow = pflow.rollout(x0_tests) # [n_save, bs, dim], [n_save, bs]
        logpx_pflow = logp0 + dlogp_pflow[-1].squeeze()  # [bs]
        xf_pflow = grab(xfs_pflow[-1].squeeze())         # [bs, dim]


        return xf_sde, logpx_sdeflow, xf_pflow, logpx_pflow


    def log_metrics(
        self,
        v_loss: torch.tensor,
        s_loss: torch.tensor,
        loss: torch.tensor,
        v_grad: torch.tensor,
        s_grad: torch.tensor,
    ) -> None:
        # log loss and gradient data
        v_loss = grab(v_loss).mean(); self.data_dict['v_losses'].append(v_loss)
        s_loss = grab(s_loss).mean(); self.data_dict['s_losses'].append(s_loss)
        loss   = grab(loss).mean();   self.data_dict['losses'].append(loss)
        v_grad = grab(v_grad).mean(); self.data_dict['v_grads'].append(v_grad)
        s_grad = grab(s_grad).mean(); self.data_dict['s_grads'].append(s_grad)

        # compute and log likelihood data
        _, logpx_sdeflow, _, logpx_pflow = self.compute_likelihoods(
            self.v, self.s, self.log_rho0, 
            self.interpolant, self.n_save, self.n_likelihood, 
            self.eps, self.likelihood_bs
        )

        # compute kl and log data
        kl_pflow, kl_sdeflow = self.compute_kl(
            self.v, self.s, self.exact_interpolant, 
            self.interpolant, self.eps, self.likelihood_bs
        )
        kl_pflow = grab(kl_pflow).mean(); self.data_dict['kl_pflow'].append(kl_pflow)
        kl_sdeflow = grab(kl_sdeflow).mean(); self.data_dict['kl_sdeflow'].append(kl_sdeflow)

        # compute norm of difference between exact v,s and model v-hat, s-hat
        v_vhat_diff, s_shat_diff = self.compute_v_diff(
            self.v, self.s, self.target, self.exact_interpolant, self.likelihood_bs
        )
        v_vhat_diff = grab(v_vhat_diff).mean(); self.data_dict['v_vhat_diff'].append(v_vhat_diff)
        s_shat_diff = grab(s_shat_diff).mean(); self.data_dict['s_shat_diff'].append(s_shat_diff)

        logpx_sdeflow = grab(logpx_sdeflow).mean(); self.data_dict['logps_sdeflow'].append(logpx_sdeflow)
        logpx_pflow = grab(logpx_pflow).mean(); self.data_dict['logps_pflow'].append(logpx_pflow)


    # TODO: add wandb uploading here.
    def make_plots(self) -> None:
        """Make plots to visualize samples and evolution of the likelihood."""
        # compute likelihood and samples for SDE and probability flow.
        xf_sde, logpx_sdeflow, xf_pflow, logpx_pflow = self.compute_likelihoods(
            self.v, self.s, self.log_rho0, self.interpolant, 
            self.n_save, self.n_likelihood, self.eps, self.likelihood_bs
        )

        ### plot the loss, test logp, and samples from interpolant flow
        fig, axes = plt.subplots(1, 5, figsize=(19,4))

        # plot loss over time.
        nsaves = len(self.data_dict['losses'])
        epochs = np.arange(nsaves)*self.metrics_freq
        axes[0].plot(epochs, self.data_dict['losses'],   label="v + s")
        axes[0].plot(epochs, self.data_dict['v_losses'], label="v")
        axes[0].plot(epochs, self.data_dict['s_losses'], label="s" )
        axes[0].set_xlabel("Epoch", fontsize = 12)
        axes[0].set_title("LOSS")
        axes[0].legend()


        # plot samples from SDE.
        axes[1].scatter(
            xf_sde[:,0], xf_sde[:,1], vmin=0.0, vmax=0.05, 
            alpha = 0.2, c=grab(torch.exp(logpx_sdeflow).detach())
        )
        axes[1].set_xlim(-10,10)
        axes[1].set_ylim(-10,10)
        axes[1].set_title("Cross-Section from SDE", fontsize=14)
        axes[1].set_xlabel(r"$x$, $d=0$", fontsize = 12)
        axes[1].set_ylabel(r"$x$, $d=1$", fontsize = 12)


        # plot samples from pflow
        axes[2].scatter(
            xf_pflow[:,0], xf_pflow[:,1], vmin=0.0, vmax=0.05, 
            alpha = 0.2, c=grab(torch.exp(logpx_pflow).detach())
        )
        axes[2].set_xlim(-10,10)
        axes[2].set_ylim(-10,10)
        axes[2].set_title("Cross-Section from PFlow", fontsize=14)
        axes[2].set_xlabel(r"$x$, $d=0$", fontsize = 12)
        axes[2].set_ylabel(r"$x$, $d=1$", fontsize = 12)


        # plot likelihood estimates.
        axes[3].plot(epochs, self.data_dict['kl_pflow'],   label='pflow', color='purple')
        axes[3].plot(epochs, self.data_dict['kl_sdeflow'], label='sde',   color='red')
        axes[3].set_title(r"$KL(\rho_1(x) | \hat\rho(1,x) )$")
        axes[3].legend(loc='best')
        ymax = max(self.data_dict['kl_pflow'])
        axes[3].set_yscale("log")
        axes[3].set_xlabel("Epoch", fontsize = 12)


        axes[4].plot(epochs, self.data_dict['v_vhat_diff'], label=r'$o = v$',  color='purple')
        axes[4].plot(epochs, self.data_dict['s_shat_diff'], label=r'$ o = s$', color='red')
        axes[4].set_title(r"$\int_0^1 dt |o(t,x) - \hat o(t,x)|^2 \rho_t(x)$", fontsize = 14)
        axes[4].legend(loc='best')
        ymax = max(self.data_dict['v_vhat_diff'])
        axes[4].set_yscale("log")
        axes[4].set_xlabel("Epoch", fontsize = 12)


        title_str = rf'{self.ndim}-dimensional GMM. $\epsilon={self.eps}$.'
        fig.suptitle(title_str)
        fig.tight_layout()


    def train_step(self) -> None:
        """Take a single step of optimization on the training set."""
        self.opt.zero_grad()

        # construct batch
        x0s = self.base(self.prior_bs)
        x1s = self.target(self.target_bs)
        ts  = torch.rand(size=(self.N_t,))

        # compute the loss
        loss_start = time.perf_counter()
        loss_val, (loss_v, loss_s) = stochastic_interpolant.loss_sv(
            self.v, self.s, x0s, x1s, ts, self.interpolant, loss_fac=self.loss_fac
        )
        loss_end = time.perf_counter()

        # compute the gradient
        loss_val.backward()
        v_grad = torch.tensor([torch.nn.utils.clip_grad_norm_(self.v.parameters(), self.clip)])
        s_grad = torch.tensor([torch.nn.utils.clip_grad_norm_(self.s.parameters(), self.clip)])

        # perform the update.
        self.opt.step()
        self.sched.step()

        return loss_val.detach(), loss_v.detach(), loss_s.detach(), v_grad.detach(), s_grad.detach()


    def compute_kl(self) -> Tuple[torch.tensor, torch.tensor]:
        """Compute the KL divergence from the target to the model,
        either produced with the ODE or the SDE."""
        
        sde_flow = stochastic_interpolant.SDEIntegrator(
            v=self.v, s=self.s, dt=self.dt, eps=self.eps, 
            interpolant=self.interpolant, n_save=1, n_likelihood=self.n_likelihood
        )

        pflow = stochastic_interpolant.PFlowIntegrator(
            v=self.v, s=self.s, method='dopri5', interpolant=self.interpolant, n_step=3
        )
        
        x1s      = self.exact_interpolant.sample_rho1(self.bs)
        log_rho1 = self.exact_interpolant.log_rho1(x1s)
        
        x0s_pflow, dlogp_pflow = pflow.rollout(x1s, reverse=True)                # [n_save, bs, dim], [n_save, bs]
        x0_pflow               = grab(x0s_pflow[-1].squeeze())                   # [bs, dim]
        logp0                  = self.exact_interpolant.log_rho0(x0s_pflow[-1])  # [bs]
        log_rho1_hat_ode       = logp0 - dlogp_pflow[-1].squeeze()               # [bs]
        
        
        # ([n_likelihood, bs, dim], [bs])
        with torch.no_grad():
            x0s_sdeflow, dlogps_sdeflow = sde_flow.rollout_likelihood(x1s)
            log_p0s = torch.reshape(
                self.exact_interpolant.log_rho0(x0s_sdeflow.reshape((self.n_likelihood*self.bs, self.ndim))),
                (self.n_likelihood, self.bs)
            )
            log_rho1_hat_sde = torch.mean(log_p0s, axis=0) - dlogps_sdeflow

        return (log_rho1 - log_rho1_hat_ode).mean(), (log_rho1 - log_rho1_hat_sde).mean()


    def compute_v_diff(self) -> Tuple[float, float]:
        """Compute the L2 error in both v and s, as appears in the KL bound."""
        x1s = self.target(self.bs)
        x0s = self.base(self.bs)
        ts = torch.rand(size=(self.bs,))
        
        xts = (x0s*(1-ts[:,None]) + x1s*ts[:,None])
        v_exact, s_exact = self.exact_interpolant.get_velocities()
        v_exact, s_exact = itf.fabrics.InputWrapper(v_exact), itf.fabrics.InputWrapper(s_exact)
        vs = v_exact(xts, ts)
        ss = s_exact(xts, ts)
        vs_hat = self.v(xts.float(), ts.float())
        ss_hat = self.s(xts.float(), ts.float())

        return (torch.abs(vs - vs_hat)**2).sum() / torch.sum(vs**2), \
                (torch.abs(ss - ss_hat)**2).sum() / torch.sum(ss**2)


    # TODO: add training loop here.
    # TODO: add finite-data training loop here.
