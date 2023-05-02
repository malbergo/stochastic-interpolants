import torch
from functorch import vmap
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any, Callable

import interflow as itf
import interflow.prior as prior
import interflow.fabrics
import interflow.stochastic_interpolant as stochastic_interpolant
import interflow.gmm as gmm
from interflow.util import grab
from copy import deepcopy, copy
import time
import wandb
import dill as pickle
from tqdm.auto import tqdm as tqdm


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
    N1: int
    scale: float
    scale_fac: float
    gamma_type: str
    path: str
    device: str

    ## network parameters
    base_lrs: list
    hidden_sizes: list
    in_size: int
    out_size: int
    inner_act: str
    final_act: str

    ## optimization parameters
    N_epochs: int
    N_t: int
    metrics_freq: int
    plot_freq: int
    save_freq: int
    loss_fac: float
    online_learning: bool
    ndata: int
    t0_opt: float
    tf_opt: float

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
    output_location: str


    def __post_init__(self) -> None:
        """Initialize the simulation."""
        ## set up storage
        self.data_dict = {
            'losses': [],
            'bv_losses': [],
            'seta_losses': [],
            'bv_grads': [],
            'seta_grads': [],
            'logps_pflow': [],
            'logps_sdeflow': [],
            'kl_pflow': [],
            'kl_sdeflow': [],
            'bv_errs': [],
            'bv_rel_errs': [],
            'seta_errs': [],
            'seta_rel_errs': []
        }

        self.params_dict = {
            'bv_params': [], 
            'seta_params': []
        }




        ## setup everything ese
        self.setup_gmm()
        self.setup_interpolant()

        if self.online_learning:
            self.n_batches = 1
        else:
            self.dataset   = self.target(self.ndata)
            self.n_batches = self.ndata // self.target_bs
            self.N_epochs  = self.N_epochs // self.n_batches

        self.setup_networks_and_loss()
        self.setup_optimizer()
        self.setup_wandb()


    def setup_gmm(self):
        """Define the base and the target."""
        # set up base
        self.N0   = 1
        self.p0s  = torch.ones(self.N0)
        self.mu0s = torch.zeros(self.ndim).reshape((self.N0, self.ndim))
        self.C0s  = torch.eye(self.ndim).reshape((self.N0, self.ndim, self.ndim))
        
        # set up target
        self.p1s  = torch.ones(self.N1) / self.N1
        
        ## hard-coded single gaussian with uneven covariance
#         self.C1s  = torch.eye(self.ndim).reshape((self.N0, self.ndim, self.ndim))
#         self.C1s[0, 0, 0] = 3.0


        ## random mixture
        self.setup_covs()
        self.setup_means()
        
        
        ## swap experiment
#        tmp_ps    = copy(self.p1s)
#        tmp_mus   = copy(self.mu1s)
#        tmp_Cs    = copy(self.C1s)
#        
#        self.p1s  = copy(self.p0s)
#        self.mu1s = copy(self.mu0s)
#        self.C1s  = copy(self.C0s)
#        
#        self.p0s  = copy(tmp_ps)
#        self.mu0s = copy(tmp_mus)
#        self.C0s  = copy(tmp_Cs)


    def setup_covs(self):
        """Define covariances of the target."""
        self.C1s = torch.zeros(self.N1, self.ndim, self.ndim)
        for ii in range(self.N1):
            C = torch.randn(self.ndim, self.ndim)
            self.C1s[ii] = C.T @ C / self.scale_fac + torch.eye(self.ndim)


    def setup_means(self) -> None:
        """Define means of the target."""
        self.mu1s = self.scale*torch.randn((self.N1, self.ndim))


    def setup_interpolant(self) -> None:
        """Set up the interpolant, the exact interpolant, and samplers."""
        print(f'Setting up interpolant! path={self.path}')
        self.exact_interpolant = gmm.GMMInterpolant(
            self.p0s, self.p1s, self.mu0s, self.mu1s, self.C0s,
            self.C1s, self.path, self.gamma_type, device=self.device,
            use_preconditioner=True
        )


        print(f'Set up the exact interpolant. path={self.exact_interpolant.path}')
        self.target   = self.exact_interpolant.sample_rho1
        self.base     = self.exact_interpolant.sample_rho0
        self.log_rho0 = self.exact_interpolant.log_rho0
        self.interpolant = stochastic_interpolant.Interpolant(
                path=self.path, gamma_type=self.gamma_type
            )
        print(f'Set up the interpolant. path={self.interpolant.path}')


        # save the interpolants
        self.interpolants = {'exact': self.exact_interpolant, 'interpolant': self.interpolant}



    def setup_networks_and_loss(self) -> None:
        """Define the networks and the loss function."""
        self.bv = itf.fabrics.make_fc_net(
                hidden_sizes=self.hidden_sizes, in_size=self.in_size,
                out_size=self.out_size, inner_act=self.inner_act,
                final_act=self.final_act
            )

        self.seta = itf.fabrics.make_fc_net(
                hidden_sizes=self.hidden_sizes, in_size=self.in_size,
                out_size=self.out_size, inner_act=self.inner_act,
                final_act=self.final_act
            )
        
        bv_loss   = stochastic_interpolant.loss_b   if self.learn_b   else stochastic_interpolant.loss_v
        seta_loss = stochastic_interpolant.loss_eta if self.learn_eta else stochastic_interpolant.loss_s
        self.loss = stochastic_interpolant.joint_loss(bv_loss, seta_loss, self.interpolant, self.loss_fac)
        
        self.t0 = 4*self.dt if self.learn_eta else 0.0
        self.tf = (1.0 - 4*self.dt) if self.learn_eta else 1.0
        

    def setup_optimizer(self) -> None:
        """Define the optimizer and the learning rate schedule."""
        self.opts = [
            torch.optim.Adam(self.bv.parameters(),   lr=self.base_lrs[0]),
            torch.optim.Adam(self.seta.parameters(), lr=self.base_lrs[1])
        ]
        
        self.scheds = [
            torch.optim.lr_scheduler.StepLR(optimizer=self.opts[0], step_size=1500, gamma=0.4),
            torch.optim.lr_scheduler.StepLR(optimizer=self.opts[1], step_size=1500, gamma=0.4)
        ]

    def setup_wandb(self) -> None:
        """Initialize the wandb run and upload all relevant experiment information."""

        self.config = {
            'learn_b': self.learn_b,
            'learn_eta': self.learn_eta,
            'ndim': self.ndim,
            'N0': self.N0,
            'N1': self.N1,
            'scale': self.scale,
            'scale_fac': self.scale_fac,
            'gamma_type': self.gamma_type,
            'path': self.path,
            'base_lrs': self.base_lrs,
            'hidden_sizes': self.hidden_sizes,
            'in_size': self.in_size,
            'out_size': self.out_size,
            'inner_act': self.inner_act,
            'final_act': self.final_act,
            'output_location': self.output_location,
            'N_epochs': self.N_epochs,
            'n_batches': self.n_batches,
            'N_t': self.N_t,
            'metrics_freq': self.metrics_freq,
            'plot_freq': self.plot_freq,
            'save_freq': self.save_freq,
            'loss_fac': self.loss_fac,
            'online_learning': self.online_learning,
            'ndata': self.ndata,
            'n_likelihood': self.n_likelihood,
            'plot_bs': self.plot_bs,
            'prior_bs': self.prior_bs,
            'target_bs': self.target_bs,
            'likelihood_bs': self.likelihood_bs,
            'dt': self.dt,
            'eps': self.eps
        }

        wandb.init(
            project='interpolant_gmms',
            name=self.wandb_name,
            config=self.config,
        )


    def get_b_and_s(self) -> Tuple:
        """Construct b and s if learning other objects."""
        # set up score model
        if self.learn_eta:
            s = stochastic_interpolant.SFromEta(self.seta, self.interpolant.gamma)
        else:
            s = self.seta
            
        # set up b model
        if self.learn_b:
            b = self.bv
        else:
            b = stochastic_interpolant.BFromVS(self.bv, s, self.interpolant.gg_dot)
            
        return b, s
    
    
    def draw_samples(self):
        """Draw samples from the probability flow and SDE models, and compute corresponding likelihoods."""
        b, s = self.get_b_and_s()

        sde = stochastic_interpolant.SDEIntegrator(
            b=b, s=s, dt=self.dt, eps=self.eps,
            interpolant=self.interpolant, n_save=1,
            n_likelihood=self.n_likelihood
        )

        ode = stochastic_interpolant.PFlowIntegrator(
            b=b, method='rk4', interpolant=self.interpolant,
            n_step=int((self.tf - self.t0) / self.dt), sample_only=True
        )

        with torch.no_grad():
            x0_tests = self.base(self.plot_bs)
            xfs_sde  = sde.rollout_forward(x0_tests, t0=self.t0, tf=self.tf) # [1, bs, dim]
            xfs_ode  = ode.rollout(x0_tests, t0=self.t0, tf=self.tf)[0]      # [1, bs, dim], [1, bs]
            xf_sde   = grab(xfs_sde[-1].squeeze())                           # [bs, dim]
            xf_ode   = grab(xfs_ode[-1].squeeze())


        return xf_sde, xf_ode


    def compute_likelihoods(self):
        """Draw samples from the probability flow and SDE 
        models, and compute corresponding likelihoods."""
        b, s           = self.get_b_and_s()
        
        sde = stochastic_interpolant.SDEIntegrator(
            b=b, s=s, dt=self.dt, eps=self.eps,
            interpolant=self.interpolant, n_save=1,
            n_likelihood=self.n_likelihood
        )

        ode = stochastic_interpolant.PFlowIntegrator(
            b=b, method='rk4', interpolant=self.interpolant,
            n_step=int((self.tf - self.t0) / self.dt), sample_only=True
        )

        with torch.no_grad():
            # ([n_likelihood, bs, dim], [bs])
            x0s_sdeflow, dlogps_sdeflow = sde.rollout_likelihood(xfs_sde[-1], t0=self.t0, tf=self.tf)
            log_p0s = torch.reshape(
                self.log_rho0(x0s_sdeflow.reshape((self.n_likelihood*self.prior_bs, self.ndim))),
                (self.n_likelihood, self.plot_bs)
            )
            logpx_sdeflow = torch.mean(log_p0s, axis=0) - dlogps_sdeflow


        logp0 = self.log_rho0(x0_tests) # [bs]
        xfs_pflow, dlogp_pflow = ode.rollout(x0_tests, t0=self.t0, tf=self.tf) # [1, bs, dim], [1, bs]
        logpx_pflow = logp0 + dlogp_pflow[-1].squeeze()                          # [bs]
        xf_pflow = grab(xfs_pflow[-1].squeeze())                                 # [bs, dim]


        return xf_sde, logpx_sdeflow, xf_pflow, logpx_pflow


    def log_metrics(
        self,
        bv_loss: torch.tensor,
        seta_loss: torch.tensor,
        loss: torch.tensor,
        bv_grad: torch.tensor,
        seta_grad: torch.tensor,
    ) -> None:
        # save loss and gradient data
        bv_loss   = grab(bv_loss).mean();   #self.data_dict['bv_losses'].append(bv_loss)
        seta_loss = grab(seta_loss).mean(); #self.data_dict['seta_losses'].append(seta_loss)
        loss      = grab(loss).mean();      #self.data_dict['losses'].append(loss)
        bv_grad   = grab(bv_grad).mean();   #self.data_dict['bv_grads'].append(bv_grad)
        seta_grad = grab(seta_grad).mean(); #self.data_dict['seta_grads'].append(seta_grad)


#        # compute and log likelihood data
#        _, logpx_sdeflow, _, logpx_pflow = self.compute_likelihoods()
#
#
#        # compute kl and log data
#        kl_pflow, kl_sdeflow = self.compute_kl()
#        kl_pflow   = grab(kl_pflow).mean(); self.data_dict['kl_pflow'].append(kl_pflow)
#        kl_sdeflow = grab(kl_sdeflow).mean(); self.data_dict['kl_sdeflow'].append(kl_sdeflow)


        # compute norm of difference between exact v,s and model v-hat, s-hat
        bv_err, bv_rel_err, seta_err, seta_rel_err = self.compute_l2_errors()
#        bv_err       = grab(bv_err);       self.data_dict['bv_errs'].append(bv_err)
#        bv_rel_err   = grab(bv_rel_err);   self.data_dict['bv_rel_errs'].append(bv_rel_err)
#        seta_err     = grab(seta_err);     self.data_dict['seta_errs'].append(seta_err)
#        seta_rel_err = grab(seta_rel_err); self.data_dict['seta_rel_errs'].append(seta_rel_err)
        
#        logpx_sdeflow = grab(logpx_sdeflow).mean(); self.data_dict['logps_sdeflow'].append(logpx_sdeflow)
#        logpx_pflow   = grab(logpx_pflow).mean();   self.data_dict['logps_pflow'].append(logpx_pflow)

        print(f'bv_err:   {bv_err},        {bv_rel_err}')
        print(f'seta_err: {seta_err},      {seta_rel_err}')
        print()
        
        # upload results to wandb.
        wandb.log({
                'losses': {'loss': loss, 'bv_loss': bv_loss, 'seta_loss': seta_loss},
                'grads':  {'bv_grad': bv_grad, 'seta_grad': seta_grad},
                'errors': {'bv_err': bv_err, 
                           'bv_rel_err': bv_rel_err, 
                           'seta_err': seta_err, 
                           'seta_rel_err': seta_rel_err}
            }
        )


    def make_plots(self) -> None:
        """Make plots to visualize samples and evolution of the likelihood."""
        # compute likelihood and samples for SDE and probability flow.
        xf_sde, xf_ode = self.draw_samples()
        target_samps = grab(self.target(self.plot_bs))

        # plot the loss, test logp, and samples from interpolant flow
        plt.close('all')
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), constrained_layout=True)

        # plot samples from SDE.
        axs[0].scatter(xf_sde[:,0], xf_sde[:,1], alpha=0.25)
        axs[0].set_xlim(-4*self.scale, 4*self.scale)
        axs[0].set_ylim(-4*self.scale, 4*self.scale)
        axs[0].set_title("SDE", fontsize=14)
        axs[0].set_xlabel(r"$x$, $d=0$", fontsize = 12)
        axs[0].set_ylabel(r"$x$, $d=1$", fontsize = 12)

        
        # plot samples from pflow
        axs[1].scatter(xf_ode[:,0], xf_ode[:,1], alpha = 0.25)
        axs[1].set_xlim(-4*self.scale, 4*self.scale)
        axs[1].set_ylim(-4*self.scale, 4*self.scale)
        axs[1].set_title("ODE", fontsize=14)
        axs[1].set_xlabel(r"$x$, $d=0$", fontsize = 12)
        axs[1].set_ylabel(r"$x$, $d=1$", fontsize = 12)
        
        
        # plot target
        axs[2].scatter(target_samps[:,0], target_samps[:,1], alpha = 0.025)
        axs[2].set_xlim(-4*self.scale, 4*self.scale)
        axs[2].set_ylim(-4*self.scale, 4*self.scale)
        axs[2].set_title("Target", fontsize=14)
        axs[2].set_xlabel(r"$x$, $d=0$", fontsize = 12)
        axs[2].set_ylabel(r"$x$, $d=1$", fontsize = 12)

        
        wandb.log({'samples': wandb.Image(fig)})


    def compute_kl(self) -> Tuple[torch.tensor, torch.tensor]:
        """Compute the KL divergence from the target to the model,
        either produced with the ODE or the SDE."""
        b, s = self.get_b_and_s()
        
        sde_flow = stochastic_interpolant.SDEIntegrator(
            b=b, s=s, dt=self.dt, eps=self.eps, 
            interpolant=self.interpolant, n_save=1, 
            n_likelihood=self.n_likelihood
        )

        pflow = stochastic_interpolant.PFlowIntegrator(
            b=b, method='rk4', interpolant=self.interpolant, n_step=int((self.tf-self.t0) / self.dt)
        )

        x1s      = self.exact_interpolant.sample_rho1(self.likelihood_bs)
        log_rho1 = self.exact_interpolant.log_rho1(x1s)
        
        x0s_pflow, dlogp_pflow = pflow.rollout(x1s, t0=self.t0, tf=self.tf, reverse=True)               # [1, bs, dim], [1, bs]
        x0_pflow               = grab(x0s_pflow[-1].squeeze())                  # [bs, dim]
        logp0                  = self.exact_interpolant.log_rho0(x0s_pflow[-1]) # [bs]
        log_rho1_hat_ode       = logp0 - dlogp_pflow[-1].squeeze()              # [bs]


        # ([n_likelihood, bs, dim], [bs])
        with torch.no_grad():
            x0s_sdeflow, dlogps_sdeflow = sde_flow.rollout_likelihood(x1s, t0=self.t0, tf=self.tf)
            log_p0s = torch.reshape(
                self.exact_interpolant.log_rho0(x0s_sdeflow.reshape((self.n_likelihood*self.likelihood_bs, self.ndim))),
                (self.n_likelihood, self.likelihood_bs)
            )
            log_rho1_hat_sde = torch.mean(log_p0s, axis=0) - dlogps_sdeflow

        return (log_rho1 - log_rho1_hat_ode).mean(), (log_rho1 - log_rho1_hat_sde).mean()


    def compute_l2_errors(self) -> Tuple[float, float, float, float]:
        """Compute the (relative) l2 error."""
        # draw samples
        x1s = self.target(self.likelihood_bs)
        x0s = self.base(self.likelihood_bs)
        ts  = torch.rand(size=(self.likelihood_bs,))
        xts = vmap(
            lambda t, x0, x1: self.interpolant.calc_xt(t, x0, x1)[0], 
            randomness='different'
        )(ts, x0s, x1s)
        
        
        # compute exact value
        v_exact, s_exact = self.exact_interpolant.get_velocities()
        v_exact, s_exact = itf.fabrics.InputWrapper(v_exact), itf.fabrics.InputWrapper(s_exact)
        if self.learn_b:
            b_exact = stochastic_interpolant.BFromVS(v_exact, s_exact, self.exact_interpolant.gg_dot)
            bvs = b_exact(xts, ts)
        else:
            bvs = v_exact(xts, ts)

        if self.learn_eta:
            eta_exact = lambda xts, ts: -self.interpolant.gamma(ts)[:, None] * s_exact(xts, ts)
            setas = eta_exact(xts, ts)
        else:
            setas = s_exact(xts, ts)   

        # compute approximate values
        bv_hats    = self.bv(xts, ts)
        seta_hats  = self.seta(xts, ts)
        
        # compute errors and norms
        bv_hat_norm = grab(torch.sum(bv_hats**2, axis=1)).mean()
        bv_norm     = grab(torch.sum(bvs**2, axis=1)).mean()
        bv_err      = grab(torch.sum((bvs - bv_hats)**2, axis=1)).mean()
        bv_rel_err  = grab(torch.sum((bvs - bv_hats)**2, axis=1)).mean() / bv_norm
        
        seta_hat_norm = grab(torch.sum(seta_hats**2, axis=1)).mean()
        seta_norm     = grab(torch.sum(setas**2, axis=1)).mean()
        seta_err      = grab(torch.sum((setas - seta_hats)**2, axis=1)).mean()
        seta_rel_err  = grab(torch.sum((setas - seta_hats)**2, axis=1)).mean() / seta_norm
        
        return bv_err, bv_rel_err, seta_err, seta_rel_err


    def train_step(self, x1s: torch.tensor) -> None:
        """Take a single step of optimization on the training set."""
        for opt in self.opts:
            opt.zero_grad()

        # construct batch
        x0s = self.base(self.prior_bs)
        ts  = self.t0_opt + (self.tf_opt - self.t0_opt)*torch.rand(size=(self.N_t,))
        
        # compute the loss
        loss_val, (loss_bv, loss_seta) = self.loss(
            self.bv, self.seta, x0s, x1s, ts, self.loss_fac
        )

        # compute the gradient
        loss_val.backward()

        # compute the norm of the gradient for tracking
        bv_grad   = torch.tensor([torch.nn.utils.clip_grad_norm_(self.bv.parameters(),   torch.inf)])
        seta_grad = torch.tensor([torch.nn.utils.clip_grad_norm_(self.seta.parameters(), torch.inf)])

        # perform the update.
        for opt, sched in zip(self.opts, self.scheds):
            opt.step()
            sched.step()

        return loss_val.detach(), loss_bv.detach(), loss_seta.detach(), bv_grad.detach(), seta_grad.detach()


    def save_data(self) -> None:
        """Save all of the information to an output file."""
        self.params_dict['bv_params'].append(deepcopy(self.bv.state_dict()))
        self.params_dict['seta_params'].append(deepcopy(self.seta.state_dict()))
        save_dict = {
                'params': self.params_dict,
                'data': self.data_dict,
                'config': self.config,
                'interpolants': self.interpolants
            }

        pickle.dump(save_dict, open(self.output_location, 'wb'))


    def train_loop(self) -> None:
        """Perform the training."""
        counter = 0
        for curr_epoch in tqdm(range(self.N_epochs)):
            for curr_batch in range(self.n_batches):
                if self.online_learning:
                    x1s = self.target(self.target_bs)
                else:
                    lb = curr_batch*self.target_bs
                    ub = lb + self.target_bs
                    x1s = self.dataset[lb:ub]

                loss, bv_loss, seta_loss, bv_grad, seta_grad = self.train_step(x1s)

                if counter % self.metrics_freq == 0:
                    self.log_metrics(bv_loss, seta_loss, loss, bv_grad, seta_grad)

                if counter % self.plot_freq == 0:
                    self.make_plots()

                if counter % self.save_freq == 0:
                    self.save_data()

                counter+=1


        # one more for good measure
        self.save_data()
