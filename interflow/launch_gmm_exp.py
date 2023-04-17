import torch
import numpy as np
import sys
from math import sqrt
sys.path.append('.')


import gmm_exp as gmm_exp
import util as itf_utils
import interflow as itf
import argparse



if torch.cuda.is_available():
    print('CUDA available, setting default tensor residence to GPU.')
    itf_utils.set_torch_device('cuda')
    device = 'cuda'
else:
    print('No CUDA device found!')
    device = 'cpu'


## fix random seed so the GMM is the same across runs.
torch.manual_seed(42)


## fixed parameters
ndim          = 128
N1            = 5
scale         = 7.5
scale_fac     = 3*sqrt(ndim)
base_lr       = 2e-3
hidden_sizes  = [256, 256, 256, 256]
in_size       = (ndim+1)
out_size      = (ndim)
inner_act     = 'relu'
final_act     = 'none'
N_epochs      = 50000
N_t           = 25
metrics_freq  = 500
plot_freq     = 500
save_freq     = 2000
loss_fac      = 4.0
ndata         = 20000
n_likelihood  = 5
plot_bs       = 500
prior_bs      = 100
target_bs     = 200
likelihood_bs = prior_bs
dt            = torch.tensor(1e-2)
eps           = torch.tensor(0.5)


def get_simulation_parameters():
    """Process command line arguments and set up associated simulation parameters."""
    parser = argparse.ArgumentParser(description='Run a MIPS simulation from the command line.')
    parser.add_argument('--gamma_type', type=str, help='Choice of gamma.')
    parser.add_argument('--path', type=str, help='Choice of It.')
    parser.add_argument('--learn_b', type=int, help='Learn b or v.')
    parser.add_argument('--learn_eta', type=int, help='Learn eta or s.')
    parser.add_argument('--online_learning', type=int, help='Use learning?')
    parser.add_argument('--output_folder', type=str, help='Where to save.')
    parser.add_argument('--output_name', type=str, help='Where to save.')
    parser.add_argument('--wandb_name', type=str, help='WANDB run name.')
    parser.add_argument('--slurm_id', type=int, help='Slurm index.')
    return parser.parse_args()


def construct_simulation(args):
    output_location = f'{args.output_folder}/{args.output_name}_{args.slurm_id}.npy'
    sim = gmm_exp.GMMExp(
        learn_b=args.learn_b,
        learn_eta=args.learn_eta,
        ndim=ndim,
        N1=N1,
        scale=scale,
        scale_fac=scale_fac,
        gamma_type=args.gamma_type,
        path=args.path,
        device=device,
        base_lr=base_lr,
        hidden_sizes=hidden_sizes,
        in_size=in_size,
        out_size=out_size,
        inner_act=inner_act,
        final_act=final_act,
        N_epochs=N_epochs,
        N_t=N_t,
        metrics_freq=metrics_freq,
        plot_freq=plot_freq,
        save_freq=save_freq,
        loss_fac=loss_fac,
        online_learning=args.online_learning,
        ndata=ndata if not args.online_learning else torch.inf,
        n_likelihood=n_likelihood,
        plot_bs=plot_bs,
        prior_bs=prior_bs,
        target_bs=target_bs,
        likelihood_bs=likelihood_bs,
        dt=dt,
        eps=eps,
        wandb_name=args.wandb_name,
        output_location=output_location
    )
    
    return sim


if __name__ == '__main__':
    sim = construct_simulation(get_simulation_parameters())
    sim.train_loop()
