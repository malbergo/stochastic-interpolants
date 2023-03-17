import numpy as np
import torch
from . import util
from . import interpolant
from . import net
import math
import hashlib
import os



class InputWrapper(torch.nn.Module):
    def __init__(self, v):
        super(InputWrapper, self).__init__()
        self.v = v
        
    def net_inp(
        self,
        t: torch.tensor,  # [1]
        x: torch.tensor   # [batch x dim]
    ) -> torch.tensor:    # [batch x (1 + dim)]
        """Concatenate time over the batch dimension."""
        # print("X SHAAAAPE:", x.shape)
        # print("T SHAAAAPE:", t.shape)
        inp = torch.cat((t.repeat(x.shape[0]).unsqueeze(1), x), dim = 1)
        return inp
    
    def forward(self, x, t):
        tx = self.net_inp(t,x)
        return self.v(tx)

def make_fc_net(hidden_sizes, in_size, out_size, inner_act, final_act, **config):
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
                
    v_net = torch.nn.Sequential(*net)
    return InputWrapper(v_net)


def make_u_net(unet_dim, dim_mult, channels, learned_sinusoidal_cond, **config):
    return net.Unet(dim = unet_dim,
                dim_mults = dim_mult,
                channels = channels,
                learned_sinusoidal_cond=learned_sinusoidal_cond 
                )


def make_interpolant(n_coeffs):
    
    return interpolant.interpolant_func(n_coeffs)



def _sample_t_beta(batch_size = 1):
    #print(util.get_torch_device())
    beta = torch.distributions.beta.Beta(1.0, 0.7)
    return beta.sample((batch_size,)).to(util.get_torch_device())

def _sample_t_unif(batch_size = 1):
    #print(util.get_torch_device())
    return torch.rand(batch_size, device=util.get_torch_device())


def _sample_t(batch_size = 1):
    #print(util.get_torch_device())
    return torch.rand(batch_size)


# def sample_Bt(batch_size =1):
#     t = _sample_t(batch_size)
#     return torch.sqrt(t*(1-t)) * torch.randn(size=batch_size)
    



def at(t):
    return (1.0-t)

def dtat(t):
    return - torch.ones(t.shape)

def bt(t):
    return t

def dtbt(t):
    return torch.ones(t.shape)

def make_loss(loss_type, **config):

    if loss_type == 'learned_interpolant':
        
        def fn(v, x0, xf, N_t, interpolant_func, **config):
            ts = _sample_t(N_t)
            n_x0 = len(x0)
            n_xf = len(xf)

            losses = torch.empty((len(ts), n_x0, n_xf))
            for i,t in enumerate(ts):
                txf = (interpolant_func.b(t))*xf
                tx0 = (interpolant_func.a(t))*x0
                xt = (tx0[:, None, :] + txf[None, :,:])
                t_xt = torch.cat([t.repeat(n_x0*n_xf).unsqueeze(1), xt.reshape(n_x0*n_xf, -1)], dim = 1)
                vtk = v(t_xt.squeeze(-1)).reshape(xt.shape)

                losses[i] = ((torch.norm(vtk, dim=-1)**2) 
                             + torch.tensor(2.0)*torch.sum(interpolant_func.dtIt(x0, xf, t) * vtk, dim=-1))

                # interpolant_func.dtIt(x0, xf, t)
            return losses.sum() / (len(ts)*n_x0*n_xf) #ts

        return fn
    
    elif loss_type == 'basic_interpolant':
        
        def fn(v, x0, xf, N_t, interpolant_func, **config):
            ts = _sample_t(N_t)
            n_x0 = len(x0)
            n_xf = len(xf)

            losses = torch.empty((len(ts), n_x0, n_xf))
            for i,t in enumerate(ts):
                xt = interpolant_func.It(t, x0, xf)
                t_xt = torch.cat([t.repeat(n_x0*n_xf).unsqueeze(1), xt.reshape(n_x0*n_xf, -1)], dim = 1)
                vtk = v(t_xt.squeeze(-1)).reshape(xt.shape)

                losses[i] = ((torch.norm(vtk, dim=-1)**2) 
                             + torch.tensor(2.0)*torch.sum(interpolant_func.dtIt(x0, xf, t) * vtk, dim=-1))


            return losses.sum() / (len(ts)*n_x0*n_xf) #ts

        return fn
    
    elif loss_type == 'noisy_interpolant':
        
        def fn(v, x0, xf, N_t, interpolant_func, **config):
            ts = _sample_t(N_t)
            n_x0 = len(x0)
            n_xf = len(xf)
            d = x0.shape[-1]

            losses = torch.empty((len(ts), n_x0, n_xf))
            for i,t in enumerate(ts):
                Bt_val = interpolant_func.Bt(t, Bt_shape = (n_x0, n_xf, d))
                xt = interpolant_func.It(t, x0, xf, Bt_val)
                t_xt = torch.cat([t.repeat(n_x0*n_xf).unsqueeze(1), xt.reshape(n_x0*n_xf, -1)], dim = 1)
                vtk = v(t_xt.squeeze(-1)).reshape(xt.shape)

                losses[i] = ((torch.norm(vtk, dim=-1)**2) 
                             - torch.tensor(2.0)*torch.sum(interpolant_func.dtIt(x0, xf, Bt_val, t) * vtk, dim=-1))


            return losses.sum() / (len(ts)*n_x0*n_xf) #ts

        return fn
    
    elif loss_type == 'interpolant_imgs':
        
        def fn(v, x0, xf, N_t, img_size, **config):
            ts = _sample_t(N_t)
            n_x0 = len(x0)
            n_xf = len(xf)

            losses = torch.empty((len(ts), n_x0, n_xf))
            for i,t in enumerate(ts):
                txf = bt(t)*xf
                tx0 = at(t)*x0

                xt = (tx0[:, None, :] + txf[None, :,:])
                t_rep = t.repeat(n_x0*n_xf)
                xt = xt.reshape(n_x0*n_xf, 1, img_size, img_size)
                vtk = v(xt, t_rep).reshape(xt.shape[0],xt.shape[1],-1)
  

                losses[i] = ((torch.norm(vtk, dim=-1)**2).reshape(n_x0,n_xf)
                             + (torch.tensor(2.0)*(torch.sum((dtat(t)*x0[:, None, :] + dtbt(t)*xf[None, :, :]) * vtk.reshape(n_x0,n_xf, img_size**2), dim=-1))))

            return losses.sum() / (len(ts)*n_x0*n_xf) #ts
        
        return fn
    elif loss_type == 'interpolant_imgsv2':
        
        def fn(v, x0, xf, N_t, img_size, **config):
            ts = _sample_t(N_t).to(x0)
            n_x0 = len(x0)
            n_xf = len(xf)
            

            losses = torch.empty((len(ts), n_x0, n_xf))
            for i,t in enumerate(ts):
                txf = bt(t)*xf
                tx0 = at(t)*x0

                xt = (tx0[:, None, :] + txf[None, :,:])
                t_rep = t.repeat(n_x0*n_xf)
                xt = xt.reshape(n_x0*n_xf, -1, img_size, img_size)
                vtk = v(xt, t_rep).reshape(xt.shape[0],xt.shape[1],-1)
  
                
                dt_xt = (dtat(t)*x0[:, None, :] + dtbt(t)*xf[None, :, :]).reshape(n_x0,n_xf, -1, img_size**2)
                losses[i] = ((torch.norm(vtk, dim=(-1,-2))**2).reshape(n_x0,n_xf)
                             + (torch.tensor(2.0)*(torch.sum(dt_xt * vtk.reshape(n_x0,n_xf, -1, img_size**2), dim=(-1,-2 ) ))))

            return losses.sum() / (len(ts)*n_x0*n_xf) #ts
        
        return fn
    
    elif loss_type == 'learned_interpolant_imgs':
        
        def fn(v, x0, xf, N_t, img_size, interpolant_func, **config):
            ts = _sample_t(N_t)
            n_x0 = len(x0)
            n_xf = len(xf)

            losses = torch.empty((len(ts), n_x0, n_xf))
            for i,t in enumerate(ts):
                txf = bt(t)*xf
                tx0 = at(t)*x0
                xt = (tx0[:, None, :] + txf[None, :,:])
                xt = xt.reshape(n_x0*n_xf, 3, img_size, img_size)
                t_rep = t.repeat(n_x0*n_xf)
                
                vtk = v(xt, t_rep).reshape(xt.shape[0],xt.shape[1],-1)
                
                dt_xt = interpolant_func.dtIt(x0, xf, t).reshape(n_x0,n_xf, -1, img_size**2)
                # dt_xt = (dtat(t)*x0[:, None, :] + dtbt(t)*xf[None, :, :]).reshape(n_x0,n_xf, -1, img_size**2)
                losses[i] = ((torch.norm(vtk, dim=(-1,-2))**2).reshape(n_x0,n_xf)
                             + torch.tensor(2.0)*torch.sum(dt_xt * vtk.reshape(n_x0,n_xf, -1, img_size**2), dim=(-1, -2)))

                # interpolant_func.dtIt(x0, xf, t)
            return losses.sum() / (len(ts)*n_x0*n_xf) #ts

        return fn
    
    elif loss_type == 'fast_interpolant':
    
        def fn(v, x0, xf, uniform_t, **config):
            
            assert len(x0) == len(xf)
            n_samp = len(x0)
            
            if uniform_t:
                ts = _sample_t_unif(n_samp)
            else:
                ts = _sample_t_beta(n_samp)

            txf = bt(ts)[..., None, None, None]*xf
            tx0 = at(ts)[..., None, None, None]*x0
            
            xt = tx0 + txf
            dt_xt = (dtat(ts)[..., None, None, None]*x0 
                     + dtbt(ts)[..., None, None, None]*xf).reshape(xt.shape[0],xt.shape[1],-1)
            
            
            vtk = v(xt, ts).reshape(xt.shape[0],xt.shape[1],-1)

            losses = ((torch.norm(vtk, dim=(1,2))**2)
                             + (torch.tensor(2.0)*(torch.sum( dt_xt * vtk, dim=(1,2)))))

            return losses.sum() / n_samp
        
        return fn
    
    elif loss_type == 'fast_interpolant_tabular':
    
        def fn(v, x0, xf, uniform_t, **config):
            
            assert len(x0) == len(xf)
            n_samp = len(x0)

            if uniform_t:
                ts = _sample_t_unif(n_samp)
            else:
                ts = _sample_t_beta(n_samp)

            txf = bt(ts)[..., None]*xf
            tx0 = at(ts)[..., None]*x0
            
            xt = tx0 + txf
            dt_xt = (dtat(ts)[..., None]*x0 
                     + dtbt(ts)[..., None]*xf).reshape(xt.shape[0],-1)
            
            t_xt = torch.cat([ts.unsqueeze(1), xt.reshape(n_samp, -1)], dim = 1)
            vtk = v(t_xt).reshape(xt.shape[0],-1)

            losses = ((torch.norm(vtk, dim=(-1))**2)
                             + (torch.tensor(2.0)*(torch.sum( dt_xt * vtk, dim=(-1)))))

            return losses.sum() / n_samp
        
        return fn
    
    else:
        raise NotImplementedError('loss type {loss_type}')



def make_activation(act):
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

    
# def make_scheduler(*, optimizer, sched_type, **sched_config):
#     if sched_type=='StepLR':
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = sched_config['step_size'], gamma = sched_config['gamma'])
#     return scheduler

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