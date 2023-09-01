import numpy as np
import torch
import hashlib


def grab(var):
    if hasattr(var, 'detach'):
        return var.detach().cpu().numpy()
    else:
        return 
    
    
def init_weights(layers, init_weights_type, init_weights_type_cfg):
    if init_weights_type is not None:
        initializer = getattr(torch.nn.init, init_weights_type)
        def save_init(m):
            if type(m) != torch.nn.BatchNorm3d and hasattr(m, 'weight') and m.weight is not None:
                initializer(m.weight, **init_weights_type_cfg)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias) # bias always zeros

        layers.apply(save_init)
        print("Weights initialization done:", initializer, init_weights_type_cfg)




def circ_padding_size(kernel_size):
    torch_version = version.parse(torch.__version__)
    version_150 = version.parse('1.5.0')
    if torch_version < version_150:
        return kernel_size - 1
    else: # for pytorch >= 1.5
        assert kernel_size % 2 == 1, 'circ padding only support for odd kernel size'
        return kernel_size // 2
    
    
    
def hash_model_parameters(model):
    state_dict = model.state_dict()
    hasher = hashlib.sha1()
    for sdk in sorted(state_dict.keys()): # conventional order
        to_hash = grab(state_dict[sdk]) # get as numpy array
        hasher.update(to_hash.data.tobytes()) # all data [vs str(arr)]
    return hasher.digest() # printable






# Handle default device placement of tensors and float precision
_device = 'cpu'
def set_torch_device(device):
    global _device
    _device = device
    update_torch_default_tensor_type()
def get_torch_device():
    return _device


def update_torch_default_tensor_type():
    if get_torch_device() == 'cpu':
        if get_float_dtype() == np.float32:
            torch.set_default_tensor_type(torch.FloatTensor)
        elif get_float_dtype() == np.float64:
            torch.set_default_tensor_type(torch.DoubleTensor)
        else:
            raise NotImplementedError(f'Unknown float dtype {get_float_dtype()}')
    elif get_torch_device() == 'cuda':
        if get_float_dtype() == np.float32:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif get_float_dtype() == np.float64:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            raise NotImplementedError(f'Unknown float dtype {get_float_dtype()}')
    elif get_torch_device() == 'cuda:1':
        if get_float_dtype() == np.float32:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif get_float_dtype() == np.float64:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            raise NotImplementedError(f'Unknown float dtype {get_float_dtype()}')
    elif get_torch_device() == 'xpu':
        import torch_ipex
        if get_float_dtype() == np.float32:
            torch.set_default_tensor_type(torch.xpu.FloatTensor)
        elif get_float_dtype() == np.float64:
            torch.set_default_tensor_type(torch.xpu.DoubleTensor)
        else:
            raise NotImplementedError(f'Unknown float dtype {get_float_dtype()}')
    else:
        raise NotImplementedError(f'Unknown device {get_torch_device()}')


_float_dtype = np.float32
def get_float_dtype():
    return _float_dtype
def set_float_dtype(np_dtype):
    global _float_dtype
    _float_dtype = np_dtype
    update_torch_default_tensor_type()
def get_float_torch_dtype():
    if _float_dtype == np.float32:
        return torch.float32
    elif _float_dtype == np.float64:
        return torch.float64
    else:
        raise NotImplementedError(f'unknown np dtype {_float_dtype}')
def get_complex_torch_dtype():
    if _float_dtype == np.float32:
        return torch.complex64
    elif _float_dtype == np.float64:
        return torch.complex128
    else:
        raise NotImplementedError(f'unknown np dtype {_float_dtype}')
def set_float_prec(prec):
    if prec == 'single':
        set_float_dtype(np.float32)
    elif prec == 'double':
        set_float_dtype(np.float64)
    else:
        raise NotImplementedError(f'Unknown precision type {prec}')
def init_device():
    if torch.cuda.is_available():
        set_torch_device('cuda')
        return 'cuda'
    try:
        import torch_ipex
        if torch.xpu.is_available():
            set_torch_device('xpu')
            return 'xpu'
    except:
        pass
    return 'cpu'



