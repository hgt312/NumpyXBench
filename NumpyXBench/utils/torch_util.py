try:
    import torch
except ImportError:
    pass

from .numpy_util import prepare_numpy_inputs


torch_type_switch = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'bool': torch.bool
}


def prepare_torch_inputs(num_input, config, grad=False, device=None):
    dtype = torch_type_switch[config['dtype']]
    inputs = prepare_numpy_inputs(num_input, config)
    inputs = [torch.tensor(i, dtype=dtype, requires_grad=grad) for i in inputs]
    return inputs
