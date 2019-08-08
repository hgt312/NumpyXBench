try:
    import torch
except ImportError:
    pass


torch_type_switch = {
    'float32': torch.float32,
    'float64': torch.float64,
}


def prepare_torch_inputs(num_input, config, grad=False, device=None):
    input_shape = config['shape']
    dtype = config['dtype']
    inputs = []
    for _ in range(num_input):
        inputs.append(torch.randn(*input_shape,
                                  dtype=torch_type_switch[dtype],
                                  requires_grad=grad))
    return inputs
