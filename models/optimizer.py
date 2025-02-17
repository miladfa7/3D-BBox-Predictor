import torch

optimizers = {
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW
}

def build_optimizer(optimizer_name, parms, config):
    optimizer = optimizers[optimizer_name]
    return optimizer(parms, **config)
