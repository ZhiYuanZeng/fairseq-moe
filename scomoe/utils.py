import torch

def distance_of_two_tensor(t1, t2):
    return torch.max(torch.abs(t1-t2))

def is_inf(tensor):
    return torch.any(torch.isinf(tensor))

def is_nan(tensor):
    return torch.any(torch.isnan(tensor))

def inverse_sort(order):
    # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
    return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))