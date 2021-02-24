import torch

def b2f_AND(e, f):    
    return torch.max(e, f)

def b2f_OR(e,f):
    return torch.min(e, f)

def b2f_NOT(e):
    return -1.0*e

def b2f_LT(e, f):
    r = (e-f)
    return r

def b2f_GT(e, f):
    r = (f-e)
    return r    

def const(c):
    return torch.tensor([c]).cuda()