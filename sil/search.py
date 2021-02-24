#This file has been generated
import numpy as np
import torch
from . import loss

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

def test_fn4(X12):
    X12 = X12.cuda()
    m1 = torch.mean(X12[0, 256:350])
    m2 = torch.mean(X12[0, 350:512])
    m3 = torch.mean(X12[1, 0:256])
    
    c0 = b2f_AND(b2f_LT(const(0.49), m1), b2f_LT(m1,
            const(0.51)))

    c1 = b2f_AND(b2f_LT(const(0.59), m2), b2f_LT(m2,
            const(0.61)))

    c2 = b2f_LT(m3, const(0.08))

    c = b2f_AND(b2f_AND(c0, c1), c2)
    

    h0 = torch.logical_and(torch.lt(const(0.49), m1), torch.lt(m1,
        const(0.51)))

    h1 = torch.logical_and(torch.lt(const(0.59), m2), torch.lt(m2,
        const(0.61)))

    h2 = torch.lt(m3, const(0.08))

    h = torch.tensor([h0, h1, h2]).all()

    return torch.stack([c]), torch.tensor([h])
