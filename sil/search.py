#This file has been generated
import numpy as np
import torch
from . import loss


def test_fn(X12):
    X12 = X12.cuda()
    m1 = torch.mean(X12[(0), 256:350])
    m2 = torch.mean(X12[(0), 350:512])
    m3 = torch.mean(X12[(1), 0:256])
    c0 = loss.b2f_AND(loss.b2f_AND(loss.b2f_AND(loss.b2f_LT(loss.const(0.49
        ), m1), loss.b2f_LT(m1, loss.const(0.51))), loss.b2f_AND(
        loss.b2f_LT(loss.const(0.59), m2), loss.b2f_LT(m2, loss.const(0.61)
        ))), loss.b2f_LT(m3, loss.const(0.08)))
    h0 = torch.logical_and(torch.logical_and(torch.logical_and(torch.lt(
        loss.const(0.49), m1), torch.lt(m1, loss.const(0.51))),
        torch.logical_and(torch.lt(loss.const(0.59), m2), torch.lt(m2,
        loss.const(0.61)))), torch.lt(m3, loss.const(0.08)))
    return torch.stack([c0]), torch.stack([h0])
