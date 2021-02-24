import numpy as np
import torch

def test_fn(X12):
	X12 = X12.cuda()
	m1 = torch.mean(X12[0, 256:350])
	m2 = torch.mean(X12[0, 350:512])
	m3 = torch.mean(X12[1, 0:256])
	if (0.49 <= m1 <= 0.51) and (0.59 <= m2 <= 0.61) and (m3 <= 0.08):
		ret1 = 1
	else:
		ret1 = 0

	return torch.stack([torch.tensor([ret1])])