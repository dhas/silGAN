import numpy as np
import torch

# def test_fn0(X12):
# 	X12 = X12.cuda()
# 	m = torch.mean(X12[:, 0, 300:512])
# 	if 0.49 <= m:
# 		ret1 = 1
# 	else:
# 		ret1 = 0
# 	return torch.stack([torch.tensor([ret1])])


# def test_fn1(X12):
# 	X12 = X12.cuda()
# 	m = torch.mean(X12[:, 0, 300:512])
# 	if 0.49 <= m <= 0.51:
# 		ret1 = 1
# 	else:
# 		ret1 = 0
# 	return torch.stack([torch.tensor([ret1])])
		

# def test_fn2(X12):
# 	X12 = X12.cuda()
# 	m1 = torch.mean(X12[:, 0, 256:350])
# 	m2 = torch.mean(X12[:, 0, 350:512])

# 	if 0.49 <= m1 <= 0.51:
# 		ret1 = 1
# 	else:
# 		ret1 = 0
	
# 	if 0.59 <= m2 <= 0.61:
# 		ret2 = 1
# 	else:
# 		ret2 = 0

# 	return torch.stack([torch.tensor([ret1]), torch.tensor([ret2])])

def test_fn4(X12):
	X12 = X12.cuda()
	m1 = torch.mean(X12[0, 256:350])
	m2 = torch.mean(X12[0, 350:512])
	m3 = torch.mean(X12[1, 0:256])
	if (0.49 <= m1 <= 0.51) and (0.59 <= m2 <= 0.61) and (m3 <= 0.08):
		ret1 = 1
	else:
		ret1 = 0

	return torch.stack([torch.tensor([ret1])])

# def test_fn3(X12):
# 	m = torch.mean(X12[0, 300:512])	
# 	if not (m < 0.49):
# 		ret1 = 1
# 	else:
# 		ret1 = 0
# 	return torch.stack([torch.tensor([ret1])])


# def test_fn4(X12):
# 	m1 = torch.mean(X12[0, 300:450])
# 	m2 = torch.mean(X12[0, 450:512])		

# 	if (m1 > 0.3) or (m2 > 0.5):
# 		ret1 = 1
# 	else:
# 		ret1 = 0
# 	return torch.stack([torch.tensor([ret1])])