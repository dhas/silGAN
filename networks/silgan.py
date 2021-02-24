import itertools
from torchsummary import summary
from .base import *
from .pytorch_ssim import SSIM
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

Tensor = torch.cuda.FloatTensor

# Adversarial ground truths
valid = 1
fake = 0

def lerp(val,low,high):
	return (1-val) * low + val * high

def logit(z):
	return torch.log(z/(1.0-z))

# def slerp(val, low, high):
# 	omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
# 	so = np.sin(omega)
# 	if so == 0:
# 		# L'Hopital's rule/LERP
# 		return (1.0-val) * low + val * high
# 	return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def slerp(val, low, high):
	low_norm = low/torch.norm(low, dim=1, keepdim=True)
	high_norm = high/torch.norm(high, dim=1, keepdim=True)
	omega = torch.acos((low_norm*high_norm).sum(1))
	so = torch.sin(omega)
	res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
	return res


class SilGANBase():
	def __init__(self):
		self.seq_lens = [128, 256, 512]
		self.criterion_recon = torch.nn.L1Loss().cuda()
		self.criterion_ssim  = SSIM().cuda()
	
	def as_one_hot(self, n, batch, num_classes):
		lt = torch.full((batch, 1), n, dtype=torch.int64)
		lt = torch.nn.functional.one_hot(lt, num_classes=num_classes)
		lt  = Variable(lt.type(Tensor))

		return lt
	
	def reconstruct(self, X1, X2):
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2		

		
		X1 = Variable(X1.type(Tensor))
		X2 = Variable(X2.type(Tensor))

		X11 = []
		for n in range(X1.shape[1]):
			lt = self.as_one_hot(n, X1.shape[0], X1.shape[1])			
			_X1 = torch.unsqueeze(X1[:, n, :], dim=1)	
			c1 = Enc1(_X1, lt)
			X11.append(Dec1(c1, lt).detach().cpu())
		X11 = torch.cat(X11, dim=1)

		c1, _lt, c2 = Enc2(X2, lt)
		X22 = Dec2(c1, _lt, c2)
		X22[:, 2] = torch.round(X22[:, 2]*15)/15.0

		return X11.detach().cpu(), X22.detach().cpu()

	def translate(self, X1, l, num_styles=4):
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2


		n_tmpl = X1.shape[1]
		X1 = torch.unsqueeze(X1[:, l, :], dim=1)
		X1 = Variable(X1.type(Tensor))
		lt = self.as_one_hot(l, X1.shape[0], n_tmpl)		
		c1= Enc1(X1, lt)

		X12 = []		
		for _ in range(num_styles):
			c2  = torch.randn(X1.shape[0], 1, 2*c1.shape[-1]-n_tmpl).cuda()
			X12.append(Dec2(c1, lt, c2).detach().cpu())
		X12 = torch.stack(X12, dim=0)
		X12[:, :, 2] = torch.round(X12[:, :, 2]*15)/15.0
		return X12

	def search_diverse_translations(self, X1, label, n_samples=4, search_lim=1000):
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2
		Enc3 = self.Enc3
		Dec3 = self.Dec3

		n_tmpl = X1.shape[1]
		X1 = torch.unsqueeze(X1[:, label, :], dim=1)
		X1 = Variable(X1.type(Tensor))
		lt = self.as_one_hot(label, X1.shape[0], n_tmpl)		
		c1= Enc1(X1, lt)
		C2A  = torch.randn(X1.shape[0], 1, 2*c1.shape[-1]-n_tmpl).cuda()
		X12A = Dec2(c1, lt, C2A)
		C3A = torch.randn(X1.shape[0], 1, self.exp_len-X12A.shape[-1]).cuda()
		X13A, _ = Dec3.expand(X12A, C3A, (self.exp_len-X12A.shape[-1])//2)
		X13A = X13A
		
		div_trans = 0
		X12B = None
		C2B = None
		for step in range(search_lim):
			c2  = torch.randn(X1.shape[0], 1, 2*c1.shape[-1]-n_tmpl).cuda()
			x   = Dec2(c1, lt, c2)
			div = self.criterion_recon(X12A, x)
			if div > div_trans:
				div_trans = div
				X12B = x
				C2B = c2
		print('div_trans', div_trans)

		div_exp = 0
		X13_B  = None
		C3B = None
		for step in range(search_lim):
			c3 = torch.randn(X1.shape[0], 1, self.exp_len-X12B.shape[-1]).cuda()
			x, _ = Dec3.expand(X12B, c3, (self.exp_len-X12B.shape[-1])//2)
			div = self.criterion_recon(X13A, x)
			if div > div_exp:
				div_exp = div
				X13B = x
				C3B = c3
		print('div_exp', div_exp)

		ps = np.linspace(0, 1, num=n_samples)
		X12 = []
		X13 = []
		for p in ps:
			c2 = (1-p)*C2A + p*C2B
			c3 = (1-p)*C3A + p*C3B			
			x12 = Dec2(c1, lt, c2.cuda())
			x13, _ = Dec3.expand(x12, c3.cuda(), (512-x12.shape[-1])//2)

			X12.append(x12.detach().cpu())
			X13.append(x13.detach().cpu())

		X12 = torch.cat(X12, dim=0)
		X13 = torch.cat(X13, dim=0)
		return X12, X13

		# X12 = torch.cat([X12A.detach().cpu(), X12B.detach().cpu()], dim=0)
		# X13 = torch.cat([X13A.detach().cpu(), X13B.detach().cpu()], dim=0)
		# return X12, X13




		# C2 = Variable(c2, requires_grad=True)
		# optimizer = torch.optim.Adam([C2], lr=0.001)
		# for step in range(100):
		# 		optimizer.zero_grad()				
		# 		x = Dec2(c1, lt, C2)
		# 		loss = -1*self.criterion_recon(X12_A, x)				
		# 		if step%10 == 0:
		# 			print(step, loss)				
		# 		loss.backward(retain_graph=True)
		# 		optimizer.step()

	
	def sample_simplex(self, X1, label, n_samples=16):
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2

		n_points = X1.shape[0]
		n_tmpl = X1.shape[1]
		X1 = torch.unsqueeze(X1[:, label, :], dim=1)
		X1 = Variable(X1.type(Tensor))

		lt = self.as_one_hot(label, X1.shape[0], n_tmpl)
		CS = Enc1(X1, lt).detach().cpu()

		p = np.hstack([
			np.zeros((n_samples, 1)),
			np.random.uniform(0, 1, (n_samples, n_points-1)),
			np.ones((n_samples, 1))])
		p = torch.from_numpy(np.diff(p, axis=1)).float()
		_CS = torch.stack(n_samples*[CS], dim=3)
		c1 = p*_CS.permute(1, 2, 3, 0)
		c1 = torch.sum(c1, dim=-1).permute(2, 0, 1).cuda()
		lt = self.as_one_hot(label, c1.shape[0], n_tmpl)
		c2 = torch.randn(c1.shape[0], 1, 2*c1.shape[-1]-2).cuda()
		X12 = Dec2(c1, lt, c2)
		return X12.detach().cpu()



	def sweep(self, X1, l, n_samples=10):
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2

		n_tmpl = X1.shape[1]
		X1 = torch.unsqueeze(X1[:, l, :], dim=1)
		X1 = Variable(X1.type(Tensor))
		lt = self.as_one_hot(l, X1.shape[0], n_tmpl)
		c1 = Enc1(X1, lt)

		ps = torch.linspace(0, 1, steps=n_samples)
		c1s = []
		for p in ps:
			c = (1-p)*c1[0] + p*c1[-1]
			c1s.append(c)
			# c1s.append(slerp(p, c1[0], c1[-1])) #(1-p)*c1[0] + p*c1[-1])
		c1s = torch.stack(c1s, dim=0).cuda()
		c2  = torch.randn(c1s.shape[0], 1, 2*c1.shape[-1]-n_tmpl).cuda()
		lt  = self.as_one_hot(l, c1s.shape[0], n_tmpl)
		X12 = Dec2(c1s, lt, c2)
		c1_12, _, _ = Enc2(X12, lt)
		X121 = Dec1(c1_12, lt)
		return X12.detach().cpu(), X121.detach().cpu()

	def metric_sweep(self, X1, l, metric_fn, n_samples=10, ovsmpl=100, verbose=False):		
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2

		n_points = n_samples
		n_tmpl = X1.shape[1]
		X1 = torch.unsqueeze(X1[:, l, :], dim=1)
		X1 = Variable(X1.type(Tensor))

		lt = self.as_one_hot(l, X1.shape[0], n_tmpl)
		c1 = Enc1(X1, lt)

		xs = torch.unsqueeze(X1[0], dim=0)
		cs = torch.unsqueeze(c1[0], dim=0)
		xe = torch.unsqueeze(X1[-1], dim=0)
		ce = torch.unsqueeze(c1[-1], dim=0)
		l1 = torch.unsqueeze(lt[0], dim=0)		

		ms = metric_fn(Dec1(cs, l1).detach().cpu(), xe.detach().cpu())
		me = metric_fn(Dec1(ce, l1).detach().cpu(), xe.detach().cpu())
		trajectory = torch.linspace(0, 1, steps=n_points)
		m_trajectory = torch.tensor([t*me  + (1-t)*ms for t in trajectory])
	
		z_alphas 	= torch.linspace(0, 1, steps=n_points)
		z_dists 	= torch.full((n_points,), np.inf)

		num_steps = n_points*ovsmpl
		for step in range(num_steps+1):
			if verbose:
				print('Step %d/%d' % (step,num_steps))
			alpha = 1.0 * step/num_steps
			c = slerp(alpha, cs, ce).cuda()
			x = Dec1(c, l1).detach().cpu()
			m_at_z = metric_fn(x, xe.detach().cpu())
			m_dists = torch.norm(m_trajectory - m_at_z)
			closest_m_dist  = m_dists.min()
			closest_m_index = m_dists.argmin()
			# print(m_trajectory, m_at_z, m_dists, closest_m_dist, closest_m_index)

			if closest_m_dist < z_dists[closest_m_index]:
				z_alphas[closest_m_index] 	= alpha
				z_dists[closest_m_index] 	= closest_m_dist

		z_mlerp = []
		for alpha in z_alphas:
			pos = slerp(alpha, cs, ce)				
			z_mlerp.append(pos)

		z_mlerp = torch.stack(z_mlerp, dim=1)				
		c2  = torch.randn(n_points, 1, 2*c1.shape[-1]-n_tmpl).cuda()
		lt  = self.as_one_hot(l, n_points, n_tmpl)		
		x_mlerp = Dec2(z_mlerp[0], lt, c2)
		X121 = Dec1(z_mlerp[0], lt)

		return x_mlerp.detach().cpu(), X121.detach().cpu()	

	def test_search(self, X1, label, search_fn, grad_thr=0.05, n_hits=16, num_steps=1000):
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2

		n_points = X1.shape[0]
		n_tmpl = X1.shape[1]
		X1 = torch.unsqueeze(X1[:, label, :], dim=1)
		X1 = Variable(X1.type(Tensor))

		lt = self.as_one_hot(label, X1.shape[0], n_tmpl)
		CS = Enc1(X1, lt).detach().cpu()
				
		c_loss, _ = search_fn(torch.randn(1, 3, X1.shape[-1]))
		c_loss = c_loss.detach().cpu()
		n_obj  = c_loss.shape[0]
		X_hits = torch.full((n_obj, n_hits, 3, X1.shape[-1]), np.nan)

		c_min = torch.full_like(c_loss, np.inf)
		pos = torch.full((n_obj, n_points), np.nan)
		C2  = torch.full((n_obj, 1, 2*CS.shape[-1]-2), np.nan)
		n = 0
		print('Random search')
		do_grad_search = False
		while n < num_steps:
			for obj in range(n_obj):
				p = np.hstack([
					np.zeros((128, 1)),
					np.random.uniform(0, 1, (128, n_points-1)),
					np.ones((128, 1))])
				p = torch.from_numpy(np.diff(p, axis=1)).float()
				_CS = torch.stack(128*[CS], dim=3)
				c1 = p*_CS.permute(1, 2, 3, 0)
				c1 = torch.sum(c1, dim=-1).permute(2, 0, 1).cuda()
				lt = self.as_one_hot(label, c1.shape[0], n_tmpl)
				c2 = torch.randn(c1.shape[0], 1, 2*c1.shape[-1]-2).cuda()
				X12 = Dec2(c1, lt, c2)
				sc = search_fn(X12)[0][obj]				
				sc = sc.detach().cpu()
				if sc.min() < c_min[obj]:
					c_min[obj] = sc.min()
					pos[obj] = p[sc.argmin()]
					C2[obj]  = c2[sc.argmin()]		
			n = n + 1
		print(n, c_min.detach(), pos)
		
		if True:
			print('Gradient search')			
			pos = Variable(pos.cuda(), requires_grad=True)
			CS  = torch.stack(n_obj*[CS], dim=0).cuda()
			lt = self.as_one_hot(label, n_obj, n_tmpl)
			C2 = Variable(C2.cuda(), requires_grad=True)
			optimizer = torch.optim.Adam([pos, C2], lr=0.001)
			hit_count = np.zeros(n_obj, dtype=np.int)
			for step in range(10000):
				optimizer.zero_grad()					
				c1 = pos*CS.permute(2, 3, 0, 1)
				c1 = torch.sum(c1, dim=-1).permute(2, 0, 1)
				x = Dec2(c1, lt, C2)
				loss = torch.tensor([0.]).cuda()
				for obj in range(n_obj):
					_x = torch.unsqueeze(x[obj], dim=0)
					_s = search_fn(_x)[0][obj]
					_h = search_fn(_x)[1][obj]
					loss += _s
					c_min[obj] = _s.detach().cpu()
					if _h.all():
						if hit_count[obj] < n_hits:
							X_hits[obj, hit_count[obj]] = x[obj]
						hit_count[obj] += 1	
				if step%100 == 0:
					print(step, c_min.detach().numpy(), hit_count, pos.detach().cpu().numpy())
				if (hit_count >= n_hits).all():
					print(step, c_min.detach().numpy(), hit_count, pos.detach().cpu().numpy())
					break
				loss.backward(retain_graph=True)
				optimizer.step()

		return X_hits.detach().cpu(), pos.detach().cpu().numpy()

	def test_metric_sweep(self, X1, l, test_fn, metric_fn, ovsmpl=100, verbose=False):		
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2

		n_points = X1.shape[0]
		n_tmpl = X1.shape[1]
		X1 = torch.unsqueeze(X1[:, l, :], dim=1)
		X1 = Variable(X1.type(Tensor))

		lt = self.as_one_hot(l, X1.shape[0], n_tmpl)
		c1 = Enc1(X1, lt)

		xs = torch.unsqueeze(X1[0], dim=0)
		cs = torch.unsqueeze(c1[0], dim=0)
		xe = torch.unsqueeze(X1[-1], dim=0)
		ce = torch.unsqueeze(c1[-1], dim=0)
		l1 = torch.unsqueeze(lt[0], dim=0)		

		ms = metric_fn(Dec1(cs, l1).detach().cpu(), xe.detach().cpu())
		me = metric_fn(Dec1(ce, l1).detach().cpu(), xe.detach().cpu())
		trajectory = np.linspace(0,1,num=n_points)
		m_trajectory = np.array([t*me  + (1-t)*ms for t in trajectory])

		z_alphas 	= np.linspace(0, 1, num=n_points)
		z_dists 	= np.full(n_points, np.inf)
		t_dists 	= np.full(n_points, np.inf)
		x_mlerp     = torch.full((n_points, 3, 512), np.nan)
		
		num_steps = n_points*ovsmpl
		for step in range(num_steps+1):
			if verbose:
				print('Step %d/%d' % (step,num_steps))
			alpha = 1.0 * step/num_steps
			c = slerp(alpha, cs, ce).cuda()
			d = torch.randn(1, 1, 2*c.shape[-1]-2).cuda()			
			y = Dec2(c, l1, d).detach().cpu()
			x = Dec1(c, l1).detach().cpu()
			m_at_z = metric_fn(x, xe.detach().cpu())	
			t_at_z = np.abs(test_fn(y[0]))
			m_dists = np.linalg.norm(m_trajectory - m_at_z, axis=1)
			closest_m_dist  = m_dists.min()
			closest_m_index = np.argmin(m_dists)			
			# print(m_trajectory, m_at_z, t_at_z, closest_m_dist, closest_m_index)
			if (closest_m_dist < z_dists[closest_m_index]) and (t_at_z < t_dists[closest_m_index]):
				# print('SAT')
				z_alphas[closest_m_index] 	= alpha
				z_dists[closest_m_index] 	= closest_m_dist
				t_dists[closest_m_index]    = t_at_z
				x_mlerp[closest_m_index]    = y[0]

		
		print(t_dists, t_dists.min(), np.argmin(t_dists), len(x_mlerp))		

		c1_12, lt_12, c2_12 = Enc2(x_mlerp.cuda(), lt)
		X121 = Dec1(c1_12, lt_12)
		# z_mlerp = []
		# for alpha in z_alphas:
		# 	pos = slerp(alpha, cs, ce)				
		# 	z_mlerp.append(pos)

		# z_mlerp = torch.stack(z_mlerp, dim=1)				
		# c2  = torch.randn(n_points, 1, 2*c1.shape[-1]-n_tmpl).cuda()
		# lt  = self.as_one_hot(l, n_points, n_tmpl)		
		# x_mlerp = Dec2(z_mlerp[0], lt, c2).detach().cpu()
		# X121 = Dec1(z_mlerp[0], lt)
		
		
		return x_mlerp.detach().cpu(), X121.detach().cpu(), t_dists

	# def test_sweep(self, X1, l, test_fn, n_points=10):
	# 	Enc1 = self.Enc1
	# 	Dec1 = self.Dec1
	# 	Enc2 = self.Enc2
	# 	Dec2 = self.Dec2

	# 	n_tmpl = X1.shape[1]
	# 	X1 = torch.unsqueeze(X1[:, l, :], dim=1)
	# 	X1 = Variable(X1.type(Tensor))
	# 	lt = self.as_one_hot(l, X1.shape[0], n_tmpl)
	# 	c1 = Enc1(X1, lt)

	# 	ps = torch.linspace(0, 1, steps=n_points)
	# 	c1s = []
	# 	for p in ps:			
	# 		c1s.append(slerp(p, c1[0], c1[-1])) #(1-p)*c1[0] + p*c1[-1])		
	# 	c1s = torch.stack(c1s, dim=0).cuda()
	# 	c2  = torch.randn(c1s.shape[0], 1, 2*c1.shape[-1]-n_tmpl).cuda()
	# 	lt  = self.as_one_hot(l, c1s.shape[0], n_tmpl)
	# 	X12 = Dec2(c1s, lt, c2)
	# 	c1_12, lt_12, _ = Enc2(X12, lt)
	# 	X121 = Dec1(c1_12, lt_12)
	# 	X12 = X12.detach().cpu()
	# 	X121 = X121.detach().cpu()
	# 	t_dis = []
	# 	for x in X12:
	# 		t_dis.append(test_fn(x.numpy()))
	# 	t_dis = np.stack(t_dis, axis=0)
	# 	print(t_dis)
	# 	return X12.detach().cpu(), X121.detach().cpu()

	def cycle_translate(self, X12, l):
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2		

		n_tmpl = 2 #X1.shape[1]
		X12 = Variable(X12.type(Tensor))
		lt = self.as_one_hot(l, X12.shape[1], n_tmpl)		

		X121 = []
		for _X12 in X12:
			c1_12, _, c2_12 = Enc2(_X12, lt)
			X121.append(Dec1(c1_12, lt).detach().cpu())
		X121 = torch.stack(X121, dim=0)
		return X121

	
	def expand(self, X1, ls, num_styles=4):
		if not hasattr(self, 'Dec3'):
			raise Exception('Expansion not supported')
		
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2
		Dec3 = self.Dec3

		if X1.shape[-1] >= self.exp_len:
			raise Exception('Expansion not supported')
		n_tmpl = X1.shape[1]
		X1 = torch.unsqueeze(X1[:, ls, :], dim=1)
		lt = torch.full((X1.shape[0], 1), ls, dtype=torch.int64)
		lt = torch.nn.functional.one_hot(lt, num_classes=n_tmpl)
		lt  = Variable(lt.type(Tensor))
		X1 = Variable(X1.type(Tensor))

		X13 = []
		for s in range(num_styles):
			c1 = Enc1(X1, lt)
			c2 = torch.randn(X1.shape[0], 1, 2*c1.shape[-1]-n_tmpl).cuda()				
			X12 = Dec2(c1, lt, c2)
								
			c3 = torch.randn(X1.shape[0], 1, self.exp_len-X12.shape[-1]).cuda()
			_X13, _ = Dec3.expand(X12, c3, (self.exp_len-X12.shape[-1])//2)
			X13.append(_X13.detach().cpu())
		X13 = torch.stack(X13, dim=0)
		return X13

	def new_test_search(self, X1, label, search_fn, grad_thr=0.05, n_hits=16, num_steps=1000):
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2

		n_points = X1.shape[0]
		n_tmpl = X1.shape[1]
		X1 = torch.unsqueeze(X1[:, label, :], dim=1)
		X1 = Variable(X1.type(Tensor))

		lt = self.as_one_hot(label, X1.shape[0], n_tmpl)
		CS = Enc1(X1, lt).detach().cpu()
				
		c_loss, _ = search_fn(torch.randn(3, X1.shape[-1]))
		c_loss = c_loss.detach().cpu()
		n_obj  = c_loss.shape[0]
		X_hits = torch.full((n_obj, n_hits, 3, X1.shape[-1]), np.nan)
		hit_count = np.zeros(n_obj, dtype=np.int)

		loss_landscape = []

		c_min = torch.full_like(c_loss, np.inf)
		pos = torch.full((n_obj, n_points), np.nan)
		C2  = torch.full((n_obj, 1, 2*CS.shape[-1]-2), np.nan)
		n = 0
		print('Random search')
		do_grad_search = False
		while n < num_steps:
			for obj in range(n_obj):
				p = np.hstack([
					np.zeros((128, 1)),
					np.random.uniform(0, 1, (128, n_points-1)),
					np.ones((128, 1))])
				p = np.sort(p, axis=1)
				p = torch.from_numpy(np.diff(p, axis=1)).float()
				_CS = torch.stack(128*[CS], dim=3)
				c1 = p*_CS.permute(1, 2, 3, 0)
				c1 = torch.sum(c1, dim=-1).permute(2, 0, 1).cuda()
				lt = self.as_one_hot(label, c1.shape[0], n_tmpl)
				c2 = torch.randn(c1.shape[0], 1, 2*c1.shape[-1]-2).cuda()
				X12 = Dec2(c1, lt, c2)
				sc = []
				for ind, x in enumerate(X12):
					_c, _h = search_fn(x)					
					sc.append(_c[obj].item())
					if _h[obj] == True:
						X_hits[obj, hit_count[obj]] = x.detach().cpu()
						hit_count[obj] += 1									
				sc = np.array(sc)
				ind = np.random.randint(0, len(sc), size=10)				
				loss_landscape.append(np.hstack([sc[ind].reshape(-1, 1), p[ind]]))
				if sc.min() < c_min[obj]:
					c_min[obj] = sc.min()
					pos[obj] = p[sc.argmin()]
					C2[obj]  = c2[sc.argmin()]		
			n = n + 1
			if n%10 == 0:
				print(n, c_min.detach(), hit_count, pos)
		
		print('Gradient search')
		pos = Variable(pos.cuda(), requires_grad=True)
		CS  = torch.stack(n_obj*[CS], dim=0).cuda()		
		lt = self.as_one_hot(label, n_obj, n_tmpl)
		C2 = Variable(C2.cuda(), requires_grad=True)
		optimizer = torch.optim.Adam([pos, C2], lr=0.001)		
		for step in range(10000):
			optimizer.zero_grad()
			npos = pos/torch.sum(pos)
			c1 = npos*CS.permute(2, 3, 0, 1)			
			c1 = torch.sum(c1, dim=-1).permute(2, 0, 1)
			x = Dec2(c1, lt, C2)
			loss = torch.tensor([0.]).cuda()
			for obj in range(n_obj):		
				_s = search_fn(x[obj])[0][obj]
				_h = search_fn(x[obj])[1][obj]

				loss += _s
				c_min[obj] = _s.detach().cpu()
				loss_landscape.append(np.hstack([c_min[obj], npos[obj].detach().cpu()]))
				if _h.all():
					if hit_count[obj] < n_hits:
						X_hits[obj, hit_count[obj]] = x[obj]
					hit_count[obj] += 1	
			if step%100 == 0:
				print(step, c_min.detach().numpy(), hit_count, npos.detach().cpu().numpy())
			if (hit_count >= n_hits).all():
				print(step, c_min.detach().numpy(), hit_count, npos.detach().cpu().numpy())
				break
			loss.backward(retain_graph=True)
			optimizer.step()


		return X_hits.detach().cpu(), np.vstack(loss_landscape)

	def flat_test_search(self, X1, label, search_fn, grad_thr=0.05, n_hits=16, num_steps=1000):
		Enc1 = self.Enc1
		Dec1 = self.Dec1
		Enc2 = self.Enc2
		Dec2 = self.Dec2

		n_points = X1.shape[0]
		n_tmpl = X1.shape[1]
		X1 = torch.unsqueeze(X1[:, label, :], dim=1)
		X1 = Variable(X1.type(Tensor))

		lt = self.as_one_hot(label, X1.shape[0], n_tmpl)
		CS = Enc1(X1, lt).detach().cpu()
				
		c_loss, _ = search_fn(torch.randn(3, X1.shape[-1]))
		c_loss = c_loss.detach().cpu()
		n_obj  = c_loss.shape[0]
		X_hits = torch.full((n_obj, n_hits, 3, X1.shape[-1]), np.nan)
		hit_count = np.zeros(n_obj, dtype=np.int)

		loss_landscape = []

		c_min = torch.full_like(c_loss, np.inf)
		pos = torch.full((n_obj, n_points), np.nan)
		C2  = torch.full((n_obj, 1, 2*CS.shape[-1]-2), np.nan)
		n = 0
		print('Random search')
		do_grad_search = False
		while n < num_steps:
			for obj in range(n_obj):
				p = np.hstack([
					np.zeros((1, 1)),
					np.random.uniform(0, 1, (1, n_points-1)),
					np.ones((1, 1))])
				p = np.sort(p, axis=1)
				p = torch.from_numpy(np.diff(p, axis=1)).float()

				c1 = p*CS.permute(1, 2, 0)
				c1 = torch.unsqueeze(torch.sum(c1, axis=-1), dim=0).cuda()
				lt = self.as_one_hot(label, c1.shape[0], n_tmpl)
				c2 = torch.randn(c1.shape[0], 1, 2*c1.shape[-1]-2).cuda()
				X12 = Dec2(c1, lt, c2)
				c, h = search_fn(X12[0])
				c = c.detach().cpu()
				loss_landscape.append(np.hstack([c.reshape(-1, 1), pos]))
				if h[obj] == True and hit_count[obj] < n_hits:
					X_hits[obj, hit_count[obj]] = X12[0].detach().cpu()
					hit_count[obj] += 1						
				if c[obj] < c_min[obj]:
					c_min[obj] = c
					pos[obj]   = p
					C2[obj]    = c2
			n = n + 1
			if n%1000 == 0:
				print(n, c_min.detach(), hit_count, pos)
		
		if (hit_count == n_hits).all():
			return X_hits.detach().cpu(), np.vstack(loss_landscape)

		print(n, c_min.detach(), hit_count, pos)
		print('Gradient search')
		pos = logit(pos)
		pos = Variable(pos.cuda(), requires_grad=True)
		CS  = torch.stack(n_obj*[CS], dim=0).cuda()		
		lt = self.as_one_hot(label, n_obj, n_tmpl)
		C2 = Variable(C2.cuda(), requires_grad=True)
		optimizer = torch.optim.Adam([pos, C2], lr=0.001)		
		for step in range(10000):
			optimizer.zero_grad()
			pos_unit = torch.sigmoid(pos)
			npos = pos_unit/torch.sum(pos_unit)
			c1 = npos*CS.permute(2, 3, 0, 1)			
			c1 = torch.sum(c1, dim=-1).permute(2, 0, 1)
			x = Dec2(c1, lt, C2)
			loss = torch.tensor([0.]).cuda()
			for obj in range(n_obj):		
				_s = search_fn(x[obj])[0][obj]
				_h = search_fn(x[obj])[1][obj]

				loss += _s
				c_min[obj] = _s.detach().cpu()
				loss_landscape.append(np.hstack([c_min[obj], npos[obj].detach().cpu()]))
				if _h.all():
					if hit_count[obj] < n_hits:
						X_hits[obj, hit_count[obj]] = x[obj]
					hit_count[obj] += 1	
			if step%100 == 0:
				print(step, c_min.detach().numpy(), hit_count, npos.detach().cpu().numpy())
			if (hit_count >= n_hits).all():
				print(step, c_min.detach().numpy(), hit_count, npos.detach().cpu().numpy())
				break
			loss.backward(retain_graph=True)
			optimizer.step()


		return X_hits.detach().cpu(), np.vstack(loss_landscape)