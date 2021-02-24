import os
import torch
import numpy as np
import json
from argparse import Namespace
import matplotlib.pyplot as plt
from matplotlib import cm
import networks
import utils
import templates
from sil import test, search


def to_logan_template(t):
	return t.repeat(1, 2, 1)

def seed_fn(worker_id):
	torch.manual_seed(5)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(opt.seed)

def l1_similarity(X1, X2):
	return torch.mean(torch.abs(X1 - X2))

def search_test_vectors(X1, label, tag):
	search_fns = [ex1.search_fn1, ex1.search_fn2, ex1.search_fn3, ex1.search_fn4]
	test_fns   = [cut.test_fn1, cut.test_fn2, cut.test_fn3, cut.test_fn4]
	
	# search_fns = [ex1.search_fn1]
	# test_fns   = [cut.test_fn1]

	if X1.shape[0] < 3:
		X12, X121 = model.sweep(X1, label)
		utils.visualize_test_mini((X121, X12), label, 
			savepath='%s/search/epoch_%d_%s_sweep.png' % (sample_dir, epoch, tag))	

	for fn, (search_fn, test_fn) in enumerate(zip(search_fns, test_fns)):
		print('Searching tests for %s' % (test_fn.__name__))
		X12, p = model.test_search(X1, label, search_fn=search_fn, num_steps=16)
		if np.isnan(X12).any():
			print('Search failed')
		else:
			n_obj  = X12.shape[0]
			n_vec = X12.shape[1]
			
			for obj in range(n_obj):
				ret = []
				scr = []
				for i, x in enumerate(X12[obj]):								
					ret.append(test_fn(x)[obj].numpy())
				ret = np.array(ret)
				hit_rate = ret.sum()/n_vec
				print('Objective %d - hit rate %0.2f' % (obj+1, 100*hit_rate))
			np.save('%s/search/epoch_%d_%s_%s_.npy' % (sample_dir, epoch, tag, test_fn.__name__), X12)
			utils.plot_hits(X12, savepath='%s/search/epoch_%d_%s_%s_.png' % (sample_dir, epoch, tag, test_fn.__name__))


def sample_test_vectors(X1, label, tag):
	X12 = model.translate(X1, label)
	np.save('%s/sample/%s_%s_ref_trans.npy' % (sample_dir, model_name, tag),
		X12)
	
	X12 = model.sample_simplex(X1, 0, n_samples=128)
	np.save('%s/sample/%s_%s_sample.npy' % (sample_dir, model_name, tag),
		X12)
	X12 = X12[np.random.randint(0, X12.shape[0], size=16)]
	utils.plot_samples(X12.reshape(4, 4, 3, 512),
		savepath='%s/sample/epoch_%d_%s_sample.png' % (sample_dir, epoch, tag))



def sweep_test_vectors(X1, label, tag, mlerp=True):
	if mlerp:
		X12, X121 = model.metric_sweep(X1, 0, metric_fn=l1_similarity, n_samples=16)
	else:
		X12, X121 = model.sweep(X1, 0, n_samples=128)
		X12 = X12[np.arange(0, 128, 128//16)]
	utils.plot_sweep(X12, X121, label, num_samples=16,
		savepath='%s/sample/epoch_%d_%s_sweep.png' % (sample_dir, epoch, tag))

def translate_templates(X1, label, tag):
	X12 = model.translate(torch.from_numpy(X1).float(), 0)
	np.save('%s/sample/%s_%s_sweep.npy' % (sample_dir, model_name, tag),
		X12)
	utils.plot_samples(X12[0].reshape(5, 2, 3, 512),
		savepath='%s/sample/epoch_%d_%s_translate.png' % (sample_dir, epoch, tag))		

def check_coverage(test_fn, X12):
	ret = []
	for x in X12:
		ret.append(test_fn(x))
	print(np.sum(ret), np.mean(ret))


if __name__ == '__main__':
		
	model_dirs = ['models/45704/'] 
	for model_dir in model_dirs:		
		model_name = model_dir.split('/')[-2]
		print('Processing %s' % model_name)

		save_dir = 'sil/hits'
		os.makedirs(save_dir, exist_ok=True)
		
		with open(model_dir+ 'opt.json', 'r') as f:
			opt = Namespace(**json.load(f))
		
		
		from networks.silgan_exp import SilGAN

		for _epoch in [opt.n_epochs]: #np.arange(opt.checkpoint_interval, opt.n_epochs+1, step=opt.checkpoint_interval):
			epoch = _epoch-1
			print('Sampling checkpoint %d' % epoch)
			model = SilGAN(opt)
			model.load(model_dir, epoch)
			seed_fn(0)
	
			seq_len = 512
			label = 0
			T1 = templates.takeoff(num=10, k=0.1, len=seq_len)
			T2 = templates.stop_takeoff(num=10, k=0.1, len=seq_len)		
			X1 = np.stack([T1[0], T1[-1], T2[-1]], axis=0)
			X1 = torch.from_numpy(X1).float()

			
			X12, loss_landscape = model.flat_test_search(X1, label, search.test_fn4, num_steps=3000)

			for x in X12[0]:
				if not torch.isnan(x).any():
					print(test.test_fn4(x))
			utils.plot_samples(X12[0].reshape(4, 4, 3, 512),
				savepath='%s/hits.png' % (save_dir))
			np.save('%s/hits.npy' % save_dir, X12[0])			
			