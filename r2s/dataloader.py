import os
import math
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from .edge_detection import get_edges


SIG_VEHICLE_SPEED_HI = 90.0 #120.0
SIG_ENGINE_SPEED_HI = 3000.0
SIG_SELECTED_GEAR_HI = 15.0

def collect(source_npz):
	np_data = np.load(source_npz)['series']	
	print('Total collected', np_data.shape)
	npoints       = np_data.shape[-1]	
	npoints_mul2  = 2**(math.floor(math.log(npoints, 2)))
	npoints_start = (npoints - npoints_mul2)//2
	npoints_end   = npoints - npoints_start
	np_data       = np_data[:, :, npoints_start:npoints_end]
	print('Reshaped to', np_data.shape)

	v = np_data[:, 0, :]
	v[v > SIG_VEHICLE_SPEED_HI] = SIG_VEHICLE_SPEED_HI
	v[v < 0.] = 0
	v = v/SIG_VEHICLE_SPEED_HI

	e = np_data[:, 1, :]
	e[e > SIG_ENGINE_SPEED_HI] = SIG_ENGINE_SPEED_HI
	e[e < 0.] = 0.
	e = e/SIG_ENGINE_SPEED_HI

	g = np_data[:, 2, :]	
	g[g < 0.] = 0
	g = g/SIG_SELECTED_GEAR_HI

	np_data = np.stack([v, e, g], axis=1)

	return np_data

def collect_templates(x_array):
	if len(x_array) > 100:
		x_array = tqdm(x_array)	

	t_array  = []
	_x_array = []
	for i, x in enumerate(x_array):
		v = x[0]
		e = x[1]
		g = x[2]
		try:
			t_v, _, _ = get_edges(v, signal_type='v')
			t_e, _, _ = get_edges(e, signal_type='e')
			# t_g, _, _ = get_edges(g, signal_type='g')
			# t = np.stack([t_v, t_e, t_g], axis=0)			
			t = np.stack([t_v, t_e], axis=0)
			t_array.append(t)
			_x_array.append(x)
		except:
			print('problem %d' % i)			
	return np.array(t_array), np.array(_x_array)



def dataloader(r2s_root, batch_size, num_workers, shuffle_2=False, shuffle_3=True, shuffle=True, worker_init_fn=None):
		
	source_npz = r2s_root + '/data/r2s-20min-renorm.npz'

	if not os.path.isfile(source_npz):		
		x_r2s = collect(os.path.join(r2s_root, 'processed/r2s-20min-renorm.npz'))
		n_test_samples = x_r2s.shape[0]//10
		np.random.shuffle(x_r2s)
		x_test  = x_r2s[:n_test_samples, :]
		x_train = x_r2s[n_test_samples:, :]
		
		print('Collecting train templates')
		t_train, x_train = collect_templates(x_train)
		print('Collecting test templates')
		t_test, x_test   = collect_templates(x_test)

		i_mini = np.random.randint(0, t_test.shape[0], size=15)
		t_mini = t_test[i_mini]
		x_mini = x_test[i_mini]
		

		np.savez_compressed(source_npz, 
			x_train=x_train, t_train=t_train,
			x_test=x_test, t_test=t_test,
			x_mini=x_mini, t_mini=t_mini)
	else:
		raw_data = np.load(source_npz)
		x_train = raw_data['x_train']
		t_train = raw_data['t_train']
		x_test  = raw_data['x_test']
		t_test  = raw_data['t_test']
		x_mini  = raw_data['x_mini']
		t_mini  = raw_data['t_mini']
	
	# x_mini = np.load(r2s_root + 'npz_20min_3d/x_mini.npy')
	# mini_idx = [5, 10, 7, 1, 9, 8, 2, 3, 13, 11, 12, 14, 0, 4, 6]
	# x_mini = x_mini[mini_idx]
	# t_mini, _ = collect_templates(x_mini)

	X1_train = t_train[:, :, 256:768]
	X2_train = x_train[:, :, 256:768]
	X1_test = t_test[:, :, 256:768]
	X2_test = x_test[:, :, 256:768]
	X1_mini = t_mini[:, :, 256:768]
	X2_mini = x_mini[:, :, 256:768]

	if shuffle_2:
		X2_train = np.random.permutation(X2_train)
		X2_test  = np.random.permutation(X2_test)
		#Mini used only for visualization, so that's not shuffled

	if shuffle_3:
		X3_train = np.random.permutation(x_train)
		X3_test = np.random.permutation(x_test)
		X3_mini = np.random.permutation(x_mini)
	else:
		X3_train = x_train
		X3_test = x_test
		X3_mini = x_mini


	train_dataset = TensorDataset(torch.from_numpy(X1_train).float(),
		torch.from_numpy(X2_train).float(), torch.from_numpy(X3_train).float())	
	test_dataset = TensorDataset(torch.from_numpy(X1_test).float(),
		torch.from_numpy(X2_test).float(), torch.from_numpy(X3_test).float())	
	test_mini = (torch.from_numpy(X1_mini).float(),
		torch.from_numpy(X2_mini).float(), torch.from_numpy(X3_mini).float())
		
	train_dataloader = DataLoader(train_dataset, 
		batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=shuffle)
	test_dataloader = DataLoader(test_dataset, 
		batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=shuffle)

	return train_dataloader, test_dataloader, test_mini
	

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import sys
	sys.path.append('../')
	import utils6 as utils

	data_root = '/cephyr/NOBACKUP/groups/snic2020-8-120/bilgan/r2s/'
	train_dataloader, test_dataloader, test_mini = dataloader(data_root, 
		128, 16, shuffle_2=False)

	batch = next(iter(train_dataloader))
	utils.visualize_batch(batch, savepath='temp/batch.png')
	utils.visualize_batch(test_mini, savepath='temp/test_mini.png')