import os
import math
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
from pytorch_lightning.loggers import WandbLogger
import wandb
import metrics


def setup_logging(project, job_id, hparams, logdir='./logs', run_id=None):
	training_run = job_id
	if run_id is None:
		wandb.init(
			name=training_run,
			project=project,
			dir=logdir
		)
	else:
		wandb.init(
			id = run_id,
			name=training_run,
			project=project,
			dir=logdir,
			resume='must'
		)
	wandb.config.update(hparams, allow_val_change=True)
	return wandb.run.id

def log_metrics(metrics, step):
	wandb.log(metrics, step=step)

def samples_to_log(samples):
	img = wandb.Image(samples)
	plt.close()
	return img

def truncate(X1, X2):
	seq_len = X2.shape[-1]
	act_len = seq_len//2
	mX1, mX2 = [], []
	M = torch.zeros_like(X1, dtype=torch.float)
	for (m, x1, x2) in zip(M, X1, X2):
		st = 256 #np.random.randint(0, seq_len-act_len)
		en = st + act_len
		m[0, st : en] = 1.0
		mX1.append(x1[:, st : en])
		mX2.append(x2[:, st : en])
	return M, torch.stack(mX1, dim=0), torch.stack(mX2, dim=0)

def sliding_window(batch_size, num_styles=4):
	M = torch.zeros((num_styles, batch_size, 1, 1024), dtype=torch.float)
	P = np.array([0, 128, 256, 512])
	for i in range(num_styles):
		M[i, :, :, P[i] : P[i]+512] = 1.0
	return M, P

def fixed_window(batch_size, num_styles=4):
	M = torch.zeros((num_styles, batch_size, 1, 1024), dtype=torch.float)
	P = np.full(num_styles, np.random.randint(0, 512))
	for i in range(num_styles):
		M[i, :, :, P[i] : P[i]+512] = 1.0
	return M, P

def frozen_window(batch_size, num_styles=4):
	M = torch.zeros((num_styles, batch_size, 1, 1536), dtype=torch.float)
	P = np.full(num_styles, 512)
	for i in range(num_styles):
		M[i, :, :, P[i] : P[i]+512] = 1.0
	return M, P


# def fixed_window(batch_size, num_styles=4):
# 	M = torch.zeros((num_styles, batch_size, 1, 1024), dtype=torch.float)
# 	P = np.array([0, 128, 256, 512])
# 	for i in range(batch_size):
# 		s = np.random.choice(P)
# 		M[:, i, :, s : s+512] = 1.0
# 	return M, P

def sliding_mask(batch_size, num_styles=3):
	M = torch.zeros((num_styles, batch_size, 1, 1024), dtype=torch.float)
	st = [0, 128, 256, 512]
	for i in range(num_styles):
		M[i, :, :, st[i] : st[i]+512] = 1.0	
	return M

def position_template(X1, l, P):
	X1 = torch.unsqueeze(X1[:, l, :], dim=1)
	X1 = X1.repeat(P.shape[0], 1, 1, 1)
	X1P = torch.full_like(P, np.nan)	
	X1P[P == 1.] = X1.view(-1)
	return X1P

def plot_samples(series, savepath=None, size=(20,10), axis='on', 
	linewidth=1.0,
	title_text=None, fig_text=None, 
	ylabels=None, ylabel_coords=None,
	yticks=True, xticks=None,
	text_pos = None, xlabels = False, 
	borders=True, goal_post=True,
	legends=False, legend_bbox=(0.0, 1.1), legend_ax=None,
	font_size=14, fig_title=None):
	
	def _subplot(a, i, j):
		s = series[i, j]
		if not borders:
			a.spines['top'].set_visible(False)
			a.spines['right'].set_visible(False)
			a.spines['left'].set_visible(False)
			a.spines['bottom'].set_visible(False)
		if not goal_post:
			a.spines['top'].set_visible(False)
			a.spines['right'].set_visible(False)
			a.spines['left'].set_visible(False)
		if not (title_text is None):
			a.set_title(title_text[i, j], fontdict={'fontsize': font_size})
		if not (ylabels is None):
			a.set_ylabel(ylabels[i, j], fontsize=font_size)
		a.set_xticks([])
		if not (xticks is None):
			if not np.isnan(xticks[i, j]).any():
				a.set_xticks(xticks[i, j])
				a.xaxis.set_tick_params(labelsize=font_size)
		if xlabels:		
			a.set_xlabel('seconds', fontsize=font_size)
		if not (fig_text is None):
			a.text(text_pos[i, j, 0], text_pos[i, j, 1], fig_text[i, j], size=font_size)
		a.plot(s[0], color='blue',label='veh', linewidth=linewidth)
		a.plot(s[1], color='green',label='eng', linewidth=linewidth)
		a.plot(s[2], color='red', alpha=0.5, label='gear', linewidth=linewidth)
		if n_channels == 4:
			a.plot(s[3], color='black')
		a.set_ylim(-0.05, 1)			
		
		if yticks :
			a.set_yticks([0, 0.5, 1.0])
			a.yaxis.set_tick_params(labelsize=font_size)
		else:
			a.set_yticks([])


	if len(series.shape) < 4:
		raise Exception('Series format (cols, rows, channels, signals)')
	c = series.shape[0]
	r = series.shape[1]
	n_channels = series.shape[2]
	seq_len = series.shape[-1]

	fig, axs = plt.subplots(ncols=c, nrows=r, figsize=size)

	
	if c == 1 and r == 1:
		_subplot(axs, 0, 0)

	else:
		if legends and (legend_ax is None):
			legend_ax = (c-1, 0)
		
		if r == 1:
			axs = np.expand_dims(axs, axis=0)

		for j, row in enumerate(axs):
			for i, col in enumerate(row):
				s = series[i, j]
				a = col
				if not (title_text is None):
					a.set_title(title_text[i, j], fontdict={'fontsize': font_size})
				if not (ylabels is None):
					a.set_ylabel(ylabels[i, j], fontsize=font_size)
					if not ylabel_coords is None:
						a.yaxis.set_label_coords(ylabel_coords[0], ylabel_coords[1])
				a.set_xticks([])
				if not (xticks is None):
					if not np.isnan(xticks[i, j]).any():
						a.set_xticks(xticks[i, j])
						a.xaxis.set_tick_params(labelsize=font_size)
				if xlabels and j == r-1:			
					a.set_xlabel('seconds', fontsize=font_size)
				if not (fig_text is None):
					a.text(text_pos[i, j, 0], text_pos[i, j, 1], fig_text[i, j], size=font_size)
				if not borders:
					a.spines['top'].set_visible(False)
					a.spines['right'].set_visible(False)
					a.spines['left'].set_visible(False)
					a.spines['bottom'].set_visible(False)
				if not goal_post:
					a.spines['top'].set_visible(False)
					a.spines['right'].set_visible(False)
					a.spines['left'].set_visible(False)
				a.plot(s[0], color='blue',label='veh')
				a.plot(s[1], color='green',label='eng')
				a.plot(s[2], color='red', alpha=0.5, label='gear')
				if n_channels == 4:
					a.plot(s[3], color='black')
				a.set_ylim(-0.05, 1)			
				
				if yticks and i == 0:
					a.set_yticks([0, 0.5, 1.0])
					a.yaxis.set_tick_params(labelsize=font_size)
				else:
					a.set_yticks([])
			if legends and (i == legend_ax[0]) and (j == legend_ax[1]):
				handles, labels = a.get_legend_handles_labels()	
				a.legend(handles, labels, ncol=3, bbox_to_anchor=legend_bbox,
					loc='upper left', frameon=False, prop={'size': font_size})
	if not (fig_title is None):
		fig.suptitle(fig_title)

	if savepath is None:
		return fig
	else:
		fig.savefig(savepath, bbox_inches='tight')
		plt.close()

	# print(axs.shape)
	
	# if c == 1 and r == 1:
	# 	axs = np.array([axs])
	# 	axs = axs.reshape(1, 1)
	# elif c == 1:
	# 	axs = axs.reshape(1, -1)
	# elif r == 1:
	# 	axs = axs.reshape(-1, 1)

	# seq_len = series.shape[-1]	
	# for i in range(c):
	# 	for j in range(r):
	# 		s 	= series[i, j]		
	# 		a   = axs[i, j]		
	# 		if not (title_text is None):
	# 			a.set_title(title_text[i, j], fontdict={'fontsize': font_size})
	# 		if not (fig_text is None):
	# 			xpos = int(0.7*seq_len) if c < 4 else int(0.6*seq_len)
	# 			a.text(xpos, 0.8, fig_text[i, j], size=font_size)
	# 		a.plot(s[0], color='blue',label='vehicle_speed')
	# 		a.plot(s[1], color='green',label='engine_speed')
	# 		a.plot(s[2], color='red', alpha=0.5, label='selected_gear')
	# 		if n_channels == 4:
	# 			a.plot(s[3], color='black',label='template')
	# 		a.set_ylim(-0.05, 1)
	# 		if xticks and j == r-1:
	# 			a.set_xticks([0, seq_len])
	# 			a.xaxis.set_tick_params(labelsize=font_size)
	# 			a.set_xlabel('seconds', fontsize=font_size)
	# 		else:
	# 			a.set_xticks([])
	# 		if yticks and i == 0:
	# 			a.set_yticks([0, 0.5, 1.0])
	# 			a.yaxis.set_tick_params(labelsize=font_size)
	# 		else:
	# 			a.set_yticks([])			
	# if legends:
	# 	handles, labels = a.get_legend_handles_labels()			
	# 	axs[c-1, 0].legend(handles, labels, prop={'size': font_size})
	# if not (fig_title is None):
	# 	fig.suptitle(fig_title)
	# if savepath is None:
	# 	return fig
	# else:
	# 	fig.savefig(savepath, bbox_inches='tight')
	# 	plt.close()

def _plot_samples(series, savepath=None, size=(20,10), axis='on', 
	title_text=None, fig_text=None, yticks=True, xticks=False,
	legends=False, font_size=14, fig_title=None):
	
	if len(series.shape) < 4:
		raise Exception('Series format (cols, rows, channels, signals)')
	c = series.shape[0]
	r = series.shape[1]
	n_channels = series.shape[2]	

	fig, axs = plt.subplots(ncols=c, nrows=r, figsize=size)
	# l_ax = axs[0, r-1]
	seq_len = series.shape[-1]	
	for i in range(c):
		for j in range(r):
			s 	= series[i, j]
			a   = axs[j, i]		
			if not (title_text is None):
				a.set_title(title_text[i, j], fontdict={'fontsize': font_size})
			if not (fig_text is None):
				xpos = int(0.7*seq_len) if c < 4 else int(0.6*seq_len)
				a.text(xpos, 0.8, fig_text[i, j], size=font_size)
			a.plot(s[0], color='blue',label='vehicle_speed')
			a.plot(s[1], color='green',label='engine_speed')
			a.plot(s[2], color='red', alpha=0.5, label='selected_gear')
			if n_channels == 4:
				a.plot(s[3], color='black',label='template')
			a.set_ylim(-0.05, 1)
			if xticks and j == r-1:
				a.set_xticks([0, seq_len])
				a.xaxis.set_tick_params(labelsize=font_size)
				a.set_xlabel('seconds', fontsize=font_size)
			else:
				a.set_xticks([])
			if yticks and i == 0:
				a.set_yticks([0, 0.5, 1.0])
				a.yaxis.set_tick_params(labelsize=font_size)
			else:
				a.set_yticks([])			
	if legends:
		handles, labels = a.get_legend_handles_labels()			
		l_ax.legend(handles, labels, prop={'size': font_size})
	if not (fig_title is None):
		fig.suptitle(fig_title)
	if savepath is None:
		return fig
	else:
		fig.savefig(savepath, bbox_inches='tight')
		plt.close()

def plot_sweep(X12, X121, _l, X1=None, t_dists=None, num_samples=10, savepath=None):	
	c = 4
	r = num_samples//2
	b = X12.shape[0]
	if X1 is None:
		n = X12.shape[1]
	else:
		n = X12.shape[1] + 1
	l = X12.shape[2]

	sequences = np.full((c, r, n, l), np.nan)
	titles = np.empty((c, r), dtype="<U25")
	texts = np.empty((c, r), dtype="<U25")
	if not t_dists is None:
		t_dists_txt = np.array(['t_dist-%0.3f' % t_dist for t_dist in t_dists[0 : num_samples]] )
		texts[1] = t_dists_txt[0 : r]
		texts[3] = t_dists_txt[r : 2*r]
		
	if not X1 is None:
		X12 = torch.cat((X12, X1[:, _l:_l+1]), dim=1)	
	sequences[0, :, _l, :] = X121.numpy()[0 : r, 0, :]
	sequences[1] = X12.numpy()[0 : r]
	
	sequences[2, :, _l, :] = X121.numpy()[r : 2*r, 0, :]
	sequences[3] = X12.numpy()[r : 2*r]
	

	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles, fig_text=texts)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, title_text=titles, fig_text=texts, savepath=savepath)

def plot_hits(X12, num_per_obj=8, savepath=None):
	c = 2*X12.shape[0]
	r = num_per_obj//2
	b = X12.shape[1]
	n = X12.shape[2]
	l = X12.shape[3]
	sequences = np.full((c, r, n, l), np.nan)
	titles = np.empty((c, r), dtype="<U25")
	texts = np.empty((c, r), dtype="<U25")

	X12 = X12.numpy()
	for obj in range(X12.shape[0]):		
		sequences[2*obj] = X12[obj, 0 : r]
		sequences[2*obj+1] = X12[obj, r : 2*r]
		titles[2*obj] = 'Objective - %d' % (obj + 1)
		titles[2*obj+1] = 'Objective - %d' % (obj + 1)

	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles, fig_text=texts)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, title_text=titles, fig_text=texts, savepath=savepath)

def visualize_test_mini(x_mini, _l, num_samples=10, savepath=None):
	X1, X2 = x_mini
	c = 4
	r = num_samples//2
	b = X2.shape[0]
	n = X2.shape[1]
	l = X2.shape[2]

	sequences = np.full((c, r, n, l), np.nan)
	titles = np.empty((c, r), dtype="<U25")
	
	sequences[0, :, _l, :] = X1.numpy()[0 : r, _l, :]
	sequences[1] = X2.numpy()[0 : r]
	sequences[2, :, _l, :] = X1.numpy()[r : 2*r, _l, :]
	sequences[3] = X2.numpy()[r : 2*r]

	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, title_text=titles, savepath=savepath)

	

def visualize_batch(batch, num_samples=5, savepath=None):

	X1, X2 = batch
	c = 4
	r = num_samples
	b = X2.shape[0]
	n = X2.shape[1]
	l = X2.shape[2]
	
	
	sequences = np.full((c, r, n, l), np.nan)
	titles = np.empty((c, r), dtype="<U25")		

	X1 = torch.cat([X1, torch.full((b, 1, l), np.nan)], dim=1)
	
	sequences[0] = X1[0: r]
	titles[0, 0] = 'Template'
	sequences[1] = X2[0: r]
	titles[1, 0] = 'Signal'

	sequences[2] = X1[r: 2*r]
	titles[2, 0] = 'Template'
	sequences[3] = X2[r: 2*r]
	titles[3, 0] = 'Signal'
	

	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, title_text=titles, savepath=savepath)


def plot_reconstruction(X1, X2, X11, X22,
	num_samples=10, savepath=None):
	c = 4
	r = num_samples//2
	sequences = np.full((c, r, X2.shape[1], X2.shape[2]), np.nan)
	
	ref = np.hstack([X2.numpy(), X1.numpy()])
	rec = np.hstack([X22.numpy(), X11.numpy()])
		
	sequences[0, :, 0:2] = X1[0: r].numpy() 
	sequences[1, :, 0:2] = X11[0: r].numpy()
	sequences[2] = X2[0 : r].numpy()
	sequences[3] = X22[0 : r].numpy()

	titles  = np.empty((c, r), dtype="<U15")
	titles[0, 0] = 'Test templates'
	titles[1, 0] = 'Reconstruction'
	titles[2, 0] = 'Test samples'
	titles[3, 0] = 'Reconstruction'

	texts = np.empty((c, r), dtype="<U15")	
	texts[1] = ['SSIM-%0.3f' % s for s in metrics.batch_ssim(X1.numpy(), X11.numpy())][0 : r]
	texts[3] = ['SSIM-%0.3f' % s for s in metrics.batch_ssim(X2.numpy(), X22.numpy())][0 : r]

	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles
			, fig_text=texts)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, title_text=titles, 
			fig_text=texts, savepath=savepath)


def plot_translation(X1, l, X12, X2=None,
	num_samples=5, savepath=None):
	num_styles = X12.shape[0]
	c = (1 + num_styles)
	r = num_samples
	n = X12.shape[2]
	s = X12.shape[3]
	sequences = np.full((c, r, n, s), np.nan)
	titles = np.empty((c, r), dtype="<U25")
	texts  = np.empty((c, r), dtype="<U15")
	ref = X1.numpy()
	trn = X12.numpy()
	sequences[0, :, l, :] = ref[0: r, l, :]
	titles[0, 0] = 'Template'	

	for i in range(num_styles):
		sequences[i + 1] = trn[i][0 : num_samples]
		title_str = 'Style - %d' % i
		titles[i + 1, 0] = title_str
		if not X2 is None:
			ssims = ['SSIM-%0.3f' % s for s in metrics.batch_ssim(X2, trn[i])]
			texts[i + 1] = ssims[0 : r]

	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles,
				fig_text=texts)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, title_text=titles, 
			fig_text=texts, savepath=savepath)

def _plot_translation(X1, l, X2, X12,
	num_samples=5, savepath=None):
	num_styles = X12.shape[0]
	c = (1 + num_styles)
	r = num_samples
	n = X2.shape[1]
	s = X2.shape[2]
	sequences = np.full((c, r, n, s), np.nan)
	titles = np.empty((c, r), dtype="<U25")
	texts  = np.empty((c, r), dtype="<U15")
	ref = X1.numpy()
	trn = X12.numpy()
	sequences[0, :, l, :] = ref[0: r, l, :]
	titles[0, 0] = 'Template'	

	for i in range(num_styles):
		sequences[i + 1] = trn[i][0 : num_samples]
		title_str = 'Style - %d' % i
		titles[i + 1, 0] = title_str
		ssims = ['SSIM-%0.3f' % s for s in metrics.batch_ssim(X2, trn[i])]
		texts[i + 1] = ssims[0 : r]

	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles,
				fig_text=texts)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, title_text=titles, 
			fig_text=texts, savepath=savepath)

def plot_cycle_translation(X1, l, X121,
	num_samples=5, savepath=None):
	num_styles = X121.shape[0]
	c = (1 + num_styles)
	r = num_samples
	n = X1.shape[1]+1
	s = X1.shape[2]
	sequences = np.full((c, r, n, s), np.nan)
	titles = np.empty((c, r), dtype="<U25")
	texts  = np.empty((c, r), dtype="<U15")
	ref = X1.numpy()
	trn = X121.numpy()
	sequences[0, :, l, :] = ref[0: r, l, :]
	titles[0, 0] = 'Template'	

	for i in range(num_styles):
		sequences[i + 1, :, l] = trn[i, 0 : num_samples, 0, :]
		title_str = 'Style - %d' % i
		titles[i + 1, 0] = title_str
		ssims = ['SSIM-%0.3f' % s for s in metrics.batch_ssim(ref[:, l:l+1, :], trn[i])]
		texts[i + 1] = ssims[0 : r]

	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles,
				fig_text=texts)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, title_text=titles, 
			fig_text=texts, savepath=savepath)

def _plot_expansion(X1, l, X13, 
	num_samples=5, savepath=None):
	X1 = torch.unsqueeze(X1[:, l, :], dim=1)
	num_styles = X13.shape[0]
	c = num_styles
	r = num_samples
	sequences = np.empty((c, r, (1 + X13.shape[2]), X13.shape[3]), 
		dtype=np.float32)
	titles  = np.empty((c, r), dtype="<U15")

	
	pad_len = X13.shape[-1] - X1.shape[-1]
	ref = torch.nn.functional.pad(X1, (pad_len//2, pad_len//2, 0, 0, 0, 0), 
			value=np.nan).numpy()	
	trn = X13.numpy()	

	for i in range(num_styles):
		if i%2 == 0:
			sequences[i] = np.hstack([np.full_like(ref, np.nan), np.full_like(ref, np.nan), np.full_like(ref, np.nan), ref])[0 : num_samples]
			sequences[i, :, l] = trn[i, :, l, :][0 : num_samples]
		else:
			sequences[i] = np.hstack([trn[i], np.full_like(ref, np.nan)])[0 : num_samples]
		titles[i, 0] = 'Style - %d' % i	
	
	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, 
			title_text=titles, savepath=savepath)

def plot_expansion(X1, l, X13, 
	num_samples=5, savepath=None):
	X1 = torch.unsqueeze(X1[:, l, :], dim=1)
	num_styles = X13.shape[0]
	c = num_styles
	r = num_samples
	sequences = np.empty((c, r, (1 + X13.shape[2]), X13.shape[3]), 
		dtype=np.float32)
	titles  = np.empty((c, r), dtype="<U15")

	
	ref = torch.nn.functional.pad(X1, (256, 256, 0, 0, 0, 0), 
			value=np.nan).numpy()	
	trn = X13.numpy()	

	for i in range(num_styles):
		if i%2 == 0:
			sequences[i] = np.hstack([np.full_like(ref, np.nan), np.full_like(ref, np.nan), np.full_like(ref, np.nan), ref])[0 : num_samples]
			sequences[i, :, l] = trn[i, :, l, :][0 : num_samples]
		else:
			sequences[i] = np.hstack([trn[i], np.full_like(ref, np.nan)])[0 : num_samples]
		titles[i, 0] = 'Style - %d' % i	
	
	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, 
			title_text=titles, savepath=savepath)

def plot_join(X1_1, X1_2, X13, l,
	num_samples=5, savepath=None):
	num_styles = X13.shape[0]
	c = num_styles
	r = num_samples
	sequences = np.full((c, r, (1 + X13.shape[2]), X13.shape[3]), np.nan)
	titles  = np.empty((c, r), dtype="<U15")

	for i in range(num_styles):
		join = X13[i].numpy()
		ref  = np.full((join.shape[0], 1, join.shape[2]), np.nan)
		ref[:, :, 0:256] = X1_1[:, l:l+1, 256:512]
		ref[:, :, 768: 1024] = X1_2[:, l:l+1, 0:256]
		if i%2 == 0:
			sequences[i, :, l, :] = join[:, l, :][0 : r]
			sequences[i, :, 3, :] = ref[:, 0, :][0 : r]
		else:
			sequences[i] = np.hstack([join, ref])[0 : r]
		titles[i, 0] = 'Style - %d' % i
	
	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, 
			title_text=titles, savepath=savepath)


def plot_full_reconstruction(X2, X22,
	num_samples=10, savepath=None):
	c = 4
	r = num_samples//2
	sequences = np.empty((c, r, X2.shape[1], X2.shape[2]), 
		dtype=np.float32)
	ref = X2
	rec = X22	
	sequences[0] = ref[0: 5]
	sequences[1] = rec[0: 5]
	sequences[2] = ref[5 : 10]
	sequences[3] = rec[5 : 10]

	titles  = np.empty((c, r), dtype="<U15")
	titles[0, 0] = 'Test sample'
	titles[1, 0] = 'Reconsturction'
	titles[2, 0] = 'Test sample'
	titles[3, 0] = 'Reconsturction'

	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, 
			title_text=titles, savepath=savepath)

def plot_full_translation(X1, X2, X12, M,
	num_samples=10, savepath=None):
	c = 4
	r = num_samples//2
	sequences = np.empty((c, r, (1+X2.shape[1]), X2.shape[2]), 
		dtype=np.float32)
	t = np.copy(X1.numpy())
	t[M == 0.] = np.nan
	ref = np.hstack([X2.numpy(), t])
	rec = np.hstack([X12.numpy(), t])
	sequences[0] = ref[0: 5]
	sequences[1] = rec[0: 5]
	sequences[2] = ref[5 : 10]
	sequences[3] = rec[5 : 10]

	titles  = np.empty((c, r), dtype="<U15")
	titles[0, 0] = 'Test sample'
	titles[1, 0] = 'Translation'
	titles[2, 0] = 'Test sample'
	titles[3, 0] = 'Translation'

	if savepath is None:
		seq = _plot_samples(sequences, title_text=titles)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, 
			title_text=titles, savepath=savepath)

def plot_masked_generation(X1, X2, X12, 
	M, UM, num_samples=5, savepath=None):
	t = np.copy(X1.numpy())
	t[M.numpy() == 0.] = np.nan
	num_styles = X12.shape[0]
	c = 1 + num_styles
	r = num_samples
	sequences = np.empty((c, r, (1 + X2.shape[1]), X2.shape[2]), 
		dtype=np.float32)
	sequences[0] = np.hstack([X2.numpy(), t])[0 : r]
	titles  = np.empty((c, r), dtype="<U15")
	texts  = np.empty((c, r), dtype="<U15")
	for i in range(num_styles):
		titles[i+1, 0] = 'Style %d' % i
		m = np.copy(UM[i].numpy())
		m[m == 0.] = np.nan
		m[m == 1.] = t[M.numpy() == 1.]
		sequences[i + 1] = np.hstack([X12[i].numpy(), m])[0: r]

	if savepath is None:
		seq = _plot_samples(sequences, 
			title_text=titles, fig_text=texts)
		return samples_to_log(seq)
	else:
		_plot_samples(sequences, 
			title_text=titles, fig_text=texts, savepath=savepath)

def histogram(f):
	hist, bins = np.histogram(f, range=(0., 1.), bins=50)
	freq = hist/np.sum(hist)
	return freq, bins

def plot_diversity(ssim_cg, ssim_ug, savepath, fontsize=14, size=(20,10)):
	p_cg, bins_cg = histogram(ssim_cg)
	p_ug, bins_ug = histogram(ssim_ug)

	fig, axs = plt.subplots(figsize=size)
	axs.bar(bins_cg[:-1], p_cg, align='edge', width=np.diff(bins_cg), color='b', alpha=0.5, label='cgen')
	axs.bar(bins_ug[:-1], p_ug, align='edge', width=np.diff(bins_ug), color='r', alpha=0.5, label='ugen')
	axs.legend(prop={'size': fontsize})
	fig.savefig(savepath)
