import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from tqdm import tqdm

def plot15(series, fname, tmpl=None):
	n_samples = series.shape[0]
	if n_samples > 15:
	  idx = np.random.randint(0, n_samples, 15)
	  series = series[idx]	  

	series = series.reshape(5, 3, -1)
	if not tmpl is None:		
		if n_samples > 15:
			tmpl = tmpl[idx]
		tmpl   = tmpl.reshape(5, 3, -1)
	figure, ax = plt.subplots(5, 3, figsize=(20,10))
	for r in range(5):
		for c in range(3):
			ax[r, c].plot(series[r,c])
			if not tmpl is None:
				ax[r, c].plot(tmpl[r,c])
			ax[r, c].set_xticks([])
			ax[r, c].set_ylim([0, 1])
	plt.savefig(fname, bbox_inches='tight')


def obtain_pulses(s_t, min_drive_time = 0):
	
	## Function to retrieve the drive pulses
	## s_t is the vehicle speed drive signal
	
	stop_store = [] 
	drive_store = []
	in_stop = False
	
	## So if the vehicle starts in a stop then that is not counted as a stop in the labelling
	drive_pulse_encountered = False
	
	curr_stop = []
	curr_drive = []
	
	## Loop through the signal
	for idx,elem in enumerate(s_t.tolist()):

	  ## We encounter a stop
		if elem == 0 and not in_stop and drive_pulse_encountered:
			curr_stop.append(idx)
			in_stop = True
	
			## Store the drive pulse encountered before the stop
			if len(curr_drive) > 0:
				if len(curr_drive) > min_drive_time:
					curr_drive.append(idx)
					drive_store.append(curr_drive)
				curr_drive = []
	
		## Store elements of the stop
		elif elem == 0 and in_stop and drive_pulse_encountered:
			curr_stop.append(idx)
		
		## We encounter a drive pulse
		elif elem != 0:
			drive_pulse_encountered = True
			in_stop = False
			## Store the stop encountered before the drive pulse
			if len(curr_stop) > 0:
				stop_store.append(curr_stop)
				curr_stop = []
			curr_drive.append(idx)
		
	
	## If the signal ends in a stop
	if len(curr_stop) > 0:
		stop_store.append(curr_stop)
	
	if len(curr_drive)>0:
		drive_store.append(curr_drive)
	
	return drive_store, stop_store

def remove_spikes(x, min_pulse_len = 15):
	drive_pulses,_ = obtain_pulses(x)
	for pulse in drive_pulses:
		mid_pulse = x[pulse][1:-1]
		## Remove short pulse
		if len(pulse) < min_pulse_len:
			x[pulse] = 0.
		## Remove pulse that goes straight to one
		elif np.all(mid_pulse == np.ones(len(mid_pulse))):
			x[pulse] = 0.
	return x

def smoothing(x, rem_edge_eff=True):
	N = 8
	if rem_edge_eff:
		n = (N // 2)
		ext_x = n*[x[0]] + x.tolist() + n*[x[-1]]
		conv_x = np.convolve(ext_x, np.ones((N,))/N, mode='same')
		return conv_x[n:-n]
	else:
		return np.convolve(x, np.ones((N,))/N, mode='same')



def filter_edges(edges):
	inf = edges[np.where(np.diff(edges) > 1)]
	prev_ind = 0
	n_edges = []
	for ind in inf:
		n_ind =  np.median(edges[np.logical_and(edges > prev_ind, edges <= ind)])
		n_edges.append(int(n_ind))
		prev_ind = ind
	return np.array(n_edges)

# def get_edges(x_src, g_thr=0.03, i_thr=1):
# 	x = np.copy(x_src)
# 	x = remove_spikes(x)
# 	# x = smoothing(x)
# 	grad = ndimage.sobel(smoothing(x))
# 	edges = np.where(abs(grad) > g_thr)[0]	
# 	_edges = np.append(edges, np.inf)	
# 	inflections = edges[np.where(np.diff(_edges) > i_thr)]	
# 	tmpl = np.zeros_like(x)
# 	prev_inf = 0
# 	num_stops = 0
# 	slopes = []
	
# 	#special case - almost constant driving
# 	if (inflections.size == 0) and (x.mean() > 0.1):
# 		tmpl[prev_inf : len(x)] = np.max(x)		
# 		return tmpl, np.array([0]), np.array([])
		
	
# 	slope1 = 0
# 	for idx, inf in enumerate(inflections):
# 		if idx == len(inflections) - 1:
# 			nxt_inf = len(x)-1
# 		else:
# 			nxt_inf = inflections[idx + 1]
# 		run = edges[np.logical_and(edges > prev_inf, edges <= inf)]
# 		nxt_run = edges[np.logical_and(edges > inf, edges <= nxt_inf)]		
		
# 		# print(run, nxt_run)		
# 		slope1 = grad[run[0]]
# 		if slope1 <= 0:
# 			pre_run  = np.max(x[prev_inf: run[0]])			
# 		else:
# 			pre_run  = np.min(x[prev_inf: run[0]])
			
# 		if nxt_run.size == 0:
# 			post_run = x[-1]
# 		else:		
# 			slope2 = grad[nxt_run[0]]
# 			if slope2 > 0:
# 				post_run = np.min(x[run[-1]: nxt_run[0]])
# 			else:
# 				post_run = np.max(x[run[-1]: nxt_run[0]])

# 		if post_run == 0.:
# 			num_stops += 1
# 		slopes.append(slope1)
# 		tmpl[prev_inf : run[0]] = pre_run
# 		tmpl[run[0] : run[-1]+1] = np.linspace(pre_run, post_run, num=(run[-1] - run[0] + 1))
# 		# print('{:3d}'.format(prev_inf),
# 		# 	'{:0.3f}'.format(pre_run),
# 		# 	'{:3d}'.format(run[0]),
# 		# 	'{:+0.3f}'.format(slope1),
# 		# 	'{:3d}'.format(run[-1]),
# 		# 	'{:0.3f}'.format(post_run),
# 		# 	'{:3d}'.format(nxt_run[0]))	
# 		prev_inf = run[-1]
# 	if slope1 > 0: ## designed to deal with the ending
# 		tmpl[prev_inf : len(x)] = np.max(x[prev_inf : len(x)])
# 	else:
# 		tmpl[prev_inf : len(x)] = np.min(x[prev_inf : len(x)]) 
# 	labels = [num_stops]
# 	return tmpl, np.array(labels), np.array(slopes)


def adjustments(signal_type):    
    if signal_type == 'e':
        return np.mean, np.mean, 8
    elif signal_type == 'g':
        return np.max, np.min, 3
    elif signal_type == 'v':
        return np.max, np.min, 1

def get_edges(x_src, signal_type = 'e'):
    x = np.copy(x_src)
    x = remove_spikes(x)
    # x = smoothing(x)
    grad = ndimage.sobel(smoothing(x))
    f1, f2, i_thr = adjustments(signal_type)
    g_thr = 0.03
    edges = np.where(abs(grad) > g_thr)[0]
    _edges = np.append(edges, np.inf)    
    
    inflections = edges[np.where(np.diff(_edges) > i_thr)] 
    tmpl = np.zeros_like(x)
    prev_inf = 0
    num_stops = 0
    slopes = []
    
    
    if (inflections.size == 0) and (x.mean() > 0.1):
        tmpl[prev_inf : len(x)] = f1(x)        
        return tmpl, np.array([0]), np.array([])
        
    
    slope1 = 0
    for idx, inf in enumerate(inflections):
        if idx == len(inflections) - 1:
            nxt_inf = len(x)-1
        else:
            nxt_inf = inflections[idx + 1]
        run = edges[np.logical_and(edges > prev_inf, edges <= inf)]
        nxt_run = edges[np.logical_and(edges > inf, edges <= nxt_inf)]        
        
        # print(run, nxt_run)        
        slope1 = grad[run[0]]
        if slope1 <= 0:
            pre_run  = f1(x[prev_inf: run[0]])            
        else:
            pre_run  = f2(x[prev_inf: run[0]])
            
        if nxt_run.size == 0:
            post_run = x[-1]
        else:        
            slope2 = grad[nxt_run[0]]
            if slope2 > 0:
                post_run = f2(x[run[-1]: nxt_run[0]])
            else:
                post_run = f1(x[run[-1]: nxt_run[0]])

        if post_run == 0.:
            num_stops += 1
        slopes.append(slope1)
        tmpl[prev_inf : run[0]] = pre_run
        tmpl[run[0] : run[-1]+1] = np.linspace(pre_run, post_run, num=(run[-1] - run[0] + 1))
        # print('{:3d}'.format(prev_inf),
        #     '{:0.3f}'.format(pre_run),
        #     '{:3d}'.format(run[0]),
        #     '{:+0.3f}'.format(slope1),
        #     '{:3d}'.format(run[-1]),
        #     '{:0.3f}'.format(post_run),
        #     '{:3d}'.format(nxt_run[0]))    
        prev_inf = run[-1]
    if slope1 > 0: ## designed to deal with the ending
        tmpl[prev_inf : len(x)] = f1(x[prev_inf : len(x)])
    else:
        tmpl[prev_inf : len(x)] = f2(x[prev_inf : len(x)]) 
    labels = [num_stops]
    return tmpl, np.array(labels), np.array(slopes)

# def get_random_mask(t, n_active, active_len=64):	
# 	mask_len = n_active*active_len
# 	if mask_len >= len(t):
# 		mask_len = len(t)
# 		mask = np.ones_like(t, dtype=np.float)
# 	else:
# 		mask = np.zeros_like(t, dtype=np.float)
# 		seg_en   = 0
# 		while mask_len > 0:
# 			seg_st = np.random.randint(seg_en, len(t)-mask_len)
# 			seg_ln = active_len if mask_len == active_len else np.random.randint(1, mask_len//active_len)*active_len
# 			seg_en = seg_st + seg_ln	
# 			mask[seg_st : seg_en] = 1.0
# 			mask_len -= seg_ln
# 	return mask

# import torch
# def get_random_mask(t):
# 	seq_len = t.shape[-1]
# 	act_len = seq_len//2	
# 	mask = torch.zeros_like(t, dtype=torch.float)
# 	for m in mask:
# 		if np.random.randint(0, 2) == 0:
# 			m[0, :] = 1.0
# 		else:
# 			st = np.random.randint(0, seq_len-act_len)
# 			m[0, st : st+act_len] = 1.0	
# 	return mask

if __name__ == '__main__':
	r2s = np.load('/cephyr/NOBACKUP/groups/snic2020-8-120/bilgan/r2s/data/r2s-20min-3d-medium.npz')
	x_mini = r2s['x_train']
	e_speed = x_mini[:, 1, :]
	tmpl = []
	for i, e in enumerate(tqdm(e_speed)):
		try:
			_tmpl, _, _ = get_edges(e, signal_type='e')
			tmpl.append(_tmpl)
		except:
			print('problem %d' % i)
	tmpl    = np.array(tmpl)
	plot15(e_speed, 'temp/edge_detection_e_speed.png', tmpl=tmpl)

	# v_speed = x_mini[:, 0, :]
	# tmpl = []
	# for v in v_speed:
	# 	_tmpl, _, _ = get_edges(v, signal_type='v')
	# 	tmpl.append(_tmpl)
	# tmpl    = np.array(tmpl)
	# plot15(v_speed, 'temp/edge_detection_v_speed.png', tmpl=tmpl)

	# np.random.seed(0)
	# import oss
	# if not os.path.isdir('temp'):
	# 	os.mkdir('temp')
	# r2s = np.load('data/r2s-20min-medium.npz')
	# tmpl = r2s['t_mini']
	
	# # plot15(x_mini, 'temp/edge_detection_1_r2s_mini.png', tmpl=tmpl)
	
	# mask = get_random_mask(torch.from_numpy(tmpl)).numpy()
	# mask = mask.reshape(5, 3, 1, 1024).astype(int)
	# tmpl = tmpl.reshape(5, 3, 1, 1024)
	# fig, ax = plt.subplots(5, 3, figsize=(20, 10))
	# for r in range(5):
	# 	for c in range(3):
	# 		t = tmpl[r, c, 0]			
	# 		m = mask[r, c, 0]			
	# 		ax[r, c].plot(t, color='r', alpha=0.5)
	# 		t[m == 0] = np.nan
	# 		ax[r, c].plot(t, color='r')
	# 		# ax[r, c].text(400, 0.8, 'm_len=%d' % (n_active*active_len))
	# 		ax[r, c].set_ylim([0, 1])
	# fig.savefig('temp/mask.png')


	# import os
	# slopes_npy = 'temp/slopes.npy'
	# if not os.path.isfile(slopes_npy):
	# 	r2s = np.load('data/r2s-medium.npz')
	# 	x_train = r2s['x_train']
	# 	v_speed = x_train[:, 0, :]
	# 	slopes = []
	# 	for idx, v in enumerate(tqdm(v_speed)):
	# 		_, _, slope = get_edges(v)		
	# 		slopes.append(slope)	
	# 	slopes = np.concatenate(slopes)
	# 	np.save(slopes_npy, slopes)
	# else:
	# 	slopes = np.load(slopes_npy)

	# slopes = (120/3.6)*np.abs(slopes) #acc in m/s^2
	# hist, bins = np.histogram(slopes, bins=15)
	# freq = hist/np.sum(hist)
	# fig, ax = plt.subplots(1, figsize=(20,10))
	# ax.bar(bins[:-1], freq, align="edge", width=np.diff(bins), color='k', alpha=0.5)
	# fig.savefig('temp/edge_detection_2_r2s_medium_slopes.png')
	# plt.close()


