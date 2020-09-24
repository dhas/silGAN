# import os
# import logan
# import utils
# import json
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
# import metrics


def lerp(val,low,high):
	return (1-val) * low + val * high

def slerp(val, low, high):
	omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
	so = np.sin(omega)
	if so == 0:
		# L'Hopital's rule/LERP
		return (1.0-val) * low + val * high
	return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


def naive_interpolation(zs,ze,use_slerp=False,dim=None,num_waypoints=15):
	ps = np.linspace(0,1,num=num_waypoints)
	z_lerp = []
	for p in ps:
		if use_slerp:
			if dim is None:
				z_lerp.append(slerp(p,zs[0],ze[0]))
			else:
				z = np.copy(zs)
				z[0,dim] = slerp(p,zs[0,dim],ze[0,dim])
				z_lerp.append(z)
		else:
			if dim is None:
				z_lerp.append(lerp(p,zs,ze))
			else:
				z = np.copy(zs)
				z[0,dim] = lerp(p,zs[0,dim],ze[0,dim])
				z_lerp.append(z)
	
	return np.vstack(z_lerp)


def metric_interpolation(z_start, z_end, model, metric_fn, m_trajectory='linear',
	x_end=None, use_slerp=False, dim=None,
	num_waypoints=15, oversampling_ratio=20, verbose=False):
		
	if m_trajectory is 'sigmoid':
		ms = np.array(metric_fn(model.generate(z_start).detach().numpy(), x_end))
		me = np.array(metric_fn(model.generate(z_end).detach().numpy(), x_end))		
		trajectory = 1/(1 + np.exp(-(np.linspace(-5,5,num=num_waypoints))))
		trajectory[0] = 0.
		trajectory[-1] = 1.
		m_trajectory = np.array([t*me  + (1-t)*ms for t in trajectory])
	else:
		ms = np.array(metric_fn(model.generate(z_start).detach().numpy(), x_end))
		me = np.array(metric_fn(model.generate(z_end).detach().numpy(), x_end))		
		#m_trajectory = np.linspace(ms,me,num=num_waypoints)
		trajectory = np.linspace(0,1,num=num_waypoints)		
		m_trajectory = np.array([t*me  + (1-t)*ms for t in trajectory])
	
	metric_dim = len(ms)

	z_alphas 	= np.linspace(0,1,num=num_waypoints)
	z_dists 	= np.full(num_waypoints, np.inf)	
	
	num_steps = num_waypoints*oversampling_ratio
	for step in range(num_steps+1):
		if verbose:
			print('Step %d/%d' % (step,num_steps))
		alpha = 1.0 * step / num_steps
		if use_slerp:
			if dim is None:
				z = slerp(alpha,z_start[0],z_end[0]).reshape(1,-1)
			else:
				z = np.copy(z_start)
				z[0,dim] = slerp(alpha,z_start[0,dim],z_end[0,dim])
		else:
			if dim is None:
				z = lerp(alpha,z_start,z_end)
			else:
				z = np.copy(z_start)
				z[0,dim] = lerp(alpha,z_start[0,dim],z_end[0,dim])
		
		m_at_z = metric_fn(model.generate(z).detach().numpy(),x_end) #model.sess.run([metric_fn],feed_dict={model.z:z,x_target:x_end})

		m_dists = np.linalg.norm(m_trajectory - m_at_z, axis=1)
		closest_m_dist  = m_dists.min()
		closest_m_index = np.argmin(m_dists)		

		if closest_m_dist < z_dists[closest_m_index]:
			z_alphas[closest_m_index] 	= alpha
			z_dists[closest_m_index] 	= closest_m_dist			

	z_mlerp = []
	for alpha in z_alphas:
		if use_slerp:
			if dim is None:
				pos = slerp(alpha,z_start[0],z_end[0]).reshape(1,-1)
			else:
				pos = np.copy(z_start)
				pos[0,dim] = slerp(alpha,z_start[0,dim],z_end[0,dim])	
		else:
			if dim is None:
				pos = lerp(alpha,z_start,z_end)
			else:
				pos 		= np.copy(z_start)
				pos[0,dim] 	= lerp(alpha,z_start[0,dim],z_end[0,dim])
		z_mlerp.append(pos)

	z_mlerp = np.vstack(z_mlerp)
	# x_mlerp = model.generate(z_mlerp)

	# m_mlerp = []
	# for pos in z_mlerp:
	# 	m_at_z = metric_fn(model.generate(pos.reshape(1,-1)).detach().numpy(),x_end) #model.sess.run(metric_fn,feed_dict={model.z:pos.reshape(1,-1),x_target:x_end})
	# 	m_mlerp.append(m_at_z)

	return z_mlerp


def neighborhood_interpolation(x,model,dims,bound=3,lerp_type='metric',oversampling_ratio=30):
	
	zs = model.encode(x)
	ze = np.copy(zs)

	x_lerps 	= []
	lerp_ssims 	= []

	for dim in dims:
		print('#####Dim %d/%d####' % (dim,model.latent_dim))
		ze[0,dim] += bound

		if lerp_type =='metric':
			z_lerp,x_lerp,_,lerp_ssim = metric_interpolation(zs,ze,model,metrics.ssim,
				x_end=x,
				dim=dim,
				verbose=True,
				oversampling_ratio=oversampling_ratio)
			lerp_ssim = np.array(lerp_ssim).reshape(1,-1)
		else:
			z_lerp 		= naive_interpolation(zs,ze,dim=dim)
			x_lerp 		= model.generate(z_lerp)
			lerp_ssim 	= np.array([metrics.ssim(model.sess,xi.reshape(1,512,2),x)[0] for xi in x_lerp])
		
		x_lerps.append(x_lerp.reshape(-1,x_lerp.shape[0],x_lerp.shape[1],x_lerp.shape[2]))
		lerp_ssims.append(lerp_ssim.reshape(-1,lerp_ssim.shape[0],lerp_ssim.shape[1]))		

	return np.vstack(x_lerps),np.vstack(lerp_ssims)


# if __name__ == '__main__':
# 	dir_model = 'save/e_5'
# 	tag = dir_model.split('/')[1]
# 	dir_eval = 'evaluation'
# 	if not os.path.isdir(dir_eval):
# 			os.mkdir(dir_eval)

# 	settings 	= json.load(open('%s/settings.json' % dir_model, 'r'))
# 	settings['savedir'] = dir_model	

# 	#fixes for compatibility
# 	if not 'beta' in settings:
# 		settings['beta'] = 1

# 	if not 'nfilt' in settings:
# 		settings['nfilt'] = settings['nfilt_init']

# 	model = logan.LoGAN(settings,True)
	
	
# 	_,x_test,x_mini = utils.load_data()

# 	x1 = utils.template_step_down(300,0.5)
# 	x2 = x_mini[4].reshape(-1,512,2)

# 	z1 	= model.encode(x1)
# 	z2 	= model.encode(x2)

	
# 	#mlerp objective with ssim
# 	r1 	 = model.generate(z1)
# 	r2 	 = model.generate(z2)
# 	ims	 = np.vstack([x1,r1,x2,r2])
# 	sims = np.array([metrics.ssim(model.sess,x.reshape(1,512,2),x2)[0] for x in ims])
# 	title_text = np.array(['Source $x_1$ (SSIM-%0.3f)' % sims[0], 
# 		'Reconstruction of $x_1$ (SSIM-%0.3f)' % sims[1],
# 		'Target $x_2$ (SSIM-%0.3f)' % sims[2], 
# 		'Reconstruction $x_2$ (SSIM-%0.3f)' % sims[3]])
# 	utils.plotGrid(ims.reshape(2,2,512,2),
# 		'%s/%s_1_mlerp_objective.png' % (dir_eval,tag),
# 		title_text=title_text.reshape(2,2,1))

# 	#mlerp vs lerp
# 	z_lerp = naive_interpolation(z1,z2)
# 	x_lerp = model.generate(z_lerp)
# 	lerp_ssim = np.array([metrics.ssim(model.sess,x.reshape(1,512,2),x2)[0] for x in x_lerp])

# 	print('MLERP with SSIM-linear')
# 	_,x_mlerp,_,mlerp_ssim = metric_interpolation(z1,z2,model,x_end=x2,
# 	metric_fn=metrics.ssim,
# 	oversampling_ratio=15)

# 	print('MLERP with SSIM-sigmoid')
# 	_,x_mlerp_sig,_,mlerp_ssim_sig = metric_interpolation(z1,z2,model,x_end=x2,
# 		metric_fn=metrics.ssim,
# 		m_trajectory='sigmoid',
# 		oversampling_ratio=30)		
	
# 	labels = ['lerp_ssim','mlerp_linear','mlerp_sigmoid']
# 	utils.plot_interpolation_summary(lerp_ssim.reshape(-1),
# 		mlerp_ssim.reshape(-1),
# 		mlerp_ssim_sig.reshape(-1),
# 		labels,
# 		'%s/%s_2_lerp_v_mlerp.png' % (dir_eval,tag))

# 	sequences = np.stack([x_lerp,x_mlerp,x_mlerp_sig])
# 	sequences = np.swapaxes(sequences,0,1)
# 	utils.plotGrid(sequences,'%s/%s_3_lerp_v_mlerp_sequences.png' % (dir_eval,tag),
# 		size=(15,15),
# 		legends='off',
# 		axis='off')

# 	#composite metric
# 	print('MLERP with roughness')
# 	_,x_mlerp_rough,_,mlerp_rough = metric_interpolation(z1,z2,model,x_end=x2,
# 	metric_fn=metrics.roughness,
# 	oversampling_ratio=150)

# 	print('MLERP with ssim-roughness')
# 	_,x_mlerp_comp,_,mlerp_comp = metric_interpolation(z1,z2,model,x_end=x2,
# 	metric_fn=metrics.ssim_roughness,
# 	oversampling_ratio=150)

# 	labels = ['mlerp_ssim','mlerp_roughness','mlerp_composite']
# 	utils.plot_interpolation_summary(mlerp_ssim.reshape(-1),
# 		mlerp_rough.reshape(-1),
# 		mlerp_comp.reshape(-1),
# 		labels,
# 		'%s/%s_4_mlerp_composite.png' % (dir_eval,tag))

# 	#neighborhood search
# 	print('Neighborhood search')
# 	lerp_neighbors,lerp_ssim_neighbors = neighborhood_interpolation(x1,
# 		model,np.arange(model.latent_dim),lerp_type='linear')

# 	mlerp_neighbors,mlerp_ssim_neighbors = neighborhood_interpolation(x1,
# 		model,np.arange(model.latent_dim),lerp_type='metric',oversampling_ratio=500)

# 	utils.plot_neighborhood_summary(lerp_ssim_neighbors.reshape(10,15), 
# 		mlerp_ssim_neighbors.reshape(10,15),
# 		'%s/%s_5_neighborhood_search.png' % (dir_eval,tag),
# 		legends=True)