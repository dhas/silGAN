import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim

def ssim_metric(x_real, x_recon):
	if len(x_real.shape) > 2:
		ssims = []
		for x, x_r in zip(x_real, x_recon):
			ssims.append(compare_ssim(x.reshape(-1,2), 
				x_r.reshape(-1,2), multichannel=True))
		return ssims
	else:
		x_real  = x_real.reshape(-1, 2)
		x_recon = x_recon.reshape(-1, 2)
		return compare_ssim(x_real, x_recon, multichannel=True)


def template_step_down(crs,level):
	x_scen = np.zeros((1, 2, 512))
	x_scen[:, :, :crs] = level	
	return x_scen

def plotGrid(series, fname, channels_first=True, 
	size=(20,10),axis='on',legends='on',
	legend_font_size=14,title_text=None,fig_title=None,tight=True):
	if len(series.shape) < 4:
		raise Exception('Series format (rows, cols, points, signals)')

	r = series.shape[0]
	c = series.shape[1]

	# ymin,ymax = 0.0,1.0

	fig, axs = plt.subplots(r, c,figsize=size)
	if r > 1:
		l_ax = axs[0,c-1]
	else:
		l_ax = axs[c-1]
	for i in range(r):
		for j in range(c):
			s 	= series[i,j]
			if r > 1:	    		
				a1 	= axs[i,j]
			else:	    
				a1 	= axs[i+j]
			a2 	= a1.twinx()
			if not (title_text is None):
				a1.title.set_text(title_text[i,j,0])
			a1.set_ylim(0, 1)
			a2.set_ylim(0, 1)
			if channels_first == True:
				v = s[0]
				e = s[1]
			else:
				v = s[:,0]
				e = s[:,1]
			a1.plot(v, color='blue',label='vehicle_speed')
			a2.plot(e, color='green',label='engine_speed')
			if axis == 'off':
				a1.axis('off')
				a2.axis('off')
			else:
				a2.axis('off')
	if legends == 'on':
		h1, l1 = a1.get_legend_handles_labels()
		h2, l2 = a2.get_legend_handles_labels()
		handles = h1 + h2
		labels 	= l1 + l2		
		# fig.legend(handles, labels, bbox_to_anchor=[0.9, 0.97], loc='upper right')
		l_ax.legend(handles, labels,prop={'size': legend_font_size})
	if not (fig_title is None):
		fig.suptitle(fig_title)	
	
	fig.savefig(fname)

def plot_interpolation_summary(lerp_metric,mlerp_1_metric,mlerp_2_metric,labels,savename):	
	fig,ax = plt.subplots(figsize=(20,10))
	ax.plot(lerp_metric,color='orange',label=labels[0])
	ax.plot(mlerp_1_metric,color='blue',label=labels[1])
	ax.plot(mlerp_2_metric,color='green',label=labels[2])	
	h, l = ax.get_legend_handles_labels()
	ax.legend(handles=h)
	fig.savefig(savename, bbox_inches='tight')
	plt.close()