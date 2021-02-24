import numpy as np
from skimage.measure import compare_ssim
from tslearn.metrics import dtw as compare_dtw
from scipy.stats import beta, rv_histogram
from scipy.integrate import quad

def get_dtw(ref, recon, tran):
  dtw_recon  = batch_dtw(ref, recon, average=True)
  dtw_tran   = batch_dtw(ref, tran, average=True)
  return dtw_recon, dtw_tran

# def get_ssim(ref, recon, tran):
#   ssim_recon = batch_ssim(ref, recon, average=True) 
#   ssim_tran  = batch_ssim(ref, tran, average=True)  
#   return ssim_recon, ssim_tran

def get_ssim(ref, recon):
  ssim_recon = batch_ssim(ref, recon, average=True)   
  return ssim_recon

def continuity(XJ1, XJ2, XJ3, normalize=False):  
  cj = 0
  for XJ in [XJ1, XJ2, XJ3]:
    if not isinstance(XJ, np.ndarray):
      XJ = XJ.numpy()
    len = XJ1.shape[-1]    
    cjn = np.abs(XJ[:, :, :, (len//2)] - XJ[:, :, :, (len//2)-1])
    cjd = np.amax(np.abs(np.diff(XJ, axis=3)), axis=3) if normalize else 1   
    cj += np.mean(cjn/cjd)
  return cj/3

def ssim_metric(x_real, x_recon, len=512):  
    n_signals = x_real.shape[1]
    x_real  = x_real.reshape(-1, len, n_signals)
    x_recon = x_recon.reshape(-1, len, n_signals)
    return compare_ssim(x_real, x_recon, multichannel=True)

def batch_ssim(x_real, x_recon, average=False): 
    
    if not isinstance(x_real, np.ndarray):
        x_real = x_real.numpy()

    if not isinstance(x_recon, np.ndarray):
        x_recon = x_recon.numpy()

    x_real  = np.swapaxes(x_real, 1, 2)
    x_recon = np.swapaxes(x_recon, 1, 2) 

    ssims = np.array(
        [compare_ssim(ref, rec, multichannel=True) for ref, rec in zip(x_real, x_recon)])

    if average:
        return ssims.mean()
    else:
        return ssims
    
def batch_L1(x_real, x_recon, average=False):
  if not isinstance(x_real, np.ndarray):
        x_real = x_real.numpy()

  if not isinstance(x_recon, np.ndarray):
      x_recon = x_recon.numpy()

  x_real  = np.swapaxes(x_real, 1, 2)
  x_recon = np.swapaxes(x_recon, 1, 2) 

  ssims = np.array(
      [np.mean(np.abs(ref - rec)) for ref, rec in zip(x_real, x_recon)])

  if average:
      return ssims.mean()
  else:
      return ssims

def batch_dtw(x_real, x_recon, average=False):
    if not isinstance(x_real, np.ndarray):
        x_real = x_real.numpy()

    if not isinstance(x_recon, np.ndarray):
        x_recon = x_recon.numpy()

    x_real  = np.swapaxes(x_real, 1, 2)
    x_recon = np.swapaxes(x_recon, 1, 2) 

    dtws = np.array(
        [compare_dtw(ref, rec) for ref, rec in zip(x_real, x_recon)])

    if average:
        return dtws.mean()
    else:
        return dtws

def _ssim_metric(x_real, x_recon, len=512):
    n_signals = x_real.shape[1]
    x_real  = x_real.reshape(len, n_signals)
    x_recon = x_recon.reshape(len, n_signals)
    return compare_ssim(x_real, x_recon, multichannel=True)

def dtw_metric(x_real, x_recon, len=512):
    n_signals = x_real.shape[1]
    x_real  = x_real.reshape(-1, len, n_signals)
    x_recon = x_recon.reshape(-1, len, n_signals)
    
    running_dtw = 0.
    for x_r, x_f in zip(x_real, x_recon):       
        running_dtw += compare_dtw(x_r, x_f)
    return running_dtw/x_real.shape[0]


def batch_exp_diversity(exp):
	exp = exp.permute(1, 0, 2, 3).numpy()
	div_s = []
	div_l = []

	for n, styles in enumerate(exp):
		styles = np.transpose(styles, (0, 2, 1))
		num_series = styles.shape[0]
		num_elems = int(num_series*(num_series - 1)/2)
		store_s = np.zeros(num_elems)
		store_l1 = np.zeros(num_elems)
		t = 0	  
		for i in range(num_series - 1):
			for j in range(i + 1, num_series):				
				s  = compare_ssim(styles[i], styles[j], multichannel = True)
				l1 = np.mean(np.abs(styles[i]- styles[j]))
				store_s[t]  = s
				store_l1[t] = l1
				t = t+1		
		div_s.append(store_s.mean())
		div_l.append(store_l1.mean())

	return np.mean(div_l)
	# return np.stack([div_s, div_l], axis=1)


## Mean of SSIM of all signals
## Lower value is better
def batch_trn_diversity(ref, trn):

	## Args:
	## styles: A batch with signals of the form (batch_size, num_channels, len_of_signal)
	## transpose_signal: To reshape the signal into the form (batch_size, len_of_signal, num_channels)
				## for SSIM.

	## Similar method might be used for images in https://arxiv.org/pdf/1802.03446.pdf
	## "Odena et al. [2] used 9 MS-SSIM to evaluate the diversity of generated
	## images. The intuition is that image pairs with higher MS-SSIM seem more
	## similar than pairs with lower MS-SSIM. They measured the MS-SSIM
	## scores of 100 randomly chosen pairs of images within a given class. The
	## higher (lower) diversity within a class, the lower (the higher) mean MSSSIM score"

	#input shape (num_styles, batch_size, n_sig, seq_len)
	trn = trn.permute(1, 0, 2, 3).numpy()
	ref   = ref.numpy()
	div_s = []
	div_l = []

	for n, styles in enumerate(trn):
		if styles.shape[0] > 10:
			styles = styles[0 : 10]
		styles = np.transpose(styles, (0, 2, 1))
		t_sim = compare_ssim(styles[0], ref[n].transpose(1, 0), multichannel = True)		
		if t_sim < 0.5:
			continue	  
		styles = styles[1:]

		num_series = styles.shape[0]
		num_elems = int(num_series*(num_series - 1)/2)
		store_s = np.zeros(num_elems)
		store_l1 = np.zeros(num_elems)
		t = 0	  
		for i in range(num_series - 1):
			for j in range(i + 1, num_series):
				s  = compare_ssim(styles[i], styles[j], multichannel = True)
				l1 = np.mean(np.abs(styles[i]- styles[j]))
				store_s[t]  = s
				store_l1[t] = l1
				t = t+1			
		if store_s.mean() > 0.5:
			div_s.append(store_s.mean())
			div_l.append(store_l1.mean())

	return np.mean(div_l)
	# return np.stack([div_s, div_l], axis=1)

def _batch_diversity(ref, tran , average=False):

  num_styles = tran.shape[0]-1
  batch_size = ref.shape[0]
  p_tran = tran[0]
  tran   = tran[1:]

  sim_tran = batch_ssim(ref, p_tran)  
  sim = np.array([batch_ssim(ref, tran[i]) 
    for i in range(num_styles)]).transpose()
  div = np.array([hell_ssim(sim[i], sim_tran[i]) 
    for i in range(batch_size)])

  if average:
    return div.mean()
  else:
    return div


## Obtain the beta parameters from the desired peak and variance
def obtain_beta_params(ssim_mean, var = 0.01, mode = 'mode'):

  m = ssim_mean
  v = var
  if mode == 'mode':
   
    a = 2*m - 1
    b = 3*m - 2
    c = (m-1)**2


    p0 = v*a**2*b
    p1 = -((2*m*c - c) + v*(a**2 + 2*a*b))
    p2 = (m*c + v*(2*a+b))
    p3 = -v
    roots = np.roots([p3,p2,p1,p0])
    beta_ind = np.where(roots > 1)[0][0]
    beta_param = roots[beta_ind]
    alpha_param = (2*m - m*beta_param - 1)/(m - 1)
  elif mode == 'mean':
   
    beta_param = (m*(1-m)**2 - v + m*v)/v
    alpha_param = m*beta_param/(1-m)

  return alpha_param, beta_param
  

## We define the Hellinger distance here
def hell_dist(x,y):
  dist = np.sum((np.sqrt(x) - np.sqrt(y))**2)
  dist = 0.5*np.sqrt(dist)
  return dist

## We work with the beta distribution as the desired target distribution
def hell_ssim(ssim_sample, target_mean, return_prob=False):

  ## ssim_sample is sample of ssim values of generated samples
  ## target_mean is where we want to place the mode or the mean
  try:
    if np.abs(target_mean - 1) < 1e-5:
      ssim_sample = ssim_sample - np.random.normal(0,0.01, ssim_sample.shape[0])**2
      target_mean = np.mean(ssim_sample)


    beta_param1, beta_param2 = obtain_beta_params(target_mean, mode = 'mode')

    ## The bins which we subdivide the distribution into
    bins = np.linspace(0, 1, 15).tolist()

    beta_func = lambda x: beta.pdf(x, beta_param1, beta_param2)
    prob_list = []
    for edge1, edge2 in zip(bins[0:-1], bins[1:]):
      prob,_ = quad(beta_func, edge1, edge2)
      prob_list.append(prob)
    prob_beta = np.array(prob_list)
    hist = np.histogram(ssim_sample, bins=bins)
    hist_dist = rv_histogram(hist)
    prob_list_ssim = []
    for edge1, edge2 in zip(bins[0:-1], bins[1:]):
      prob,_ = quad(hist_dist.pdf, edge1, edge2)
      prob_list_ssim.append(prob)
    prob_ssim = np.array(prob_list_ssim)
    
    ## Hellinger distance here
    if return_prob:
      return hell_dist(prob_beta, prob_ssim), prob_beta, prob_ssim
    else: 
      return hell_dist(prob_beta, prob_ssim)
  except:
    return -1. ## To see that somthing went wrong



if __name__ == '__main__':
    x_real = np.load('./r2s/logan/test_mini.npy').astype(np.float32)
    x_real = x_real[:, :, 0].reshape(-1, 1, 512)
    # x_recon = np.load('test_mini_dec.npy').astype(np.float32)
    dtw = dtw_metric(x_real, x_real)
    # ssim = ssim_metric(x_real, x_recon)
