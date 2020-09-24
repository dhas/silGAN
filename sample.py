import torch
import numpy as np
import mlerp
import utils
from network import Generator, Discriminator


## loading pretrained generator model
pretrained_G = 'models/saved_logan_6292/G_epoch0025.pt'
#hardcoding settings for LoGAN1.2 
G = Generator(10, n_feat=64, n_layers=4)
G.load_state_dict(torch.load(pretrained_G))	

## loading mini test_set
x = np.load('data/test_mini.npy').swapaxes(1,2)
utils.plotGrid(x.reshape(5, 3, 2, 512), 'generated/1_test_mini.png')

x = torch.from_numpy(x).float()
x_rec = G.reconstruct(x).detach()

utils.plotGrid(x_rec.numpy().reshape(5, 3, 2, 512), 'generated/2_test_mini_reconstructed.png')

z = torch.randn(x.shape[0], 10)
x_hat = G.generate(z).detach()
utils.plotGrid(x_hat.numpy().reshape(5, 3, 2, 512), 'generated/0_random_samples.png')

x1 = torch.from_numpy(utils.template_step_down(300,0.5)).float()
x2 = x[4:5]

x1_rec = G.reconstruct(x1).detach()
x2_rec = G.reconstruct(x2).detach()
ims	 = np.vstack([x1, x1_rec, x2, x2_rec])
sims = np.array([utils.ssim_metric(x2[0].numpy(), x) for x in ims])
title_text = np.array(['Source $x_1$ (SSIM-%0.3f)' % sims[0], 
	'Reconstruction of $x_1$ (SSIM-%0.3f)' % sims[1],
	'Target $x_2$ (SSIM-%0.3f)' % sims[2], 
	'Reconstruction $x_2$ (SSIM-%0.3f)' % sims[3]])
utils.plotGrid(ims.reshape(2, 2, 2, 512),
	'generated/3_template_and_real_reconstructions.png',
	title_text=title_text.reshape(2,2,1))

z1 = G.encode(x1).detach()
z2 = G.encode(x2).detach()
z_lerp = mlerp.naive_interpolation(z1,z2)
x_lerp = G.generate(torch.from_numpy(z_lerp)).detach().numpy()
lerp_ssim = np.array([utils.ssim_metric(x2[0].numpy(), x) for x in x_lerp])


z_mlerp = mlerp.metric_interpolation(z1, z2, G, x_end=x2.numpy(),
metric_fn=utils.ssim_metric, oversampling_ratio=15)
x_mlerp = G.generate(torch.from_numpy(z_mlerp)).detach().numpy()
mlerp_ssim = np.array([utils.ssim_metric(x2[0].numpy(), x) for x in x_mlerp])


z_mlerp_sigmoid = mlerp.metric_interpolation(z1, z2, G, x_end=x2.numpy(),
metric_fn=utils.ssim_metric, oversampling_ratio=15, m_trajectory='sigmoid')
x_mlerp_sigmoid = G.generate(torch.from_numpy(z_mlerp_sigmoid)).detach().numpy()
mlerp_sigmoid_ssim = np.array([utils.ssim_metric(x2[0].numpy(), x) for x in x_mlerp_sigmoid])

sequences = np.stack([x_lerp, x_mlerp, x_mlerp_sigmoid])
sequences = np.swapaxes(sequences, 0, 1)
utils.plotGrid(sequences,'generated/4_lerp_v_mlerp_sequences.png',
	size=(15,15),
	legends='off',
	axis='off')

labels = ['lerp','mlerp_linear','mlerp_sigmoid']
utils.plot_interpolation_summary(lerp_ssim.reshape(-1),
	mlerp_ssim.reshape(-1),
	mlerp_sigmoid_ssim.reshape(-1),
	labels,
	'generated/5_lerp_v_mlerp_ssim.png')