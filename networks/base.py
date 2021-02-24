import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torchsummary import summary

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)


def param_count(model):
	sum = 0
	for c in model.children():
		for p in c.parameters():
			sum += p.numel()
	return sum

class LambdaLR:
	def __init__(self, n_epochs, offset, decay_start_epoch):
		assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
		self.n_epochs = n_epochs
		self.offset = offset
		self.decay_start_epoch = decay_start_epoch

	def step(self, epoch):
		return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

##############################
#       Custom Blocks
##############################
class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args

	def forward(self, x):
		return x.view(self.shape)

class ResidualBlock1D(nn.Module):
	def __init__(self, features, norm="in"):
		super(ResidualBlock1D, self).__init__()

		norm_layer = AdaptiveInstanceNorm1d if norm == "adain" else nn.InstanceNorm1d

		self.block = nn.Sequential(
			nn.ReflectionPad1d(1),
			nn.Conv1d(features, features, 3),
			norm_layer(features),
			nn.ReLU(inplace=True),
			nn.ReflectionPad1d(1),
			nn.Conv1d(features, features, 3),
			norm_layer(features),
		)

	def forward(self, x):
		return x + self.block(x)


class ResidualBlock2D(nn.Module):
	def __init__(self, features, norm="in"):
		super(ResidualBlock2D, self).__init__()

		norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d

		self.block = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(features, features, 3),
			norm_layer(features),
			nn.ReLU(inplace=True),
			nn.ReflectionPad2d(1),
			nn.Conv2d(features, features, 3),
			norm_layer(features),
		)

	def forward(self, x):
		return x + self.block(x)


##############################
#        Custom Layers
##############################


class AdaptiveInstanceNorm2d(nn.Module):
	"""Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

	def __init__(self, num_features, eps=1e-5, momentum=0.1):
		super(AdaptiveInstanceNorm2d, self).__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		# weight and bias are dynamically assigned
		self.weight = None
		self.bias = None
		# just dummy buffers, not used
		self.register_buffer("running_mean", torch.zeros(num_features))
		self.register_buffer("running_var", torch.ones(num_features))

	def forward(self, x):
		assert (
			self.weight is not None and self.bias is not None
		), "Please assign weight and bias before calling AdaIN!"
		b, c, h, w = x.size()
		running_mean = self.running_mean.repeat(b)
		running_var = self.running_var.repeat(b)

		# Apply instance norm
		x_reshaped = x.contiguous().view(1, b * c, h, w)

		out = F.batch_norm(
			x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
		)

		return out.view(b, c, h, w)

	def __repr__(self):
		return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
	def __init__(self, num_features, eps=1e-5, affine=True):
		super(LayerNorm, self).__init__()
		self.num_features = num_features
		self.affine = affine
		self.eps = eps

		if self.affine:
			self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
			self.beta = nn.Parameter(torch.zeros(num_features))

	def forward(self, x):
		shape = [-1] + [1] * (x.dim() - 1)
		mean = x.view(x.size(0), -1).mean(1).view(*shape)
		std = x.view(x.size(0), -1).std(1).view(*shape)
		x = (x - mean) / (std + self.eps)

		if self.affine:
			shape = [1, -1] + [1] * (x.dim() - 2)
			x = x * self.gamma.view(*shape) + self.beta.view(*shape)
		return x

##############################
#        Base networks
##############################
class UpsampleBlock(nn.Module):
	def __init__(self, in_feat, out_feat, scale, op='1D', k_size=5, norm='LN'):
		super(UpsampleBlock, self).__init__()
		conv_layer = nn.Conv1d if op == '1D' else nn.Conv2d
		norm_layer = LayerNorm if norm =='LN' else (nn.BatchNorm1d if op=='1D' else nn.BatchNorm2d)
		self.block = nn.Sequential(*[
			nn.Upsample(scale_factor=scale),
			conv_layer(in_feat, out_feat, k_size, stride=1, padding=k_size//2),
			norm_layer(out_feat),
			nn.ReLU(inplace=True)
		])

	def forward(self, f):
		return self.block(f)

class DownsampleBlock(nn.Module):
	def __init__(self, in_feat, out_feat, op='1D', k_size=4):
		super(DownsampleBlock, self).__init__()
		conv_layer = nn.Conv1d if op == '1D' else nn.Conv2d
		norm_layer = nn.InstanceNorm1d if op == '1D' else nn.InstanceNorm2d
		self.block= nn.Sequential(*[
			conv_layer(in_feat, out_feat, k_size, stride=2, padding=1),
			norm_layer(out_feat),
			nn.ReLU(inplace=True)
		])

	def forward(self, f):
		return self.block(f)

class Encoder1(nn.Module):
	def __init__(self, in_channels, n_class=2, dim=16, n_residual=1, n_downsample=2):
		super(Encoder1, self).__init__()

		self.in_channels = in_channels		
		# Initial convolution block
		layers = [
			nn.ReflectionPad1d(3),
			nn.Conv1d(in_channels, dim//2, 7),
			nn.InstanceNorm1d(dim//2),
			nn.ReLU(inplace=True),
		]

		self.emb_x = nn.Sequential(*layers)

		layers = [
			nn.Linear(n_class, 2*32),
			Reshape(-1, 2, 32),
			UpsampleBlock(in_feat=2, out_feat=dim//2, scale=2),
			UpsampleBlock(in_feat=dim//2, out_feat=dim//2, scale=2),
			UpsampleBlock(in_feat=dim//2, out_feat=dim//2, scale=2),
			UpsampleBlock(in_feat=dim//2, out_feat=dim//2, scale=2),
		]
		self.embl_1 = nn.Sequential(*layers)

		# Downsampling
		layers = []
		for _ in range(n_downsample):
			layers += [
				DownsampleBlock(in_feat=dim, out_feat=dim*2)				
			]
			dim *= 2

		# Residual blocks
		for _ in range(n_residual):
			layers += [ResidualBlock1D(dim, norm="in")]

		self.model = nn.Sequential(*layers)


	def forward(self, x, l):
		ind = 2 + int(math.log(x.shape[-1]//32, 2))
		l = self.embl_1[:ind](l)
		x = self.emb_x(x)
		c1 = torch.cat((x, l), 1)
		c1 = self.model(c1)
		return c1		

class Decoder1(nn.Module):
	def __init__(self, out_channels, dim=16, n_residual=1, n_upsample=2, k_size=5):
		
		super(Decoder1, self).__init__()
		self.out_channels = out_channels		
		self.n_upsample = n_upsample
		
		dim = dim * 2 ** n_upsample

		layers = [
			UpsampleBlock(1, dim, scale=1),
			UpsampleBlock(dim, dim, scale=2),
			UpsampleBlock(dim, dim, scale=2),
			UpsampleBlock(dim, dim, scale=2),
			UpsampleBlock(dim, dim, scale=2),
		]
		self.embl_1 = nn.Sequential(*layers)

		# Upsampling
		layers = []
		for _ in range(n_residual):
			layers += [ResidualBlock1D(dim, norm="in")]

		layers += [UpsampleBlock(in_feat=dim, out_feat=dim//2, scale=1)]
		for i in range(n_upsample-1):			
			dim = dim//2			
			layers += [
				UpsampleBlock(in_feat=dim, out_feat=dim//2, scale=2)
			]			

		# Output layer
		layers += [nn.Conv1d(dim//2, out_channels, 7, padding=7//2), nn.Sigmoid()]
		self.model = nn.Sequential(*layers)		

	def forward(self, c1, l):
		ind = int(math.log(c1.shape[-1], 2))
		l = self.embl_1[:ind](l)
		c1 = torch.cat((c1, l), 2)
		x = self.model(c1)
		return x

class Encoder2(nn.Module):
	def __init__(self, in_channels=2, n_class=2, dim=16, n_residual=1, n_downsample=2):
		super(Encoder2, self).__init__()

		in_channels = in_channels + 1
		self.n_class = n_class

		layers = [
			nn.Linear(n_class, 1*32),
			Reshape(-1, 1, 32),
			UpsampleBlock(in_feat=1, out_feat=1, scale=4),
			UpsampleBlock(in_feat=1, out_feat=1, scale=2),
			UpsampleBlock(in_feat=1, out_feat=1, scale=2)			
		]
		self.embl_1 = nn.Sequential(*layers)

		# Initial convolution block
		layers = [
			nn.ReflectionPad1d(3),
			nn.Conv1d(in_channels, dim, 7),
			nn.InstanceNorm1d(dim),
			nn.ReLU(inplace=True),
		]

		# Downsampling
		for _ in range(n_downsample):
			layers += [
				DownsampleBlock(in_feat=dim, out_feat=dim*2)				
			]
			dim *= 2

		# Residual blocks
		for _ in range(n_residual):
			layers += [ResidualBlock1D(dim, norm="in")]

		layers += [UpsampleBlock(in_feat=dim, out_feat=dim*2, scale=1)]		
		self.encoder_1 = nn.Sequential(*layers)
		self.encoder_2 = nn.Sequential(*layers)
		
		layers = [
			UpsampleBlock(dim, dim, scale=2),
			nn.Conv1d(dim, 1, 5, stride=1, padding=2),
			nn.BatchNorm1d(1),
			nn.ReLU(inplace=True),
			# nn.Linear(fsz, z_len)
		]
		self.rev_embed_c = nn.Sequential(*layers)

	def forward(self, x, l):
		ind = 1 + int(math.log(x.shape[-1]//32, 2))		
		l = self.embl_1[:ind](l)		
		x = torch.cat((x, l), dim=1)
		c = self.encoder_1(x)
		c1, c2 = torch.split(c, c.shape[1]//2, dim=1)
		c2 = self.rev_embed_c(c2)
		c2, l = torch.split(c2, c2.shape[-1]-self.n_class, dim=2)		
		return c1, l, c2

class Decoder2(nn.Module):
	def __init__(self, out_channels=2, z_len=64, dim=32, n_residual=1, n_upsample=2):
		super(Decoder2, self).__init__()
				
		dim = 2*dim * 2 ** n_upsample		
		
		layers = [
			DownsampleBlock(in_feat=1, out_feat=1),
			UpsampleBlock(in_feat=1, out_feat=dim//4, scale=1, norm='BN'),
			UpsampleBlock(in_feat=dim//4, out_feat=dim//2, scale=1, norm='BN'),
		]
		self.emb_c = nn.Sequential(*layers)

		layers = []
		for _ in range(n_residual):
			layers += [ResidualBlock1D(dim, norm="in")]

		# Upsampling
		for _ in range(n_upsample):
			layers += [
				UpsampleBlock(in_feat=dim, out_feat=dim//2, scale=2)				
			]
			dim = dim // 2

		# Output layer
		layers += [nn.Conv1d(dim, out_channels, 7, padding=7//2), nn.Sigmoid()]
		self.decoder = nn.Sequential(*layers)

	def forward(self, c1, l, c2):
		c2 = torch.cat((c2, l), dim=2)
		c2 = self.emb_c(c2)		
		c  = torch.cat((c1, c2), dim=1)		
		x = self.decoder(c)
		return x


class Decoder3(nn.Module):
	def __init__(self, out_channels=2, seq_len=1024, dim=16, n_residual=1, n_upsample=2):
		super(Decoder3, self).__init__()

		dim = dim * 2 ** n_upsample
		self.out_channels = out_channels
		self.seq_len = seq_len

		layers = [
			UpsampleBlock(in_feat=1, out_feat=dim//2, scale=1),
			UpsampleBlock(in_feat=dim//2, out_feat=dim, scale=1),	
		]
		self.emb_c  = nn.Sequential(*layers)
		
		layers = [
			UpsampleBlock(in_feat=3, out_feat=dim, scale=1),
			UpsampleBlock(in_feat=dim, out_feat=dim, scale=3)
		]
		self.emb_x  = nn.Sequential(*layers)		


		layers = []		
		for _ in range(n_residual):
			layers += [ResidualBlock2D(dim, norm="in")]		
		for _ in range(n_upsample):
			layers += [
				UpsampleBlock(in_feat=dim, out_feat=dim//2, scale=1, op='2D')				
			]
			dim = dim // 2		
		layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Sigmoid()]
		self.decoder = nn.Sequential(*layers)

	def reshape(self, x, c):	
		if x.shape[-1] == 384:
			sz = (x.shape[0], -1, 16, 24)
		elif x.shape[-1] == 256:
			sz = (x.shape[0], -1, 16, 16)
		elif x.shape[-1] == 512:
			sz = (x.shape[0], -1, 32, 16)
		elif x.shape[-1] == 768:
			sz = (x.shape[0], -1, 32, 24)
		x = x.view(sz)
		c = c.view(sz)
		return x, c	

	def expand(self, x, c3, p):
		f = self.forward(x, c3)		
		f1, f2 = torch.split(f, f.shape[-1]//2, dim=2)
		x = torch.cat((f1, x, f2), dim=2)
		x = x[:, :, p:p+self.seq_len]
		return x, f

	def forward(self, x, c3):
		c3 = self.emb_c(c3)
		x  = self.emb_x[0:1](x) if x.shape[-1]==self.seq_len//2 else self.emb_x(x)		
		x, c3 = self.reshape(x, c3)			
		f  = torch.cat((x, c3), dim=2)				
		f  = self.decoder(f)
		f  = f.view(-1, self.out_channels, f.shape[-1]*f.shape[-2])		
		return f


	
class Encoder3(nn.Module):
	def __init__(self, in_channels, seq_len=1024, dim=16, n_residual=1, n_downsample=2):
		super(Encoder3, self).__init__()

		dim = dim * 2 ** n_downsample

		self.in_channels = in_channels
		
		layers = [DownsampleBlock(in_feat=in_channels, out_feat=dim)]
		self.emb_x = nn.Sequential(*layers)
		
		layers = []			
		for _ in range(n_downsample):
			layers += [
				UpsampleBlock(in_feat=dim, out_feat=dim//2, scale=1, op='2D'),
			]
			dim = dim//2

		layers += [UpsampleBlock(in_feat=dim, out_feat=1, scale=1, op='2D'),]
		self.model = nn.Sequential(*layers)

	

	def reshape(self, x):
		if x.shape[-1] == 384:
			sz = (x.shape[0], -1, 16, 24)
		elif x.shape[-1] == 256:
			sz = (x.shape[0], -1, 16, 16)
		elif x.shape[-1] == 512:
			sz = (x.shape[0], -1, 32, 16)
		elif x.shape[-1] == 768:
			sz = (x.shape[0], -1, 32, 24)
		x = x.view(sz)		
		return x

	def forward(self, x):
		x = self.emb_x(x)		
		c3 = self.model(self.reshape(x))
		c3 = c3.view(-1, 1, c3.shape[-2]*c3.shape[-1])
		return c3


def discriminator_block(in_filters, out_filters, normalize=True):
	"""Returns downsampling layers of each discriminator block"""
	layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
	if normalize:
		layers.append(nn.InstanceNorm2d(out_filters))
	layers.append(nn.LeakyReLU(0.2, inplace=True))
	return layers


class Discriminator(nn.Module):
	def __init__(self, in_channels=2, n_dis=3):
		super(Discriminator, self).__init__()

		# Extracts three discriminator models
		self.models = nn.ModuleList()
		for i in range(n_dis):
			self.models.add_module(
				"disc_%d" % i,
				nn.Sequential(
					*discriminator_block(in_channels, 64, normalize=False),
					*discriminator_block(64, 128),
					*discriminator_block(128, 256),
					# *discriminator_block(256, 512),
					nn.Conv2d(256, 1, 3, padding=1)
				),
			)

		# self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)
		self.downsample = nn.AvgPool2d(in_channels, stride=2, count_include_pad=False)
		self.in_channels = in_channels

	def compute_loss(self, x, gt):
		"""Computes the MSE between model output and scalar gt"""		
		loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
		return loss

	def sim(self, x1, x2, crop_len=None):
		if not crop_len is None:
			l = x1.shape[-1]
			p = torch.randint(0, l-crop_len, (1,))
			x1 = x1[:, :, p:p+crop_len]
			x2 = x2[:, :, p:p+crop_len]
		return -1*torch.mean(torch.abs(self.fmap(x1) - self.fmap(x2)))

	def fmap(self, x):
		sz = self.get_new_shape(x)
		x = x.view(-1, self.in_channels, sz[0], sz[1])
		m = next(iter(self.models))
		return m[:4](x)

	def get_new_shape(self, x):
		if x.shape[-1] == 64:
			sz = (8, 8)
		elif x.shape[-1] == 128:
			sz = (16, 8)
		elif x.shape[-1] == 256:
			sz = (16, 16)
		elif x.shape[-1] == 512:
			sz = (32, 16)
		elif x.shape[-1] == 1024:
			sz = (32, 32)
		return sz
	
	def _forward(self, x):
		sz = self.get_new_shape(x)			
		x = x.view(-1, self.in_channels, sz[0], sz[1])		
		outputs = []
		for i, m in enumerate(self.models):
			outputs.append(m(x))			
			x = self.downsample(x)			
			if x.shape[-2] == 1 or x.shape[-1] == 1:
				break
		return outputs

	def reshape(self, x):
		if x.shape[-1] == 64:
			sz = (8, 8)
		elif x.shape[-1] == 128:
			sz = (16, 8)
		elif x.shape[-1] == 256:
			sz = (16, 16)
		elif x.shape[-1] == 512:
			sz = (32, 16)
		elif x.shape[-1] == 1024:
			sz = (32, 32)
		return x.view(-1, self.in_channels, sz[0], sz[1])
		

	def forward(self, x):		
		outputs = []		
		for i, m in enumerate(self.models):
			outputs.append(m(self.reshape(x)))
			if x.shape[-1] > 128:
				seq_len = x.shape[-1]//2
				p  = torch.randint(0, x.shape[-1]-seq_len, (1,))
				x = x[:, :, p:p+seq_len]		
		return outputs

class ConditionalDiscriminator(nn.Module):
	def __init__(self, in_channels=2, con_len=2, n_dis=3):
		super(ConditionalDiscriminator, self).__init__()

		in_channels = in_channels + 1 #label

		layers = [
			nn.Linear(con_len, 2*32),
			Reshape(-1, 2, 32),
			UpsampleBlock(in_feat=2, out_feat=32, scale=4),
			UpsampleBlock(in_feat=32, out_feat=32, scale=4),
			nn.Conv1d(32, 1, 3, padding=1)
		]		
		self.emb_l = nn.Sequential(*layers)

		# Extracts three discriminator models
		self.models = nn.ModuleList()
		for i in range(n_dis):
			self.models.add_module(
				"disc_%d" % i,
				nn.Sequential(
					*discriminator_block(in_channels, 64, normalize=False),
					*discriminator_block(64, 128),
					*discriminator_block(128, 256),
					# *discriminator_block(256, 512),
					nn.Conv2d(256, 1, 3, padding=1)
				),
			)

		# self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)
		self.downsample = nn.AvgPool2d(in_channels, stride=2, count_include_pad=False)
		self.in_channels = in_channels

	def compute_loss(self, x, l, gt):
		"""Computes the MSE between model output and scalar gt"""		
		loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x, l)])
		return loss

	# def sim(self, x1, x2, crop_len=None):
	# 	if not crop_len is None:
	# 		l = x1.shape[-1]
	# 		p = torch.randint(0, l-crop_len, (1,))
	# 		x1 = x1[:, :, p:p+crop_len]
	# 		x2 = x2[:, :, p:p+crop_len]
	# 	return -1*torch.mean(torch.abs(self.fmap(x1) - self.fmap(x2)))

	def fmap(self, x, l):
		l = self.emb_l(l)
		x = torch.cat((x, l), dim=1)
		m = next(iter(self.models))
		return m[:4](self.reshape(x))
	

	def reshape(self, x):
		if x.shape[-1] == 64:
			sz = (8, 8)
		elif x.shape[-1] == 128:
			sz = (16, 8)
		elif x.shape[-1] == 256:
			sz = (16, 16)
		else:
			sz = (32, 16)
		return x.view(-1, self.in_channels, sz[0], sz[1])
		

	def forward(self, x, l):
		l = self.emb_l(l)
		x = torch.cat((x, l), dim=1)
		outputs = []
		for i, m in enumerate(self.models):
			outputs.append(m(self.reshape(x)))
			if x.shape[-1] > 128:
				seq_len = x.shape[-1]//2
				p  = torch.randint(0, x.shape[-1]-seq_len, (1,))
				x = x[:, :, p:p+seq_len]		
		return outputs

if __name__ == '__main__':
	dim   = 16
	depth = 4
	sz = (dim*2**depth, 512//2**depth)
	n_tmpl = 2
	exp_len = 1024

	Enc1 = Encoder1(in_channels=1, dim=dim, n_downsample=depth)
	Dec1 = Decoder1(out_channels=1, dim=dim, n_upsample=depth)
	Enc2 = Encoder2(in_channels=3, dim=dim, n_downsample=depth)
	Dec2 = Decoder2(out_channels=3, dim=dim, n_upsample=depth)
	Enc3 = Encoder3(in_channels=3, seq_len=exp_len, dim=dim, n_downsample=depth)
	Dec3 = Decoder3(out_channels=3, seq_len=exp_len, dim=dim, n_upsample=depth)
	
	for seq_len in [256, 512]:
		X1 = torch.randn(2, 1, seq_len)
		l  = torch.randn(2, 1, 2)
		c1 = Enc1(X1, l)
		X11 = Dec1(c1, l)
		c2  = torch.randn(2, 1, 2*c1.shape[-1]-2)
		X12 = Dec2(c1, l, c2)
		c1_12, l_12, c2_12 = Enc2(X12, l)
		# print('Samples', X1.shape, X11.shape, X12.shape)
		# print('Codes', c1.shape, c1_12.shape, c2.shape, c2_12.shape)
		# print('Label', l.shape, l_12.shape)
		c3 = torch.randn(2, 1, exp_len-X12.shape[-1])
		p  = torch.randint(0, exp_len-X12.shape[-1], (1,))
		X13, F13 = Dec3.expand(X12, c3, p)
		c3_13 = Enc3(F13)
		print(X1.shape, X11.shape, X12.shape, X13.shape, F13.shape)
		print(c1.shape, c2.shape, c1_12.shape, c2_12.shape, c3.shape, c3_13.shape)
	# summary(Dec1.cuda(), [sz, (1, 2)])

	# D1 = ConditionalDiscriminator(in_channels=1, n_dis=3, con_len=512+2)
	# out = D1(torch.randn(2, 1, 512), torch.randn(2, 1, 512+2))
	# loss = D1.compute_loss(torch.randn(2, 1, 512), torch.randn(2, 2), 0)

	# Enc2 = Encoder2(in_channels=3, seq_len=512,	dim=dim, n_downsample=depth)
	# summary(Enc2.cuda(), [(3, 512)])

	# Dec2 = Decoder2(out_channels=3, seq_len=512, dim=dim, n_upsample=depth)
	# summary(Dec2.cuda(), [sz, (1, 64)])

	# Enc3 = Encoder3(in_channels=3, seq_len=1024, dim=dim, n_downsample=depth)
	# c3 = Enc3(torch.randn(2, 3, 1024))
	# summary(Enc3.cuda(), (3, 1024))

	# Dec3 = Decoder3(out_channels=3, dim=dim, n_upsample=depth)
	# # xjn, xex, f = Dec3.fill(torch.randn(2, 3, 512), torch.randn(2, 3, 512), torch.randn(2, 128, 2, 2), 256)
	# xj1, xj2, xj3 = Dec3.joints(torch.randn(2, 3, 512), torch.randn(2, 3, 512), torch.randn(2, 128, 2, 2))

	# D = Discriminator(in_channels=3)
	# out = D(torch.randn(2, 3, 128))
	# loss = D.compute_loss(torch.randn(2, 3, 128), 1)
	# # param_count(Dec3)
	# # summary(Dec3.cuda(), [(3, 512), (3, 512), (128, 2, 2)])
	# DL = Discriminator(in_channels=3, n_dis=1)
	# out = DL(torch.randn(2, 3, 128))
	# loss = DL.compute_loss(torch.randn(2, 3, 128), 1)


	