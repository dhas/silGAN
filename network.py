import torch
from torch import nn
from torchsummary import summary

def initialize_weights(model):
	for m in model.modules():
		if isinstance(m, nn.Conv1d):
			m.weight.data.normal_(0, 0.02)
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.ConvTranspose1d):
			m.weight.data.normal_(0, 0.02)
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.02)
			if m.bias is not None:
				m.bias.data.zero_()

def add_conv_block(model, n, in_feat, out_feat, kernel_size, stride, padding, transpose=False):
	if transpose:
		conv_layer = nn.ConvTranspose1d(in_feat, out_feat,
			kernel_size=kernel_size, stride=stride,
			padding=padding, output_padding=1, bias=False)		
	else:
		conv_layer = nn.Conv1d(in_feat, out_feat,
			kernel_size=kernel_size, stride=stride,
			padding=padding, bias=False)

	model.add_module('Conv1d-%d' % n, conv_layer)
	model.add_module('BatchNorm1d-%d' % n, nn.BatchNorm1d(out_feat))
	model.add_module('ReLU-%d' % n, nn.ReLU(inplace=True))

class Encoder(nn.Module):
	def __init__(self, nz, window,  n_feat, n_layers):
		super(Encoder, self).__init__()

		assert 512//2**n_layers >= 2, 'Insufficient n_feat'

		padding = window//2
		stride  = 2
		hidden = n_feat*(512//2**n_layers)
		
		self.conv_block = nn.Sequential()
		add_conv_block(self.conv_block, 0, 2, n_feat, window, stride, padding)		
		for i in range(n_layers-1):
			add_conv_block(self.conv_block, i+1, n_feat, n_feat, window, stride, padding)
		self.mu = nn.Linear(hidden, nz, bias=False)
		self.logvar = nn.Linear(hidden, nz, bias=False)
		initialize_weights(self)
	
	def forward(self, x):
		batch_size = x.size(0)
		x = self.conv_block(x)		
		x = x.view(batch_size, -1) 
		mu = self.mu(x)
		logvar = self.logvar(x)
		return mu, torch.clamp(logvar, -2, 2)

	
class Decoder(nn.Module):
	def __init__(self, nz, window, n_feat, n_layers):
		super(Decoder, self).__init__()

		self.n_feat = n_feat
		self.hidden = n_feat*(512//2**n_layers)

		self.fc1 = nn.Sequential(
			nn.Linear(nz, self.hidden, bias=False),
			nn.BatchNorm1d(self.hidden),
			nn.ReLU(inplace=True)
		)

		stride  = 2
		padding = window//2
		self.deconv_block = nn.Sequential()
		for i in range(n_layers):
			add_conv_block(self.deconv_block, i, n_feat, n_feat, 
				window, stride, padding, transpose=True)		
		self.conv5 = nn.Sequential(
			nn.Conv1d(n_feat, 2, window, 1, padding),
			nn.Sigmoid()
		)

		initialize_weights(self)

	def forward(self, x):
		x = self.fc1(x)
		x = x.view(-1, self.n_feat, self.hidden//self.n_feat)
		x = self.deconv_block(x)		
		x = self.conv5(x)
		return x

class Generator(nn.Module):
	"""
	Generator class
	"""
	def __init__(self, nz, window=9, n_feat=64, n_layers=4):
		super(Generator, self).__init__()

		self.encoder = Encoder(nz, window, n_feat, n_layers)
		self.decoder = Decoder(nz, window, n_feat, n_layers)
	
	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)
		else:
			return mu

	def forward(self, x):
		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)
		xhat = self.decoder(z)
		return xhat, mu, logvar
	
	def generate(self, z):
		self.eval()
		samples = self.decoder(z)
		return samples

	def encode(self, x):
		self.eval()
		z, _ = self.encoder(x)
		return z

	def reconstruct(self, x):
		self.eval()
		mu, _ = self.encoder(x)
		xhat = self.decoder(mu)
		return xhat

class Discriminator(nn.Module):  
	def __init__(self, nz, window=9, n_feat=64, n_layers=4):
		super(Discriminator, self).__init__()
		stride  = 2
		padding = window//2
		self.hidden = n_feat*(512//2**n_layers)
		
		self.conv_block = nn.Sequential()
		add_conv_block(self.conv_block, 0, 2, n_feat, window, stride, padding)
		for i in range(n_layers-1):
			add_conv_block(self.conv_block, i+1, n_feat, n_feat, window, stride, padding)		
		self.fc5 = nn.Sequential(
			nn.Linear(self.hidden, nz, bias=False),
			nn.BatchNorm1d(nz),
			nn.ReLU(inplace=True)
		)

		self.fc6 = nn.Sequential(
			nn.Linear(nz, 1),
		)

		self.fc7 = nn.Sequential(
			nn.Sigmoid()
		)
		initialize_weights(self)

	def feature(self, x):
		f = self.conv_block(x)		
		return f.view(-1, self.hidden)

	def forward(self, x):
		x = self.feature(x)
		x = self.fc5(x)
		x = self.fc6(x)
		return x
	
	def prob(self, x):
		x = self.forward(x)
		x = self.fc7(x)
		return x
		
if __name__ == '__main__':
	nz = 10
	n_feat  = 64
	n_layers = 6

	encoder = Encoder(10, 9, n_feat, n_layers)
	encoder.cuda()
	summary(encoder, (2,512))

	# decoder = Decoder(10, 9, 64, 4)
	# decoder.cuda()
	# summary(decoder,(1,10))

	generator = Generator(nz, n_feat=n_feat, n_layers=n_layers)
	generator.cuda()
	summary(generator, (2,512))

	disciminator = Discriminator(nz, n_feat=n_feat, n_layers=n_layers)
	disciminator.cuda()
	summary(disciminator, (2,512))