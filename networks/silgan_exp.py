from .silgan import *

class SilGAN(SilGANBase):
	def __init__(self, opt):
		super(SilGAN, self).__init__()

		self.opt = opt
		self.exp_len = 1024
		self.seq_lens = [512]
		
		Enc1 = Encoder1(in_channels=1,  dim=opt.dim, n_downsample=opt.n_downsample)
		Dec1 = Decoder1(out_channels=1, dim=opt.dim, n_upsample=opt.n_downsample)		

		Enc2 = Encoder2(in_channels=3,  dim=opt.dim, n_downsample=opt.n_downsample)
		Dec2 = Decoder2(out_channels=3, dim=opt.dim, n_upsample=opt.n_downsample)
		D2   = Discriminator(in_channels=3)

		Enc3 = Encoder3(in_channels=3,  seq_len=self.exp_len, dim=opt.dim, n_downsample=opt.n_downsample)
		Dec3 = Decoder3(out_channels=3, seq_len=self.exp_len, dim=opt.dim, n_upsample=opt.n_downsample)
		D3   = Discriminator(in_channels=3)

		self.Enc1 = Enc1.cuda()
		self.Dec1 = Dec1.cuda()	

		self.Enc2 = Enc2.cuda()
		self.Dec2 = Dec2.cuda()		
		self.D2   = D2.cuda()
		
		self.Enc3 = Enc3.cuda()
		self.Dec3 = Dec3.cuda()
		self.D3   = D3.cuda()

		# Initialize weights
		self.Enc1.apply(weights_init_normal)
		self.Dec1.apply(weights_init_normal)		
		
		self.Enc2.apply(weights_init_normal)
		self.Dec2.apply(weights_init_normal)		
		self.D2.apply(weights_init_normal)

		self.Enc3.apply(weights_init_normal)
		self.Dec3.apply(weights_init_normal)
		self.D3.apply(weights_init_normal)

		
		self.optimizer_G = torch.optim.Adam(
			itertools.chain(Enc1.parameters(), Dec1.parameters(), 
				Enc2.parameters(), Dec2.parameters(),				
				Enc3.parameters(), Dec3.parameters()),
			lr=opt.lr,
			betas=(opt.b1, opt.b2),
		)
				
		self.optimizer_D2 = torch.optim.Adam(D2.parameters(),
			lr=opt.lr, betas=(opt.b1, opt.b2))	

		self.optimizer_D3 = torch.optim.Adam(D3.parameters(),
			lr=opt.lr, betas=(opt.b1, opt.b2))	

	def load(self, models_dir, epoch):
		self.Enc1.load_state_dict(torch.load(models_dir + 'Enc1_%d.pth' % epoch))
		self.Dec1.load_state_dict(torch.load(models_dir + 'Dec1_%d.pth' % epoch))		
		self.Enc2.load_state_dict(torch.load(models_dir + 'Enc2_%d.pth' % epoch))
		self.Dec2.load_state_dict(torch.load(models_dir + 'Dec2_%d.pth' % epoch))		
		self.D2.load_state_dict(torch.load(models_dir + 'D2_%d.pth' % epoch))
		self.Enc3.load_state_dict(torch.load(models_dir + 'Enc3_%d.pth' % epoch))
		self.Dec3.load_state_dict(torch.load(models_dir + 'Dec3_%d.pth' % epoch))
		self.D3.load_state_dict(torch.load(models_dir + 'D3_%d.pth' % epoch))

	def save(self, models_dir, epoch):
		torch.save(self.Enc1.state_dict(), models_dir + '/Enc1_%d.pth' % epoch)
		torch.save(self.Dec1.state_dict(), models_dir + '/Dec1_%d.pth' % epoch)		
		torch.save(self.Enc2.state_dict(), models_dir + '/Enc2_%d.pth' % epoch)
		torch.save(self.Dec2.state_dict(), models_dir + '/Dec2_%d.pth' % epoch)
		torch.save(self.D2.state_dict(),   models_dir + '/D2_%d.pth' % epoch)
		torch.save(self.Enc3.state_dict(), models_dir + '/Enc3_%d.pth' % epoch)
		torch.save(self.Dec3.state_dict(), models_dir + '/Dec3_%d.pth' % epoch)	
		torch.save(self.D3.state_dict(),   models_dir + '/D3_%d.pth' % epoch)
					
	def eval(self):
		self.Enc1.eval()
		self.Dec1.eval()		
		self.Enc2.eval()
		self.Dec2.eval()		
		self.D2.eval()
		self.Enc3.eval()
		self.Dec3.eval()
		self.D3.eval()

	def train(self):
		self.Enc1.train()
		self.Dec1.train()		
		self.Enc2.train()
		self.Dec2.train()		
		self.D2.train()
		self.Enc3.train()
		self.Dec3.train()
		self.D3.train()

	def summary(self):
		print('Enc1 - %d params' % param_count(self.Enc1))
		print('Dec1 - %d params' % param_count(self.Dec1))		
		print('Enc2 - %d params' % param_count(self.Enc2))
		print('Dec2 - %d params' % param_count(self.Dec2))
		print('D2   - %d params' % param_count(self.D2))
		print('Enc3 - %d params' % param_count(self.Enc3))
		print('Dec3 - %d params' % param_count(self.Dec3))
		print('D3   - %d params' % param_count(self.D3))


	def train_step(self, epoch, batch, _X1, X2, X3):
		opt  = self.opt
		Enc1 = self.Enc1
		Dec1 = self.Dec1		
		Enc2 = self.Enc2
		Dec2 = self.Dec2		
		D2   = self.D2
		Enc3 = self.Enc3
		Dec3 = self.Dec3
		D3   = self.D3
		optimizer_G = self.optimizer_G		
		optimizer_D2 = self.optimizer_D2
		optimizer_D3 = self.optimizer_D3
		criterion_recon = self.criterion_recon		
		criterion_ssim = self.criterion_ssim

		n_tmpl = _X1.shape[1]
		ls = batch % n_tmpl
		X1 = torch.unsqueeze(_X1[:, ls, :], dim=1)
		lt = torch.full((X1.shape[0], 1), ls, dtype=torch.int64)
		lt = torch.nn.functional.one_hot(lt, num_classes=n_tmpl)		

		# Set model input
		X1 = Variable(X1.type(Tensor))
		lt = Variable(lt.type(Tensor))
		X2 = Variable(X2.type(Tensor))
		X3 = Variable(X3.type(Tensor))		
		
		
					
		# -------------------------------
		#  Generation
		# -------------------------------
		optimizer_G.zero_grad()
		
		#stage-1
		c1 = Enc1(X1, lt)
		X11 = Dec1(c1, lt)
		
				
		#stage-2
		_c1, _lt, _c2 = Enc2(X2, lt)
		X22 = Dec2(_c1, _lt, _c2)
		
		c2_A  = torch.randn(X3.shape[0], 1, 2*c1.shape[-1]-n_tmpl).cuda()		
		X12_A = Dec2(c1, lt, c2_A)
		c1_12, _, c2_12 = Enc2(X12_A, lt)
		X121 = Dec1(c1_12, lt)		

		c2_B  = torch.randn(X3.shape[0], 1, 2*c1.shape[-1]-n_tmpl).cuda()		
		X12_B = Dec2(c1, lt, c2_B)

		#stage-3		
		c3_A = torch.randn(X3.shape[0], 1, X3.shape[-1]-X12_A.shape[-1]).cuda()
		p = torch.randint(0, X3.shape[-1]-X12_A.shape[-1], (1,))
		X13, F13_A = Dec3.expand(X12_A, c3_A, p)		
		c3_13 = Enc3(F13_A)
		
		c3_B = torch.randn(X3.shape[0], 1, X3.shape[-1]-X12_A.shape[-1]).cuda()
		p = torch.randint(0, X3.shape[-1]-X12_A.shape[-1], (1,))
		_, F13_B = Dec3.expand(X12_A, c3_B, p)


		# Losses
		loss_GAN = opt.lambda_gan * (			
			D2.compute_loss(X12_A, valid)+
			D3.compute_loss(X13, valid))

		loss_ID  = opt.lambda_id  * (
			criterion_recon(X11, X1) +
			criterion_recon(X22, X2)) 
		
		loss_c   = opt.lambda_cr  * (			
			criterion_recon(c1_12, c1.detach()) +			
			criterion_recon(c2_12, c2_A.detach())+
			criterion_recon(c3_13, c3_A.detach()))
		
		loss_cyc = opt.lambda_cyc  * (
			criterion_recon(X121, X1))

		loss_pair = opt.lambda_pair * (
			criterion_recon(X12_A[:, ls, :], X2[:, ls, :]))

		loss_m2   = opt.lambda_ms2*(
			-1*torch.mean(torch.abs(D2.fmap(X12_A) - D2.fmap(X12_B)))/torch.mean(torch.abs(c2_A - c2_B)))
				
		loss_m3   = opt.lambda_ms3*(torch.mean(criterion_ssim(F13_A, F13_B))/torch.mean(torch.abs(c3_A - c3_B)))

		loss_cont = torch.Tensor([0])
		# loss_cont = opt.lambda_cont*(
		# 	criterion_recon(XJ1[:, :, 63], XJ1[:, :, 64]) +
		# 	criterion_recon(XJ2[:, :, 63], XJ2[:, :, 64]) + 
		# 	criterion_recon(XJ3[:, :, 63], XJ3[:, :, 64]))		
		
		
		loss_G = (loss_GAN + loss_ID + loss_c + loss_cyc + loss_pair + loss_m2 + loss_m3 + loss_cont)
		

		loss_G.backward()
		optimizer_G.step()


		# -----------------------
		#  Discrimination - Translation
		# -----------------------
		optimizer_D2.zero_grad()

		loss_D2 = (D2.compute_loss(X2, valid) +			      
			      D2.compute_loss(X12_A.detach(), fake))

		loss_D2.backward()
		optimizer_D2.step()		

		# -----------------------
		#  Discrimination - Expansion
		# -----------------------
		optimizer_D3.zero_grad()

		loss_D3 = (D3.compute_loss(X3, valid) +			      
			      D3.compute_loss(X13.detach(), fake))

		loss_D3.backward()
		optimizer_D3.step()		

		loss_D = (loss_D2 + loss_D3)

		return loss_G, loss_D, loss_ID , loss_c, loss_cyc , loss_pair , loss_m2, loss_m3, loss_cont




if __name__ == '__main__':	

	import sys
	sys.path.append('..')
	# from r2s.dataloader4 import dataloader
	import utils6 as utils
	import metrics
	from argparse import Namespace

	opt = Namespace(batch_size=128, n_residual=1, dim=16, n_downsample=4, use_pairing=False,
		lr=0.0001, b1=0.5, b2=0.999,
		lambda_gan=1.0, lambda_id=10.0, lambda_cr=1.0, lambda_cyc=20.0, lambda_pair=10.0, lambda_ms2=1.0, lambda_ms3=1.0, lambda_cont=0.5)

	model = SilGAN(opt)	
	# train_dataloader, test_dataloader, (X1_mini, X2_mini, X3_mini) = dataloader('/cephyr/NOBACKUP/groups/snic2020-8-120/bilgan/r2s', 128, 16)
	
	
	model.summary()

	# X1, X2, X3 = next(iter(train_dataloader))	
	batch = np.load('_batch.npz')
	X1, X2, X3 = batch['X1'], batch['X2'], batch['X3']
	X1 = torch.from_numpy(X1).float()
	X2 = torch.from_numpy(X2).float()
	X3 = torch.from_numpy(X3).float()

	for n in range(4):
		print('step-', n)
		model.train_step(0, n, X1, X2, X3)
	

	
	
	X11, X22 = model.reconstruct(X1, X2)
	for l in range(X1.shape[1]):
		X12 = model.translate(X1, l)
		# X12, X121 = model.sweep(X1, l)
		# X12, X121 = model.metric_sweep(X1, l, metric_fn=metrics.batch_L1, ovsmpl=10)
		X121 = model.cycle_translate(X12, l)

		X13 = model.expand(X1, l)

	# 		utils.plot_translation(X1, l, X2, X12,
	# 			savepath='_2_trans_%d_%d.png' % (seqlsen, l))
	# 	print(X1.shape, X11.shape, X2.shape, X22.shape, X12.shape)