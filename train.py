import argparse
import json
import os
import numpy as np
import sys
import torch
from r2s.dataloader import dataloader
import utils as utils
# import metrics
# import templates


def seed_fn(worker_id):
	np.random.seed(opt.seed)
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed_all(opt.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False	

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=1, help="random seed")
	parser.add_argument("--job_id", type=str, default=None, help="SLURM id of job to resume")	
	parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
	parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
	parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
	parser.add_argument("--batch_lim", type=int, default=-1, help="number of batches to use in an epoch")
	parser.add_argument("--config", default='exp', choices=['exp'], help="model config")	
	parser.add_argument("--no_log", action='store_true')
	parser.add_argument("--lambda_gan", type=float, default=1.0)
	parser.add_argument("--lambda_id", type=float, default=10.0)
	parser.add_argument("--lambda_pair", type=float, default=0.0)
	parser.add_argument("--lambda_cyc", type=float, default=1.0)
	parser.add_argument("--lambda_cr", type=float, default=1.0)
	parser.add_argument("--lambda_mgan", type=float, default=0.0)
	parser.add_argument("--lambda_ms2", type=float, default=0.1)
	parser.add_argument("--lambda_ms3", type=float, default=0.3)
	parser.add_argument("--lambda_cont", type=float, default=0.0)
	parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
	parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
	parser.add_argument("--sample_interval", type=int, default=100, help="interval saving generator samples")
	parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
	parser.add_argument("--n_downsample", type=int, default=4, help="number downsampling layers in encoder")
	parser.add_argument("--n_residual", type=int, default=1, help="number of residual blocks in encoder / decoder")
	parser.add_argument("--dim", type=int, default=16, help="number of filters in first encoder layer")	
	
	opt = parser.parse_args()

	seed_fn(0)

	if opt.config == 'exp':
		from networks.silgan_exp import SilGAN

	storage_root = '/cephyr/NOBACKUP/groups/snic2020-8-120/bilgan/'
	log_root = '%s/logs' % storage_root
	if opt.job_id is None:
		try:
			job_id = os.environ["SLURM_JOB_ID"]
		except Exception:
			from datetime import datetime
			now = datetime.now()
			job_id = now.strftime("%d-%m-%Y--%H-%M-%S")
		opt.job_id = job_id
		models_dir = '%s/logs/models/%s/%s/' % (storage_root, 'silgan', opt.job_id)
		os.makedirs(models_dir)
		opt.job_name =  'silgan_%s/%s' % (opt.job_id, opt.config)
		if not opt.no_log:
			opt.run_id = utils.setup_logging('bilgan', opt.job_name, opt, logdir=log_root)
		model = SilGAN(opt)
	else:
		models_dir = '%s/logs/models/%s/%s/' % (storage_root, 'silgan', opt.job_id)
		if not os.path.isdir(models_dir):
			raise Exception('job_id does not exist')
		with open(models_dir+ 'opt.json', 'r') as f:
			saved_opt = argparse.Namespace(**json.load(f))
		saved_opt.epoch = opt.epoch
		saved_opt.n_epochs = opt.n_epochs
		saved_opt.batch_lim = opt.batch_lim
		
		opt = saved_opt						
		utils.setup_logging('bilgan', job_name, opt.job_id, logdir=log_root, run_id=opt.run_id)
		model = SilGAN(opt)
		model.load(models_dir, opt.epoch-1)
	
	print(opt)
	
	with open(models_dir + 'opt.json', 'w') as f:
		json.dump(vars(opt), f)

	
	data_root = '%s/r2s/' % storage_root
	train_dataloader, test_dataloader, (X1_mini, X2_mini, X3_mini) = dataloader(data_root, 
		opt.batch_size, opt.n_cpu, 	worker_init_fn=seed_fn, shuffle_2=(True if opt.lambda_pair == 0 else False))
		
	X1_mini1 = X1_mini[torch.randperm(X1_mini.shape[0])]
		

	for epoch in range(opt.epoch, opt.n_epochs):		
		for i, batch in enumerate(train_dataloader):
			X1, X2, X3 = batch			
			losses = model.train_step(epoch, i, X1, X2, X3)

			loss_G, loss_D, loss_ID , loss_c, loss_cyc , loss_pair , loss_m2, loss_m3, loss_cont = losses

			
			# # --------------
			# #  Log Progress
			# # --------------
			batches_done = epoch * len(train_dataloader) + i
			sys.stdout.write(
				"\r[Step %d/%d] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
				% (batches_done, opt.n_epochs * len(train_dataloader),
				 epoch, opt.n_epochs, i, len(train_dataloader), 
				 loss_D.item(), loss_G.item())
			)
			sys.stdout.flush()

			if (batches_done % opt.sample_interval == 0):
				model.eval()

				log_dict = dict()
				log_dict['epoch']  = epoch
				#training metrics
				log_dict['G_loss'] = loss_G.item()				
				log_dict['D_loss'] = loss_D.item()
				log_dict['ID_loss'] = loss_ID.item()
				log_dict['CR_loss'] = loss_c.item()
				log_dict['CY_loss'] = loss_cyc.item()
				log_dict['PR_loss'] = loss_pair.item()
				log_dict['M2_loss'] = loss_m2.item()
				log_dict['M3_loss'] = loss_m3.item()
				log_dict['CO_loss'] = loss_cont.item()
				
								
				ssim_11 = []
				ssim_22 = []
				for seq_len in model.seq_lens:
					if seq_len < X2_mini.shape[-1]:
						mid = 512//2
						shf = seq_len//2
						_X1 = X1_mini[:, :, mid-shf: mid+shf]
						_X2 = X2_mini[:, :, mid-shf: mid+shf]					
					else:
						_X1 = X1_mini
						_X2 = X2_mini
					for l in range(X1_mini.shape[1]):
						X12  = model.translate(_X1, l)
						log_dict['trans_%d_%d' % (seq_len, l)] = utils.plot_translation(_X1, l, X12)
						
						if seq_len == 512:
							X121 = model.cycle_translate(X12, l)
							log_dict['cyc_trans_%d' % l] = utils.plot_cycle_translation(_X1, l, X121)

						if 'exp' in opt.config and seq_len < model.exp_len:
							X13 = model.expand(_X1, l)
							log_dict['exp_%d_%d' % (seq_len, l)] = utils._plot_expansion(_X1, l, X13)						
						# X13jn, _, _, _ = model.join(X1_mini1, X1_mini, l)
						# log_dict['fill_%d' % l] = utils.plot_join(X1_mini1, X1_mini, X13jn, l)
				
				X1_test, X2_test, X3_test = next(iter(test_dataloader))					
				X11, X22 = model.reconstruct(X1_test, X2_test)				
				# log_dict['SSIM_11'] = metrics.get_ssim(ref=X1_test, recon=X11)
				# log_dict['SSIM_22'] = metrics.get_ssim(ref=X2_test, recon=X22)
				
				
				# T1 = templates.takeoff(num=10, k=0.1)
				# T1 = torch.from_numpy(T1).float()
				# for l in range(T1.shape[1]):
				# 	if not torch.isnan(T1[:, l]).any():
				# 		T12 = model.translate(T1, l, num_styles=1)
				# 		T12 = torch.cat((T12[0], torch.unsqueeze(T1[:, l], 1)), dim=1)
				# 		log_dict['takeoff_trans_%d' % (l)] = utils.visualize_test_mini((T1, T12), l)

				# 		T12, T121 = model.sweep(T1, l)
				# 		log_dict['takeoff_sweep_%d' % (l)] = utils.plot_sweep(T12, T121, l, X1=T1)

				# 		T12, T121 = model.metric_sweep(T1, l, metric_fn=metrics.batch_L1, ovsmpl=100)
				# 		log_dict['takeoff_mlerp_%d' % (l)] = utils.plot_sweep(T12, T121, l, X1=T1)
				# div_trn = []
				# div_exp = []
				# con_nrm = []
				# con_unm = []
				# for l in range(X1.shape[1]):				
				# 	X12 = model.translate(X1_test, l, X2_test, num_styles=100)
				# 	div_trn.append(metrics.batch_trn_diversity(X2, X12))
				# 	X13 = model.expand(X1_test, l)					
				# 	div_exp.append(metrics.batch_exp_diversity(X13))
				# 	_, XJ1, XJ2, XJ3 = model.join(X1_test[torch.randperm(X1_test.shape[0])], X1_test, l)
				# 	con_nrm.append(metrics.continuity(XJ1, XJ2, XJ3, normalize=True))
				# 	con_unm.append(metrics.continuity(XJ1, XJ2, XJ3, normalize=False))

				# log_dict['DIV_TRN'] = np.mean(div_trn)
				# log_dict['DIV_EXP'] = np.mean(div_exp)
				# log_dict['CON_NRM'] = np.mean(con_nrm)
				# log_dict['CON_UNM'] = np.mean(con_unm)
				
				if (not opt.no_log):
					utils.log_metrics(log_dict, step=batches_done)

				model.train()
			
			if opt.batch_lim != -1 and i >= opt.batch_lim:
				break

		if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
			model.save(models_dir, epoch)