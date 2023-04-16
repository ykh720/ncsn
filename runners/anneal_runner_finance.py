import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from datasets.Dset import * 
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image
import scipy.io 
from sklearn.model_selection import train_test_split
import argparse
import random


# what's the use of this?
__all__ = ['AnnealRunnerFin']


class AnnealRunnerFin():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    # def getdataset(self, )

    def train(self):

        ## Older, from historical data
        # totalvarsurface = scipy.io.loadmat('msft_hestonlike_IVS.mat') # the address is relative to the address of thescript that calls this class
        # totalvarsurface = totalvarsurface['totalvarsurface']

        ## Edited at Feb 19, data obtain from Heston model
        # totalvarsurface = scipy.io.loadmat('HestonIVSgrid_NI25652.mat')
        totalvarsurface = scipy.io.loadmat(self.config.data.datasetpath)
        totalvarsurface = totalvarsurface['HestonIVS2D']
        datasetsize = totalvarsurface.shape[0]
        if self.config.data.scale:
            # if scale is true, we use total implied variance instead! 

            tlist = scipy.io.loadmat(self.config.data.finname + '_tlist' + str(datasetsize) + '.mat')
            tlist = tlist['tlist'] # note that they are 2D array 
            tlist = tlist.reshape((-1))
            # Klist = scipy.io.loadmat('msft_Klist.mat')
            Klist = scipy.io.loadmat(self.config.data.finname + '_Klist'+ str(datasetsize)+'.mat')
            Klist = Klist['Klist'] # note that they are 2D array
            Klist = Klist.reshape((-1))
            tlist_broadcast = tlist.reshape((1, len(tlist), 1))
            totalvarsurface = totalvarsurface**2 * tlist_broadcast

        IVStrain, IVStest = train_test_split(totalvarsurface, test_size = 0.2, random_state = 42)

        dataset = Dset(IVStrain)
        # batch_size = 24 
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size,shuffle=True)

        test_dataset = Dset(IVStest)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size,shuffle=True, drop_last=True)

        # dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        # test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
        #                          num_workers=4, drop_last=True)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        score = CondRefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score)

        pytorch_total_params = sum(p.numel() for p in score.parameters() if p.requires_grad)
        print('Total number of parameters:', pytorch_total_params)
        
        # return 0

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        # linear noise scale? wait we have exp, so might be still geometric
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)

        # r = 1.6681005372000586, simgas[1] / sigmas[2]
        # if we want to fix r, given sigma_begin, sigma_end
        # then we should pick num_classes n by 
        # n = (np.log(sigma_begin) - np.log(sigma_end))/ np.log(r) + 1
        # for sigma_begin = 1, sigma_end = 0.001
        # n = 15 (after round up to nearest integer)

        # if self.config.training.log_all_sigmas:
        if False:
            ### Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(len(sigmas))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(len(sigmas)):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(len(sigmas)):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        for epoch in range(self.config.training.n_epochs):
            for i, X in enumerate(dataloader):
                step += 1
                score.train()
                X = X[:, None, :, :] # to match the dimension
                X = X.to(self.config.device)
                # print(X.shape)

                # why do we need to add noise to X?
                # X = X / 256. * 255. + torch.rand_like(X) / 256.
                
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    # loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                    loss = anneal_dsm_score_estimation(score, X, sigmas, None,
                                                   self.config.training.anneal_power,
                                                   hook)

                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=1) # self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X= next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X = next(test_iter)

                    test_X = test_X[:, None, :, :] # to match the dimension
                    test_X = test_X.to(self.config.device)
                    # test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    with torch.no_grad():
                        # test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                                                                    # self.config.training.anneal_power)
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, sigmas, labels=None, anneal_power=self.config.training.anneal_power, hook=hook)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images

    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        with torch.no_grad():
            # denoising loop, decreasing sigma 
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                # inner loop for each sigma
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu')) 
                    # this is alright, note that our totalIVS maximum values is smaller than 1
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))
                print("signal to noise ratio:", step_size * grad.abs().mean() / noise.abs().mean())

            # output images contain each images generated in each inner step of Langevin dynamics
            return images


    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)
        # print(self.config)
        # tlist = scipy.io.loadmat('msft_tlist.mat')

        string1 = self.config.data.datasetpath.split("HestonIVSgrid_NI")[1]
        datasetsizestr = string1.split('.mat')[0]

        tlist = scipy.io.loadmat(self.config.data.finname + '_tlist' + datasetsizestr + '.mat')
        tlist = tlist['tlist'] # note that they are 2D array 
        tlist = tlist.reshape((-1))
        tlist_broadcast = tlist.reshape((1, len(tlist), 1))
        # Klist = scipy.io.loadmat('msft_Klist.mat')
        Klist = scipy.io.loadmat(self.config.data.finname + '_Klist' + datasetsizestr + '.mat')
        Klist = Klist['Klist'] # note that they are 2D array
        Klist = Klist.reshape((-1))

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()
        # denote the number of images to be generated grid_size = 5 means 5^2 = 25 images will be gen
        grid_size = 5 

        imgs = []
        if self.config.data.dataset == 'Finance':
            # sample from uniform dist? why? 
            grid_size = 1 # set grid_size for finance dataset

            # samples = torch.rand(grid_size ** 2, 1, self.config.data.image_size, self.config.data.image_size, device=self.config.device) 

            g_cpu = torch.Generator(device=self.config.device)
            g_cpu.manual_seed(2847587347)
            samples = torch.randn(grid_size ** 2, 1, self.config.data.image_size, self.config.data.image_size, device=self.config.device, generator=g_cpu) 
            print(samples)

            step_per_noise = 100
            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, step_per_noise, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                # if i % 10 == 0:
                #     im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                #     # im = Image.fromarray(image_grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                #     # I guess this might makes more sense, not in place and keeps the correct pic when i % 10 == 0
                #     imgs.append(im)

                # print(type(sample))
                # print(sample.shape) # note that it is a 4D tensor
                savepath = os.path.join(self.args.image_folder, 'ivs_image_{}.png'.format(i))
                # if i > step_per_noise * self.config.model.num_classes * 0.8:
                if i > -1:
                    if self.config.data.scale:
                        sample = torch.sqrt(sample.cpu()/ tlist_broadcast)
                    IVS_visualize(sample.reshape((self.config.data.image_size,self.config.data.image_size)), Klist, tlist, savepath=savepath, plotname = str(i) + "step")
                if i == 999:
                    print('step', i)
                    torch.set_printoptions(precision=10)
                    print(sample)
                if i == 0:
                    # totalvarsurface = scipy.io.loadmat('msft_hestonlike_IVS.mat') # the address is relative to the address of thescript that calls this class
                    totalvarsurface = scipy.io.loadmat(self.config.data.datasetpath)
                    # totalvarsurface = totalvarsurface['totalvarsurface']
                    totalvarsurface = totalvarsurface['HestonIVS2D']
                    IVS_visualize(totalvarsurface[i], Klist, tlist, savepath=os.path.join(self.args.image_folder, 'ivs_image_original{}.png'.format(i)))

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

        elif self.config.data.dataset == 'MNIST':
            # sample from uniform dist? why? 
            samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device) 
            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    # im = Image.fromarray(image_grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    # I guess this might makes more sense, not in place and keeps the correct pic when i % 10 == 0
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))


        else:
            samples = torch.rand(grid_size ** 2, 3, 32, 32, device=self.config.device)

            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    # im = Image.fromarray(image_grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    # I guess this might makes more sense, not in place and keeps the correct pic when i % 10 == 0
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)), nrow=10)
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

        # imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)

    def anneal_Langevin_dynamics_inpainting(self, x_mod, refer_image, scorenet, sigmas, mask, n_steps_each=100,
                                            step_lr=0.000008, noarb = False, sampmethod=2, samplenoise=False, samplenoiselast=False, samplenoise_noshowing=False,
                                             denoising_step=True):
        
        print('noarbitrage sampling:', noarb)

        string1 = self.config.data.datasetpath.split("HestonIVSgrid_NI")[1]
        datasetsizestr = string1.split('.mat')[0]
        
        tlist = scipy.io.loadmat(self.config.data.finname + '_tlist' + datasetsizestr + '.mat')
        tlist = tlist['tlist'] # note that they are 2D array 
        tlist = tlist.reshape((-1))
        tlist_broadcast = tlist.reshape((1, len(tlist), 1))
        # Klist = scipy.io.loadmat('msft_Klist.mat')
        Klist = scipy.io.loadmat(self.config.data.finname + '_Klist' + datasetsizestr + '.mat')
        Klist = Klist['Klist'] # note that they are 2D array
        Klist = Klist.reshape((-1))
        # note that S0 = 100
        r = 0.03
        logmlist = np.log(Klist / 100) - r* tlist

        # these should be treated as inputs, or in the future deduce a independent conditions that is based on sigma
        # also note that this change makes the code only applicable to num_class = 10 for now.
        # dlossthreshold = torch.tensor([7.1260, 3.8, 2.2, 1.4, 0.9, 0.55, 0.23, 0.07, 0.01, 0])
        # clossthreshold = torch.tensor([3.9, 2.3, 1.34, 0.76, 0.42, 0.21, 0.09, 0.03, 0.005, 0.0312])

        images = []

        refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_image = refer_image.contiguous().view(-1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)

        x_mod = x_mod.view(-1, self.config.data.channels, self.config.data.image_size ,self.config.data.image_size)
        # half_refer_image = refer_image[..., :self.config.data.image_size//2]
        testsize = x_mod.shape[0]
        updateavglist = []
        d25quantilelist = []
        d90quantilelist = []
        c25quantilelist = []
        c90quantilelist = []
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="annealed Langevin dynamics sampling inpainting"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                # corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                # corrupted_half_image = half_refer_image
                x_mod[:, :, mask] = refer_image[..., mask] # one more step compared to the algo 

                dlosslist = []
                closslist = []
                dlossold = 0
                clossold = 0
                updatepertlist_pernoise = []
                for s in range(n_steps_each):
                    
                    if samplenoiselast == True:
                        noise_mask = torch.randn_like(refer_image) * sigma
                        x_mod[:, :, mask] = refer_image[..., mask]  + noise_mask[..., mask]
                        
                    
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))

                    if samplenoiselast == True:
                        x_mod[:,:, mask] = refer_image[..., mask] 

                    if samplenoise_noshowing == True:
                        noise_mask = torch.randn_like(refer_image) * sigma
                        x_mod[:, :, mask] = refer_image[..., mask]  + noise_mask[..., mask]

                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)

                    # print(x_mod.shape)
                    x_prop = x_mod + step_size * grad + noise
                    
                    #### 04-16 update
                    # x_prop[:, :, mask] = refer_image[..., mask]
                    #### 04-16 update 


                    if self.config.data.scale:
                        dloss = Dcondloss_torch(torch.squeeze(x_prop), torch.squeeze(x_prop), logmlist) # .unsqueeze(0)
                        closs = calloss_torch(torch.squeeze(x_prop), torch.squeeze(x_prop), tlist) # .unsqueeze(0)
                        dlosslist.append(dloss)
                        closslist.append(closs)
                    else: 
                        x_prop_scale = x_prop**2 * torch.from_numpy(tlist_broadcast).float().to(self.config.device)
                        dloss = Dcondloss_torch(torch.squeeze(x_prop_scale), torch.squeeze(x_prop_scale), logmlist) # .unsqueeze(0)
                        closs = calloss_torch(torch.squeeze(x_prop_scale), torch.squeeze(x_prop_scale), tlist) # .unsqueeze(0)
                        dlosslist.append(dloss)
                        closslist.append(closs)

                    if noarb: 
                        if sampmethod == 2: 
                        # index = (dloss < dlossthreshold[c]) & (closs < clossthreshold[c])
                            index = (dloss <= dlossold) & (closs <= clossold)
                            # print('indexshape', index.shape)
                            x_mod[index] = x_prop[index]
                        elif sampmethod == 3:
                            u = torch.rand(testsize, device=self.config.device)*1.2
                            index = torch.maximum(dloss/ (dlossold + 0.001), closs/(clossold + 0.001)) <= u  
                            x_mod[index] = x_prop[index]
                    else: 
                        x_mod = x_prop
                        index = torch.ones(1) 
                        # just a dummy for the case of no noarb so that the following code can run smoothly
                        
                    # print(index)
                    # print(type(index))
                    updatepert = torch.sum(index) / len(index)
                    updatepertlist_pernoise.append(updatepert)

                    # x_mod[:, :, :, :16] = corrupted_half_image
                    
                    if samplenoise == False:
                    # noise_mask = torch.randn_like(refer_image) * sigma
                        x_mod[:, :, mask] = refer_image[..., mask] # + noise_mask[..., mask]
                    else: 
                        noise_mask = torch.randn_like(refer_image) * sigma
                        x_mod[:, :, mask] = refer_image[..., mask]  + noise_mask[..., mask]

                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

                    # note that for torch.tensor we need to use .clone()
                    dlossold = dloss.clone()
                    clossold = closs.clone()

                print('\n current sigma' , sigma)
                dloss_thislevel = torch.cat(dlosslist, dim=0)
                closs_thislevel = torch.cat(closslist, dim=0)
                # print(dloss_thislevel.shape) # n_steps_each * testsize
                # updateavg = np.mean(np.array(updatepertlist_pernoise))
                # print(updatepertlist_pernoise)
                # updateavg = torch.mean(torch.cat(updatepertlist_pernoise,dim=0))
                updateavg = torch.mean(torch.tensor(updatepertlist_pernoise))
                d25quantile = torch.quantile(dloss_thislevel, 0.25)
                c25quantile = torch.quantile(closs_thislevel, 0.25)
                d90quantile = torch.quantile(dloss_thislevel, 0.90)
                c90quantile = torch.quantile(closs_thislevel, 0.90)
                updateavglist.append(updateavg)
                d25quantilelist.append(d25quantile)
                c25quantilelist.append(c25quantile)
                d90quantilelist.append(d90quantile)
                c90quantilelist.append(c90quantile)
                print('d25quantile:', d25quantile)
                print('d90qunatile:', d90quantile)
                print('c25quantile:', c25quantile)
                print('c90qunatile:', c90quantile)
                print('avg update percentage:', updateavg)
                # print(c) # starts from 0 

            yamldict = {}
            yamldict['d25quantile'] = [x.item() for x in d25quantilelist]
            yamldict['d90quantile'] = [x.item() for x in d90quantilelist]
            yamldict['c25quantile'] = [x.item() for x in c25quantilelist]
            yamldict['c90quantile'] = [x.item() for x in c90quantilelist]
            yamldict['updateavg'] = [x.item() for x in updateavglist]
            yamldict['noise_level'] = [x.item() for x in sigmas]

            if denoising_step == True:
                last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
                last_noise = last_noise.long()
                x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
                # for only showing the inpainting points
                x_mod[:, :, mask] = refer_image[..., mask]
                images.append(x_mod.to('cpu'))

            import yaml
            with open(os.path.join(self.args.image_folder, 'sampleresult.yml'), 'w') as f:
                documents = yaml.dump(yamldict, f)

            sampleresultoutputfolder = os.path.join(self.args.image_folder, 'sampleresultPlot')
            plot_sampleresult(os.path.join(self.args.image_folder, 'sampleresult.yml'), sampleresultoutputfolder)
            
            return images

    def test_inpainting(self, n_steps_each=100, step_lr=0.00002, batch_size=10, noarb=False, mask_choice="2", sampmethod=2, denoising_step=True):
        # n_steps_each number of steps for sampling in each noise level
        # step_lr learning rate for each step
        # batch_size is actually the number of test samples to be considered 

        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))
        score.eval()

        imgs = []

        width = self.config.data.image_size
        
        if self.config.data.dataset == 'Finance':
            # tlist = scipy.io.loadmat('msft_tlist.mat')

            string1 = self.config.data.datasetpath.split("HestonIVSgrid_NI")[1]
            datasetsizestr = string1.split('.mat')[0]
            
            tlist = scipy.io.loadmat(self.config.data.finname + '_tlist' + datasetsizestr + '.mat')
            tlist = tlist['tlist'] # note that they are 2D array 
            tlist = tlist.reshape((-1))
            tlist_broadcast = tlist.reshape((1, len(tlist), 1))
            # Klist = scipy.io.loadmat('msft_Klist.mat')
            Klist = scipy.io.loadmat(self.config.data.finname + '_Klist' + datasetsizestr + '.mat')
            Klist = Klist['Klist'] # note that they are 2D array
            Klist = Klist.reshape((-1))

            # totalvarsurface = scipy.io.loadmat('msft_hestonlike_IVS.mat') # the address is relative to the address of thescript that calls this class
            totalvarsurface = scipy.io.loadmat(self.config.data.datasetpath)
            # totalvarsurface = totalvarsurface['totalvarsurface']
            totalvarsurface = totalvarsurface['HestonIVS2D']

            if self.config.data.scale:
                # if scale is true, we use total implied variance instead! 
                totalvarsurface = totalvarsurface**2 * tlist_broadcast

            IVStrain, IVStest = train_test_split(totalvarsurface, test_size = 0.2, random_state = 42)

            dataset = Dset(IVStest)
            # batch_size = 10
            # batch_size = 1282 # all data in IVStest
            # batch_size = 2
            dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False, drop_last=True)
            # we use test dataset, use the same variable dataloader to avoid codes changes 

            refer_image = next(iter(dataloader))

            refer_image = refer_image[:, None, :, :] # to match the dimension
            refer_image = refer_image.to(self.config.device)

            if batch_size == 1:
                savepath = os.path.join(self.args.image_folder, 'originalIVS_inpainting.png')
                if self.config.data.scale:
                    savepath2 = os.path.join(self.args.image_folder, 'originaltotalvar_inpainting.png')
                    IVS_visualize(refer_image.reshape((width,width)), Klist, tlist, savepath=savepath2, plotname = "original total var")
                    refer_image_unscaled = torch.sqrt(refer_image.cpu() / tlist_broadcast)
                    IVS_visualize(refer_image_unscaled.reshape((width,width)), Klist, tlist, savepath=savepath, plotname = "original IV")
                else:
                    IVS_visualize(refer_image.reshape((width,width)), Klist, tlist, savepath=savepath, plotname = "original IV")

            # sample from uniform?! why

            # samples = torch.rand(batch_size, batch_size, 1, self.config.data.image_size, self.config.data.image_size,
            #                      device=self.config.device)

            samples = torch.rand(batch_size, 1, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)
                                 
            mask = torch.ones(self.config.data.image_size,self.config.data.image_size, dtype=torch.bool)
            if mask_choice == "2":
                mask[1,1] = False
                mask[6,6] = False
            elif mask_choice == "half":  
                indexlist = [(1,1), (1,3), (1,5), (2,2), (2,4), (2,6), (3,1), (3,3), (3,5),
                             (4,2), (4,4), (4,6), (5,1), (5,3), (5,5), (6,2), (6,4), (6,6)]
                for x in indexlist:
                    mask[x] = False
            elif mask_choice == "out1":
                indexlist = [(0,0), (0,2), (0,4), (0,6), (1,7), (3,7), (5,7), (7,7),
                             (2,0), (4,0), (6,0), (7,1), (7,3), (7,5)]
                for x in indexlist:
                    mask[x] = False
            elif mask_choice == "out2":
                indexlist1 = [(0,x) for x in range(8)] + [(x,0) for x in range(8)]
                indexlist2 = [(7,x) for x in range(8)] + [(x,7) for x in range(8)]
                indexlist = indexlist1 + indexlist2
                for x in indexlist:
                    mask[x] = False
            elif mask_choice == "mid4":
                indexlist =  [(2,2), (2,5), (4,2), (4,5)]
                for x in indexlist: 
                    mask[x] = False
            elif mask_choice == "mid4v2":
                i_d = int(np.ceil(self.config.data.image_size/4)) # i_d for indexdist
                indexlist = [(i_d, i_d), (i_d, -i_d-1), (-i_d-1, i_d), (-i_d-1,-i_d-1)]
                # print('Hi!!!!!!', i_d)
                for x in indexlist: 
                    mask[x] = False
            elif mask_choice == "random10": # 10 for around 10% of the points are randomly blackened
                numBpts = int(np.floor(0.1 * self.config.data.image_size **2))
                indexlist = [(i,j) for i in range(self.config.data.image_size) for j in range(self.config.data.image_size)]
                indexlist = random.sample(indexlist, numBpts)
                for x in indexlist: 
                    mask[x] = False
            elif mask_choice == "random20": # 10 for around 10% of the points are randomly blackened
                numBpts = int(np.floor(0.2 * self.config.data.image_size **2))
                indexlist = [(i,j) for i in range(self.config.data.image_size) for j in range(self.config.data.image_size)]
                indexlist = random.sample(indexlist, numBpts)
                for x in indexlist: 
                    mask[x] = False
            elif mask_choice == "random30": # 10 for around 10% of the points are randomly blackened
                numBpts = int(np.floor(0.3 * self.config.data.image_size **2))
                indexlist = [(i,j) for i in range(self.config.data.image_size) for j in range(self.config.data.image_size)]
                indexlist = random.sample(indexlist, numBpts)
                for x in indexlist: 
                    mask[x] = False
            elif mask_choice == "random40": # 10 for around 10% of the points are randomly blackened
                numBpts = int(np.floor(0.4 * self.config.data.image_size **2))
                indexlist = [(i,j) for i in range(self.config.data.image_size) for j in range(self.config.data.image_size)]
                indexlist = random.sample(indexlist, numBpts)
                for x in indexlist: 
                    mask[x] = False
            elif mask_choice == "random50": # 10 for around 10% of the points are randomly blackened
                numBpts = int(np.floor(0.5 * self.config.data.image_size **2))
                indexlist = [(i,j) for i in range(self.config.data.image_size) for j in range(self.config.data.image_size)]
                indexlist = random.sample(indexlist, numBpts)
                for x in indexlist: 
                    mask[x] = False
            elif mask_choice == "random60": # 10 for around 10% of the points are randomly blackened
                numBpts = int(np.floor(0.6 * self.config.data.image_size **2))
                indexlist = [(i,j) for i in range(self.config.data.image_size) for j in range(self.config.data.image_size)]
                indexlist = random.sample(indexlist, numBpts)
                for x in indexlist: 
                    mask[x] = False
            elif mask_choice == "random70": # 10 for around 10% of the points are randomly blackened
                numBpts = int(np.floor(0.7 * self.config.data.image_size **2))
                indexlist = [(i,j) for i in range(self.config.data.image_size) for j in range(self.config.data.image_size)]
                indexlist = random.sample(indexlist, numBpts)
                for x in indexlist: 
                    mask[x] = False
            elif mask_choice == "random80": # 10 for around 10% of the points are randomly blackened
                numBpts = int(np.floor(0.8 * self.config.data.image_size **2))
                indexlist = [(i,j) for i in range(self.config.data.image_size) for j in range(self.config.data.image_size)]
                indexlist = random.sample(indexlist, numBpts)
                for x in indexlist: 
                    mask[x] = False
            elif mask_choice == "random90": # 10 for around 10% of the points are randomly blackened
                numBpts = int(np.floor(0.9 * self.config.data.image_size **2))
                indexlist = [(i,j) for i in range(self.config.data.image_size) for j in range(self.config.data.image_size)]
                indexlist = random.sample(indexlist, numBpts)
                for x in indexlist: 
                    mask[x] = False

            import yaml 
            newdict = self.config
            newdict.sampling = argparse.Namespace()
            # newdict.sampling.n_steps_each = n_steps_each
            # newdict.sampling.step_lr = step_lr
            # newdict.sampling.mask_choice = mask_choice
            # newdict.sampling.testsize = batch_size
            # newdict.sampling.sampmethod = sampmethod

            setattr(newdict.sampling, 'noarb', noarb)
            setattr(newdict.sampling, 'n_steps_each', n_steps_each)
            setattr(newdict.sampling, 'step_lr', step_lr)
            setattr(newdict.sampling, 'mask_choice', mask_choice)
            setattr(newdict.sampling, 'testsize', batch_size)
            setattr(newdict.sampling, 'sampmethod', sampmethod)

            # newdict.noarb = noarb
            # newdict.n_steps_each = n_steps_each
            # newdict.step_lr = step_lr
            # newdict.mask_choice = mask_choice
            # newdict.testsize = batch_size
            # newdict.sampmethod = sampmethod

            with open(os.path.join(self.args.image_folder, 'config.yml'), 'w') as f:
                documents = yaml.dump(newdict, f)
            

            # all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, mask, 100, 0.00002)
            # all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, mask, 100, 0.000008)
            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, mask, n_steps_each, step_lr, noarb, sampmethod)

            if denoising_step:
                denoised_sample = all_samples[-1]
                all_samples = all_samples[:-1]

            # torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))

            if self.config.data.scale:
                refer_image_unscaled = torch.sqrt(refer_image.cpu() / tlist_broadcast)
                # refer_image = torch.sqrt(refer_image.cpu() / tlist_broadcast)
            


            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                # sample = sample.view(batch_size**2, self.config.data.channels, self.config.data.image_size,
                #                      self.config.data.image_size)
                sample = sample.view(batch_size, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)
                
                if self.config.data.scale:
                    sample_unscaled = torch.sqrt(sample.cpu()/tlist_broadcast)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                # save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                
                if batch_size ==1:
                    savepath = os.path.join(self.args.image_folder, 'ivs_image_inpainting{}.png'.format(i))
                    IVS_visualize(sample.reshape((self.config.data.image_size,self.config.data.image_size)), Klist, tlist, savepath=savepath, plotname = str(i) + "step_inpainting")

                    # print(type(sample))
                    # print(sample.shape)
                    # print(type(refer_image))
                    # print(refer_image.shape)
                    if i > n_steps_each * self.config.model.num_classes - 10: 
                        savepath2 = os.path.join(self.args.image_folder, 'ivs_error_inpainting{}.png'.format(i))
                        inpainting_error(refer_image.reshape((batch_size**2, self.config.data.image_size,self.config.data.image_size)).cpu(),
                             sample.reshape((batch_size**2, self.config.data.image_size,self.config.data.image_size)).cpu(),
                               Klist, tlist, savepath2, mask)
                else: # batch_size == 10  
                    if i > n_steps_each * self.config.model.num_classes - 10: 
                        if self.config.data.scale:
                            savepath = os.path.join(self.args.image_folder, 'totalvar_image_inpainting{}.png'.format(i))
                            IVS_visualize(sample[0].reshape((width,width)), Klist, tlist, savepath=savepath, plotname = str(i) + "step_inpainting")
                            savepath2 = os.path.join(self.args.image_folder, 'totalvar_error_inpainting{}.png'.format(i))
                            savepathyml2 = os.path.join(self.args.image_folder, 'totalvar_error_inpainting{}.yml'.format(i))
                            inpainting_error(refer_image.reshape((batch_size, width,width)).cpu(),
                                            sample.reshape((batch_size, width,width)).cpu(), Klist, tlist, savepath2, mask, savepathyml2)
                            savepath3 = os.path.join(self.args.image_folder, 'ivs_image_inpainting{}.png'.format(i))
                            IVS_visualize(sample_unscaled[0].reshape((width,width)), Klist, tlist, savepath=savepath3, plotname = str(i) + "step_inpainting")
                            savepath4 = os.path.join(self.args.image_folder, 'ivs_error_inpainting{}.png'.format(i))
                            savepathyml4 = os.path.join(self.args.image_folder, 'ivs_error_inpainting{}.yml'.format(i))
                            inpainting_error(refer_image_unscaled.reshape((batch_size, width,width)).cpu(),
                                            sample_unscaled.reshape((batch_size, width,width)).cpu(), Klist, tlist, savepath4, mask, savepathyml4)
                            
                        else:
                            savepath = os.path.join(self.args.image_folder, 'ivs_image_inpainting{}.png'.format(i))
                            # IVS_visualize(sample[0].reshape((width,width)), Klist, tlist, savepath=savepath, plotname = str(i) + "step_inpainting")
                            IVS_visualize(sample[81].reshape((width,width)), Klist, tlist, savepath=savepath, plotname = str(i) + "step_inpainting")

                            savepath2 = os.path.join(self.args.image_folder, 'ivs_error_inpainting{}.png'.format(i))
                            savepathyml = os.path.join(self.args.image_folder, 'ivs_error_inpainting{}.yml'.format(i))
                            inpainting_error(refer_image.reshape((batch_size, width,width)).cpu(),
                                            sample.reshape((batch_size, width,width)).cpu(), Klist, tlist, savepath2, mask, savepathyml)
                        # a = sample.reshape((batch_size, self.config.data.image_size,self.config.data.image_size)).cpu()
                        # b = refer_image.reshape((batch_size, self.config.data.image_size,self.config.data.image_size)).cpu()
                        # print(torch.sum(torch.abs(a-b), dim=(0,1,2)))

                # else:
                #     # if i > 800:
                #     if i > n_steps_each * self.config.model.num_classes * 0:
                #         savepath = os.path.join(self.args.image_folder, 'ivs_error_inpainting{}.png'.format(i))
                #         inpainting_error(refer_image.reshape((batch_size, self.config.data.image_size,self.config.data.image_size)).cpu(),
                #                          sample.reshape((batch_size, self.config.data.image_size,self.config.data.image_size)).cpu(), Klist, tlist, savepath, )
                
                # torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))
            if denoising_step:
                ## assume not scaled 
                if not self.config.data.scale:
                    savepath = os.path.join(self.args.image_folder, 'ivs_image_inpainting_denoised.png')
                    # IVS_visualize(denoised_sample[0].reshape((width,width)), Klist, tlist, savepath=savepath, plotname = str(i) + "step_inpainting")
                    IVS_visualize(denoised_sample[81].reshape((width,width)), Klist, tlist, savepath=savepath, plotname = "Denoised " + str(i) + "step_inpainting")

                    savepath2 = os.path.join(self.args.image_folder, 'ivs_error_inpainting_denoised.png')
                    savepathyml = os.path.join(self.args.image_folder, 'ivs_error_inpainting_denoised.yml')
                    inpainting_error(refer_image.reshape((batch_size, width,width)).cpu(),
                                    denoised_sample.reshape((batch_size, width,width)).cpu(), Klist, tlist, savepath2, mask, savepathyml)
        
        elif self.config.data.dataset == 'CELEBA':
            dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(self.config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            refer_image, _ = next(iter(dataloader))

            # sample from uniform?! why
            samples = torch.rand(20, 20, 3, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)
            mask = torch.ones(self.config.data.image_size,self.config.data.image_size, dtype=torch.bool)
            mask[1,1] = False
            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)
            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=batch_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))

        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'CIFAR10':
                dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                                  transform=transform)
            elif self.config.data.dataset == 'SVHN':
                dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                               transform=transform)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            refer_image, _ = next(data_iter)

            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))
            samples = torch.rand(20, 20, self.config.data.channels, self.config.data.image_size,
                                 self.config.data.image_size).to(self.config.device)
            mask = torch.ones(self.config.data.image_size,self.config.data.image_size, dtype=torch.bool)
            mask[1,1] = False
            mask[6,6] = False
            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))


        imgs[0].save(os.path.join(self.args.image_folder, "inpainting_movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
