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
        totalvarsurface = scipy.io.loadmat('HestonIVSgrid_NI.mat')
        totalvarsurface = totalvarsurface['HestonIVS2D']

        if self.config.data.scale:
            # if scale is true, we use total implied variance instead! 

            tlist = scipy.io.loadmat(self.config.data.finname + '_tlist.mat')
            tlist = tlist['tlist'] # note that they are 2D array 
            tlist = tlist.reshape((-1))
            # Klist = scipy.io.loadmat('msft_Klist.mat')
            Klist = scipy.io.loadmat(self.config.data.finname + '_Klist.mat')
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
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 20 == 0:
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
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                                                                    self.config.training.anneal_power)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
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
        tlist = scipy.io.loadmat(self.config.data.finname + '_tlist.mat')
        tlist = tlist['tlist'] # note that they are 2D array 
        tlist = tlist.reshape((-1))
        tlist_broadcast = tlist.reshape((1, len(tlist), 1))
        # Klist = scipy.io.loadmat('msft_Klist.mat')
        Klist = scipy.io.loadmat(self.config.data.finname + '_Klist.mat')
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
                        sample = torch.sqrt(sample/ tlist_broadcast)
                    IVS_visualize(sample.reshape((self.config.data.image_size,self.config.data.image_size)), Klist, tlist, savepath=savepath, plotname = str(i) + "step")
                if i == 999:
                    print('step', i)
                    torch.set_printoptions(precision=10)
                    print(sample)
                if i == 0:
                    # totalvarsurface = scipy.io.loadmat('msft_hestonlike_IVS.mat') # the address is relative to the address of thescript that calls this class
                    totalvarsurface = scipy.io.loadmat('HestonIVSgrid_NI.mat')
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
                                            step_lr=0.000008):
        images = []

        refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_image = refer_image.contiguous().view(-1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)

        x_mod = x_mod.view(-1, self.config.data.channels, self.config.data.image_size ,self.config.data.image_size)
        half_refer_image = refer_image[..., :self.config.data.image_size//2]
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="annealed Langevin dynamics sampling inpainting"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                # corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                corrupted_half_image = half_refer_image
                x_mod[:, :, mask] = refer_image[..., mask] # one more step compared to the algo 
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # x_mod[:, :, :, :16] = corrupted_half_image
                    x_mod[:, :, mask] = refer_image[..., mask]
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def test_inpainting(self):
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
        
        if self.config.data.dataset == 'Finance':
            # tlist = scipy.io.loadmat('msft_tlist.mat')
            tlist = scipy.io.loadmat(self.config.data.finname + '_tlist.mat')
            tlist = tlist['tlist'] # note that they are 2D array 
            tlist = tlist.reshape((-1))
            tlist_broadcast = tlist.reshape((1, len(tlist), 1))
            # Klist = scipy.io.loadmat('msft_Klist.mat')
            Klist = scipy.io.loadmat(self.config.data.finname + '_Klist.mat')
            Klist = Klist['Klist'] # note that they are 2D array
            Klist = Klist.reshape((-1))

            # totalvarsurface = scipy.io.loadmat('msft_hestonlike_IVS.mat') # the address is relative to the address of thescript that calls this class
            totalvarsurface = scipy.io.loadmat('HestonIVSgrid_NI.mat')
            # totalvarsurface = totalvarsurface['totalvarsurface']
            totalvarsurface = totalvarsurface['HestonIVS2D']

            if self.config.data.scale:
                # if scale is true, we use total implied variance instead! 
                totalvarsurface = totalvarsurface**2 * tlist_broadcast

            IVStrain, IVStest = train_test_split(totalvarsurface, test_size = 0.2, random_state = 42)

            dataset = Dset(IVStest)
            # batch_size = 10
            batch_size = 1282 # all data in IVStest
            dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True, drop_last=True)
            # we use test dataset, use the same variable dataloader to avoid codes changes 

            refer_image = next(iter(dataloader))

            refer_image = refer_image[:, None, :, :] # to match the dimension
            refer_image = refer_image.to(self.config.device)

            if batch_size == 1:
                savepath = os.path.join(self.args.image_folder, 'originalIVS_inpainting.png')
                IVS_visualize(refer_image.reshape((self.config.data.image_size,self.config.data.image_size)), Klist, tlist, savepath=savepath, plotname = "original")


            # sample from uniform?! why

            # samples = torch.rand(batch_size, batch_size, 1, self.config.data.image_size, self.config.data.image_size,
            #                      device=self.config.device)

            samples = torch.rand(batch_size, 1, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)
                                 
            mask = torch.ones(self.config.data.image_size,self.config.data.image_size, dtype=torch.bool)
            mask[1,1] = False
            mask[6,6] = False
            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, mask, 100, 0.00002)
            # all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, mask, 100, 0.000008)

            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))

            if self.config.data.scale:
                refer_image = torch.sqrt(refer_image / tlist_broadcast)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                # sample = sample.view(batch_size**2, self.config.data.channels, self.config.data.image_size,
                #                      self.config.data.image_size)
                sample = sample.view(batch_size, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)


                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                
                if batch_size ==1:
                    savepath = os.path.join(self.args.image_folder, 'ivs_image_inpainting{}.png'.format(i))
                    IVS_visualize(sample.reshape((self.config.data.image_size,self.config.data.image_size)), Klist, tlist, savepath=savepath, plotname = str(i) + "step_inpainting")

                    # print(type(sample))
                    # print(sample.shape)
                    # print(type(refer_image))
                    # print(refer_image.shape)
                    if i > 800: 
                        savepath2 = os.path.join(self.args.image_folder, 'ivs_error_inpainting{}.png'.format(i))
                        inpainting_error(sample.reshape((batch_size**2, self.config.data.image_size,self.config.data.image_size)).cpu(), 
                            refer_image.reshape((batch_size**2, self.config.data.image_size,self.config.data.image_size)).cpu(), Klist, tlist, savepath2, )
                else:
                    if i > 800:
                        savepath = os.path.join(self.args.image_folder, 'ivs_error_inpainting{}.png'.format(i))
                        if self.config.data.scale:
                            sample = torch.sqrt(sample/ tlist_broadcast)
                        inpainting_error(sample.reshape((batch_size, self.config.data.image_size,self.config.data.image_size)).cpu(),
                                          refer_image.reshape((batch_size, self.config.data.image_size,self.config.data.image_size)).cpu(), Klist, tlist, savepath, )
                
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))
        
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
