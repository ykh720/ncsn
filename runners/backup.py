def anneal_Langevin_dynamics_inpainting(self, x_mod, refer_image, scorenet, sigmas, n_steps_each=100,
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
            x_mod[:, :, :, :self.config.data.image_size//2] = corrupted_half_image # one more step compared to the algo 
            for s in range(n_steps_each):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                # x_mod[:, :, :, :16] = corrupted_half_image
                x_mod[:, :, :, :self.config.data.image_size//2] = corrupted_half_image
                # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                #                                                          grad.abs().max()))

        return images
    

### Back up for noise annealing bias sampling, that is we only pick sample with loss smaller than a threshold(noise level)

def anneal_Langevin_dynamics_inpainting_noarb_Feb28(self, x_mod, refer_image, scorenet, sigmas, mask, n_steps_each=100,
                                        step_lr=0.000008, noarb = False):
    
    tlist = scipy.io.loadmat(self.config.data.finname + '_tlist.mat')
    tlist = tlist['tlist'] # note that they are 2D array 
    tlist = tlist.reshape((-1))
    tlist_broadcast = tlist.reshape((1, len(tlist), 1))
    # Klist = scipy.io.loadmat('msft_Klist.mat')
    Klist = scipy.io.loadmat(self.config.data.finname + '_Klist.mat')
    Klist = Klist['Klist'] # note that they are 2D array
    Klist = Klist.reshape((-1))
    # note that S0 = 100
    logmlist = np.log(Klist / 100)

    # these should be treated as inputs, or in the future deduce a independent conditions that is based on sigma
    # also note that this change makes the code only applicable to num_class = 10 for now.
    dlossthreshold = torch.tensor([7.1260, 3.8, 2.2, 1.4, 0.9, 0.55, 0.23, 0.07, 0.01, 0])
    clossthreshold = torch.tensor([3.9, 2.3, 1.34, 0.76, 0.42, 0.21, 0.09, 0.03, 0.005, 0.0312])

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

            dlosslist = []
            closslist = []

            for s in range(n_steps_each):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)

                # print(x_mod.shape)
                x_prop = x_mod + step_size * grad + noise
                dloss = Dcondloss_torch(torch.squeeze(x_prop), torch.squeeze(x_prop), logmlist) # .unsqueeze(0)
                closs = calloss_torch(torch.squeeze(x_prop), torch.squeeze(x_prop), tlist) # .unsqueeze(0)
                dlosslist.append(dloss)
                closslist.append(closs)
                
                if noarb: 
                    index = (dloss < dlossthreshold[c]) & (closs < clossthreshold[c])
                    x_mod[index] = x_prop[index]
                else: 
                    x_mod = x_prop
                    
                # x_mod[:, :, :, :16] = corrupted_half_image
                x_mod[:, :, mask] = refer_image[..., mask]
                # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                #                                                          grad.abs().max()))

            print('\n current sigma' , sigma)
            dloss_thislevel = torch.cat(dlosslist, dim=0)
            closs_thislevel = torch.cat(closslist, dim=0)
            d25quantile = torch.quantile(dloss_thislevel, 0.25)
            c25quantile = torch.quantile(closs_thislevel, 0.25)

            print('d25quantile:', d25quantile)
            print('c25quantile:', c25quantile)
            # print(c) # starts from 0 

        return images