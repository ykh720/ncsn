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