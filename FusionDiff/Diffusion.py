# Denoising diffusion probabilistic models (DDPM)

import torch
import math
import torch.nn.functional as F
import os
import cv2
import time

from utils import tensor2img

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps):
    return betas_for_alpha_bar(
        timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )


# Create a beta schedule that discretizes the given alpha_t_bar function
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


class GaussianDiffusion:
    def __init__(
            self,
            timesteps=2000,
            beta_schedule='cosine'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )

    # Get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None, return_noise=False):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        if return_noise:
            return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
        else:
            return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # Compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # Compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, sourceImg1, sourceImg2, x_t, t, concat_type, clip_denoised=True):
        # Concatenation is performed in the channel dimension
        if concat_type == "ABX":
            input = torch.cat([sourceImg1, sourceImg2, x_t], dim=1)
        if concat_type == "XAB":
            input = torch.cat([x_t, sourceImg1, sourceImg2], dim=1)
        if concat_type == "AXB":
            input = torch.cat([sourceImg1, x_t, sourceImg2], dim=1)
        pred_noise = model(input, t)

        # Get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, sourceImg1, sourceImg2, x_t, t, concat_type, add_noise, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(
            model, sourceImg1, sourceImg2, x_t, t, concat_type, clip_denoised=clip_denoised)

        # Random noise is added except for t=0 steps
        if add_noise:
            noise = torch.randn_like(x_t)
            # no noise when t == 0
            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
            # Compute x_{t-1}
            pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            return pred_img
        else:
            return model_mean

    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, sourceImg1, sourceImg2, concat_type, add_noise, log_info):
        step, valid_step_sum, num, generat_imgs_num = log_info
        log_step = 100

        # Start from pure noise (for each example in the batch)
        imgs = torch.randn(sourceImg1.shape, device=device)

        # reverse process
        for i in reversed(range(0, self.timesteps)):
            if i % log_step == 0:
                now_time = time.strftime('%Y%m%d_%H%M%S')
                print(f"[valid step] {int((step - 1) / sourceImg1.shape[0]) + 1}/{valid_step_sum}    "
                      f"[generate step] {num + 1}/{generat_imgs_num}    "
                      f"[reverse process] {i}/{self.timesteps}    "
                      f"[time] {now_time}")
            t = torch.full((sourceImg1.shape[0],), i, device=device, dtype=torch.long)
            imgs = self.p_sample(model, sourceImg1, sourceImg2, imgs, t, concat_type, add_noise)
        return imgs

    # Sample new images
    @torch.no_grad()
    def sample(self, model, sourceImg1, sourceImg2, add_noise, concat_type, model_name, model_path,
               generat_imgs_num, step, timestr, valid_step_sum, dataset_name):
        extension_list = ["jpg", "tif", "png", "jpeg"]
        for num in range(generat_imgs_num):
            log_info = [step, valid_step_sum, num, generat_imgs_num]
            imgs = self.p_sample_loop(model, sourceImg1, sourceImg2, concat_type, add_noise, log_info)
            for i in range(imgs.shape[0]):
                img_id = step + i
                dirPath = os.path.join("generate_imgs",
                                       dataset_name,
                                       timestr,
                                       model_name,
                                       )

                # Save images in multiple formats
                image = tensor2img(imgs[i])
                for extension in extension_list:
                    subdirPath = os.path.join(dirPath, extension + "_imgs")
                    if not os.path.exists(subdirPath):
                        os.makedirs(subdirPath)

                    # valid log
                    valid_log_path = os.path.join(subdirPath, "valid_log.txt")
                    valid_log = open(valid_log_path, "w")
                    valid_log.write(f"time: {timestr} \n")
                    valid_log.write(f"model_path: {model_path} \n")

                    # Save imgs
                    if generat_imgs_num == 1:
                        if img_id < 10:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_0" + str(
                                                             img_id) + "." + extension)
                        else:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_" + str(
                                                             img_id) + "." + extension)
                        cv2.imwrite(img_file_path, image)
                    else:
                        if img_id < 10:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_0" + str(
                                                             img_id) + "_num" + str(
                                                             num) + "." + extension)
                        else:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_" + str(
                                                             img_id) + "_num" + str(
                                                             num) + "." + extension)
                        cv2.imwrite(img_file_path, image)

    # Compute train losses
    def train_losses(self, model, sourceImg1, sourceImg2, x_start, t, concat_type, loss_scale):
        noise = torch.randn_like(x_start)
        # Get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)

        # Concatenation is performed in the channel dimension
        if concat_type == "ABX":
            input = torch.cat([sourceImg1, sourceImg2, x_noisy], dim=1)
        if concat_type == "XAB":
            input = torch.cat([x_noisy, sourceImg1, sourceImg2], dim=1)
        if concat_type == "AXB":
            input = torch.cat([sourceImg1, x_noisy, sourceImg2], dim=1)
        predicted_noise = model(input, t)
        assert predicted_noise.shape == noise.shape
        return loss_scale * F.mse_loss(noise, predicted_noise)
