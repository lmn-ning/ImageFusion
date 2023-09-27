import math
import numpy as np
import matplotlib.pyplot as plt


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
    return betas


if __name__ == '__main__':
    T = 2000
    x = np.linspace(1, 2000, num=2000)
    y = cosine_beta_schedule(T)

    plt.plot(x, y)
    plt.show()