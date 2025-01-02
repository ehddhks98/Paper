import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_fakes(netG, n=1, sp500max=1, params=(0, 1, 0), sp500log_mean=0, device='cpu', cumsum=True):
    fakes = []
    for i in range(n):
        noise = torch.randn(1, 2432, 3, device=device)
        fake = netG(noise).detach().cpu().reshape(2432).numpy()
        sp500fake = inverse(fake * sp500max, params) + sp500log_mean
        fakes.append(sp500fake)
    if n > 1:
        if not cumsum:
            return pd.DataFrame(fakes).T
        fakes_df = pd.DataFrame(fakes).T.cumsum()
        return fakes_df
    elif not cumsum:
        return sp500fake
    return sp500fake.cumsum()

def plot_generated_paths(netG, device, sp500max, params, sp500log_mean):
    plt.figure(figsize=(10,6))
    plt.plot(generate_fakes(netG, 5, sp500max, params, sp500log_mean, device), linewidth=1, alpha=0.7)
    plt.grid(True)
    plt.xlabel('T (number of days)')
    plt.ylabel('Log path')
    plt.title('5 generated log paths')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(generate_fakes(netG, 50, sp500max, params, sp500log_mean, device), linewidth=1, alpha=0.7)
    plt.grid(True)
    plt.xlabel('T (number of days)')
    plt.ylabel('Log path')
    plt.title('50 generated log paths')
    plt.show()
