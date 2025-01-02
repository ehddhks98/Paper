import torch
import torch.optim as optim
from tqdm import tqdm

def train(netG, netD, dataloader, device, num_epochs=50, nz=3, lr=0.0002, clip_value=0.01):
    optD = optim.RMSprop(netD.parameters(), lr=lr)
    optG = optim.RMSprop(netG.parameters(), lr=lr)
    t = tqdm(range(num_epochs))

    for epoch in t:
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real = data.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            noise = torch.randn(batch_size, seq_len, nz, device=device)
            fake = netG(noise).detach()

            lossD = -torch.mean(netD(real)) + torch.mean(netD(fake))
            lossD.backward()
            optD.step()

            for p in netD.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if i % 5 == 0:
                netG.zero_grad()
                lossG = -torch.mean(netD(netG(noise)))
                lossG.backward()
                optG.step()            

        t.set_description(f'Loss_D: {lossD.item():.8f} Loss_G: {lossG.item():.8f}')
        torch.save(netG, f'sp500_netG_epoch_{epoch}.pth')
        torch.save(netD, f'sp500_netD_epoch_{epoch}.pth')
