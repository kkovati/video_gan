import cairo
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


class SquareDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        np.random.seed(0)
        self.points = np.random.rand(size, 2)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        # https://stackoverflow.com/questions/10031580/how-to-write-simple-geometric-shapes-into-numpy-arrays
        x, y = (self.points[idx] * 100).astype(int)
        img = np.zeros((128, 128), dtype=float)
        img[y:y + 20, x:x + 20] = 1
        return img


class Vae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        latent_size = 1000

        self.enc_conv_0 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

        self.enc_lin_0 = torch.nn.Linear(1 * 126 * 126, int(1 * 126 * 126 / 20))
        self.enc_lin_1 = torch.nn.Linear(int(1 * 126 * 126 / 20), latent_size)

        self.dec_lin_0 = torch.nn.Linear(latent_size, int(1 * 126 * 126 / 20))
        self.dec_lin_1 = torch.nn.Linear(int(1 * 126 * 126 / 20), 1 * 126 * 126)

        self.dec_conv_0 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.enc_conv_0(x)
        x = torch.tanh(x)

        x = torch.flatten(x, start_dim=1)
        x = self.enc_lin_0(x)
        x = torch.tanh(x)
        x = self.enc_lin_1(x)
        x = torch.tanh(x)
        l = x

        x = self.dec_lin_0(x)
        x = torch.tanh(x)
        x = self.dec_lin_1(x)
        x = torch.tanh(x)

        x = torch.reshape(x, (x.shape[0], 1, 126, 126))

        x = self.dec_conv_0(x)
        x = torch.sigmoid(x)

        return x, l


class VaeV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        latent_size = 128

        self.enc_conv_0 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.enc_conv_1 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.enc_conv_2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.enc_conv_3 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.enc_conv_4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)

        self.enc_lin_0 = torch.nn.Linear(8 * 4 * 4, latent_size)

        self.dec_lin_0 = torch.nn.Linear(latent_size, 8 * 4 * 4)

        self.dec_conv_0 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.dec_conv_1 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.dec_conv_2 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.dec_conv_3 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.dec_conv_4 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1)

        # self._enc_mu = torch.nn.Linear(100, 8)
        # self._enc_log_sigma = torch.nn.Linear(100, 8)

    # def _sample_latent(self, h_enc):
    #     mu = self._enc_mu(h_enc)
    #     log_sigma = self._enc_log_sigma(h_enc)
    #     sigma = torch.exp(log_sigma)
    #     std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
    #
    #     self.z_mean = mu
    #     self.z_sigma = sigma

    # return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x):
        x = self.enc_conv_0(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.enc_conv_1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.enc_conv_2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.enc_conv_3(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.enc_conv_4(x)
        x = torch.nn.functional.leaky_relu(x)

        x = torch.flatten(x, start_dim=1)
        x = self.enc_lin_0(x)
        x = torch.sigmoid(x)
        l = x

        x = self.dec_lin_0(x)
        x = torch.reshape(x, (x.shape[0], 8, 4, 4))
        # x = torch.unsqueeze(x, dim=2)
        # x = torch.unsqueeze(x, dim=3)

        x = self.dec_conv_0(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dec_conv_1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dec_conv_2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dec_conv_3(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dec_conv_4(x)
        x = torch.sigmoid(x)

        # h_enc = self.encoder(state)
        # z = self._sample_latent(h_enc)
        return x, l


def latent_mean_loss(latent_batch):
    return torch.mean(torch.abs(torch.mean(latent_batch, dim=0)))


def latent_var_loss(latent_batch, device):
    # return torch.mean(torch.abs(torch.ones(latent_batch.shape[1]) - torch.var(latent_batch, dim=0)))
    return torch.mean(torch.abs(torch.subtract(torch.var(latent_batch, dim=0), 1)))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Available device: {device.type}")

    square_dataset = SquareDataset(5000)

    train_dataloader = DataLoader(square_dataset, batch_size=32, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    print("Init VAE")
    vae = VaeV2().to(device)

    # reconstruction_loss = torch.nn.MSELoss()
    reconstruction_loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)

    print("Start training")
    n_epoch = 100
    for i_epoch in range(n_epoch):
        avg_lml, avg_lvl, avg_rl = 0, 0, 0
        for i_batch, img_batch in enumerate(train_dataloader):
            img_batch = torch.unsqueeze(img_batch, dim=1).float().to(device)
            optimizer.zero_grad()
            recon_img_batch, latent_batch = vae(img_batch)

            lml = latent_mean_loss(latent_batch)
            lvl = latent_var_loss(latent_batch, device)
            rl = reconstruction_loss(recon_img_batch, img_batch)

            # net_loss = lml + lvl + rl
            net_loss = rl

            net_loss.backward()
            optimizer.step()

            avg_lml += lml
            avg_lvl += lvl
            avg_rl += rl

            if i_batch == len(train_dataloader) - 1:
                f, axarr = plt.subplots(1, 2)
                axarr[0].imshow(np.squeeze(img_batch[0].detach().cpu().numpy()), vmin=0, vmax=1)
                axarr[1].imshow(np.squeeze(recon_img_batch[0].detach().cpu().numpy()), vmin=0, vmax=1)
                plt.savefig(f"epoch_{i_epoch + 1}.png")
                plt.clf()

        avg_lml /= len(train_dataloader)
        avg_lvl /= len(train_dataloader)
        avg_rl /= len(train_dataloader)

        print(f"Epoch {i_epoch + 1}/{n_epoch}")
        print(f"avg_lml = {avg_lml:.6f}, avg_lvl =  {avg_lvl:.6f}, avg_rl = {avg_rl:.6f}")

    # https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    # https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
