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
        img = np.zeros((128, 128), dtype=int)
        img[y:y + 20, x:x + 20] = 255
        return img


class Vae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv_0 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2)
        self.enc_conv_1 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2)
        self.enc_conv_2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2)
        self.enc_conv_3 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2)
        self.enc_conv_4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=6, stride=1)

        self.enc_lin_0 = torch.nn.Linear(8, 2)

        self.dec_lin_0 = torch.nn.Linear(2, 8)

        self.dec_conv_0 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=6, stride=1)
        self.dec_conv_1 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2)
        self.dec_conv_2 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2)
        self.dec_conv_3 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2)
        self.dec_conv_4 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=6, stride=2)

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
        l = x

        x = self.dec_lin_0(x)
        x = torch.unsqueeze(x, dim=2)
        x = torch.unsqueeze(x, dim=3)

        x = self.dec_conv_0(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dec_conv_1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dec_conv_2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dec_conv_3(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dec_conv_4(x)
        x = torch.nn.functional.leaky_relu(x)

        # h_enc = self.encoder(state)
        # z = self._sample_latent(h_enc)
        return x, l


if __name__ == '__main__':
    square_dataset = SquareDataset(100)

    train_dataloader = DataLoader(square_dataset, batch_size=256, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    vae = Vae()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0001)

    n_epoch = 10
    for epoch in range(n_epoch):
        for img_batch in train_dataloader:
            img_batch = torch.unsqueeze(img_batch, dim=1).float()
            optimizer.zero_grad()
            y_hat_batch, latent_batch = vae(img_batch)
            a = 1

            # ll = latent_loss(vae.z_mean, vae.z_sigma)
            # loss = criterion(dec, inputs) + ll
            # loss.backward()
            # optimizer.step()
            # l = loss.data[0]

    # https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    # https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
