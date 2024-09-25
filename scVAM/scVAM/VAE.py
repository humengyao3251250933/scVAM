import load_data as loader
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import scipy.io

from scEMC import scEMC, parse_arguments
from layers import ZINBLoss
from utils import cluster_acc

# 定义VAE模型类
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, z_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD


# 加载.mat文件数据
def load_mat_data(path):
    with h5py.File(path, 'r') as f:
        X = f['X']
        X1 = np.array(f[X[0, 0]])  # 解析X的第一个对象
        X2 = np.array(f[X[0, 1]])  # 解析X的第二个对象
        Y = np.array(f['Y'])
    return X1, X2, Y


# 训练VAE模型
def train_vae_model(X, input_dim, hidden_dim, z_dim, lr=0.0005, num_epochs=200, batch_size=8, device='cuda'):
    vae = VAE(input_dim, hidden_dim, z_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    data_loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=True)

    vae.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch in data_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss = vae.loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'VAE Epoch {epoch + 1}, Loss: {train_loss / len(data_loader.dataset)}')

    return vae


# 生成新样本
def generate_samples(vae, num_cells, input_dim, z_dim, device='cuda'):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num_cells, z_dim).to(device)
        generated = vae.decode(z).cpu().numpy()
    return generated


def main():
    # 加载数据
    X1, X2, Y = load_mat_data('D:/code/scEMC-master/scEMC/datasets/BMNC.mat')
    Y = Y.reshape(-1, 1)

    # 打印原始数据的形状
    # print(f"Original X1 shape: {X1.shape}")  # 应该是 (30672, 1000)
    # print(f"Original X2 shape: {X2.shape}")  # 应该是 (30672, 25)
    X1 = X1.T
    X2 = X2.T
    # 归一化数据
    scaler_X1 = MinMaxScaler()
    scaler_X2 = MinMaxScaler()
    X1_scaled = scaler_X1.fit_transform(X1)
    X2_scaled = scaler_X2.fit_transform(X2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 训练VAE模型
    hidden_dim = 256
    z_dim = 10
    vae_scRNA = train_vae_model(X1_scaled, X1_scaled.shape[1], hidden_dim, z_dim, batch_size=128, device=device)
    vae_scATAC = train_vae_model(X2_scaled, X2_scaled.shape[1], hidden_dim, z_dim, batch_size=128, device=device)

    # 生成新样本
    num_cells = X1_scaled.shape[0]  # 细胞数量（行数）应该是 30672
    generated_scRNA = generate_samples(vae_scRNA, num_cells, X1_scaled.shape[1], z_dim, device=device)
    generated_scATAC = generate_samples(vae_scATAC, num_cells, X2_scaled.shape[1], z_dim, device=device)
    # 反归一化生成的数据
    generated_scRNA = scaler_X1.inverse_transform(generated_scRNA)
    generated_scATAC = scaler_X2.inverse_transform(generated_scATAC)

    # 保存生成的样本到.mat文件
    generated_data = {'X1': generated_scRNA, 'X2': generated_scATAC, 'Y': Y}
    scipy.io.savemat('D:/code/scEMC/scEMC/datasets/BMNC_generated.mat', generated_data)

if __name__ == "__main__":
    main()
