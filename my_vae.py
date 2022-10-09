import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import os

from torchvision.utils import save_image

from utils import get_progress_bar, update_progress_bar, ApplyTransform


# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 256, 256)
    return x

# 1. Load and normalizing dataset
# 1. Data augmentation
transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transforms_test = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

total_dataset = datasets.ImageFolder('D:/Projects/vae/dataset/UCMerced_LandUse/UCMerced_LandUse/Images', transform=None)
dataset = ApplyTransform(total_dataset, transform=transforms_train)
train_dataset_loader = DataLoader(dataset=dataset, batch_size=16)


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.enc_mu = torch.nn.Linear(H, latent_size)
        self.enc_log_sigma = torch.nn.Linear(H, latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        return torch.distributions.Normal(loc=mu, scale=sigma)


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = torch.tanh(self.linear2(x))
        return torch.distributions.Normal(mu, torch.ones_like(mu))


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, state):
        q_z = self.encoder(state)
        z = q_z.rsample()
        return self.decoder(z), q_z


transform = transforms.Compose(
    [transforms.ToTensor(),
     # Normalize the images to be -0.5, 0.5
     transforms.Normalize(0.5, 1)]
)

input_dim = 32 * 32
batch_size = 128
num_epochs = 10
learning_rate = 0.001
hidden_size = 512
latent_size = 8

print('Number of samples: ', len(dataset))

encoder = Encoder(input_dim, hidden_size, latent_size)
decoder = Decoder(latent_size, hidden_size, input_dim)

vae = VAE(encoder, decoder).to(device)

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for data in train_dataset_loader:
        inputs, _ = data
        inputs = inputs.view(-1, input_dim).to(device)
        optimizer.zero_grad()
        p_x, q_z = vae(inputs)
        log_likelihood = p_x.log_prob(inputs).sum(-1).mean()
        kl = torch.distributions.kl_divergence(
            q_z,
            torch.distributions.Normal(0, 1.)
        ).sum(-1).mean()
        loss = -(log_likelihood - kl)
        loss.backward()
        optimizer.step()
        l = loss.item()
    print(epoch, l, log_likelihood.item(), kl.item())

    if epoch % 10 == 0:
        save = to_img(q_z.cpu().data)
        save_image(save, './vae_img/image_{}.png'.format(epoch))


torch.save(vae.state_dict(), 'D:/Projects/vae/checkpoint/vae.pth')