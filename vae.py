# Data
# X: either [1, 0] or [0,1]
# y: normal sample from N(-1) or N(1)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import cv2
import torch
from torch import nn
import random
import tqdm

from pathlib import Path

import hydra
import numpy as np
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder, Snapshot

def data_generator():
    i = random.randint(0,1)
    x = torch.tensor(i).cuda()
    #x = torch.rand(5).cuda()
    if i == 0:
        #x = torch.distributions.Normal(torch.ones(5), .001).rsample().cuda()
        y = torch.distributions.Normal(5, 3).rsample().cuda()
    elif i == 1:
        #x = torch.distributions.Normal(-torch.ones(5), .001).rsample().cuda()
        y = torch.distributions.Normal(-5, 3).rsample().cuda()
    return x, y

def get_batch(n=32):
    xs = []
    ys = []
    for _ in range(n):
        x, y = data_generator()
        xs.append(x[None])
        ys.append(y[None][None])

    return torch.concat(xs), torch.concat(ys)


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1


class VAE(nn.Module):
    def __init__(
        self, in_dim=2, hidden_dim=8, latent_dim=3, out_dim=1, conditional=False
    ):
        super().__init__()
        #self.embedding = nn.Embedding(in_dim, hidden_dim)
        self.conditional = conditional
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + in_dim*self.conditional, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # p = torch.distributions.Uniform(-torch.ones_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def elbo(self, x, y, beta=1):
        #x_emb = self.embedding(x)
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        #kl = self.kl_divergence(z, mu, std)

        if self.conditional:
            y_hat = self.decoder(torch.concat([z, x], 1))
        else:
            y_hat = self.decoder(z)

        recon_loss = torch.square(y_hat - y)

        elbo = beta * kl.mean() + recon_loss.mean()

        return elbo

    def gen_sample(self, x):
        #x_emb = self.embedding(x)
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        std = torch.exp(log_var / 2)
        #p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(log_var))
        p = torch.distributions.Normal(mu, std)
        z = p.rsample()

        # sample z from q
        if self.conditional:
            y_hat = self.decoder(torch.concat([z, x], 1))
        else:
            y_hat = self.decoder(z)

        return y_hat


@hydra.main(config_path=".", config_name="vae")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    # create envs
    env = dmc.make(cfg.task, seed=cfg.seed, goal=(0.15, 0.15))

    data_specs = (
        env.observation_spec(),
        env.action_spec(),
        env.reward_spec(),
        env.discount_spec(),
    )

    # create data storage
    domain = get_domain(cfg.task)
    datasets_dir = work_dir / cfg.replay_buffer_dir
    replay_dir = datasets_dir.resolve() / domain / cfg.expl_agent / "buffer"
    print(f"replay dir: {replay_dir}")

    replay_loader = make_replay_loader(
        env,
        replay_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.discount,
        offset=100,
        vae=True,
    )
    replay_iter = iter(replay_loader)
    # next(replay_iter) will give obs, action, reward, discount, next_obs

    # create video recorders
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    timer = utils.Timer()
    import IPython as ipy; ipy.embed(colors="neutral")

    vae = VAE(in_dim=4, hidden_dim=128, latent_dim=32, out_dim=2, conditional=True)
    vae.cuda()

    optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)
    losses = []

    for _ in tqdm.tqdm(range(100)):
        batch = next(replay_iter)
        x, _, _, _, y = batch
        x = x.cuda()[:, :4]
        y = y.cuda()[:, :2]
        #x = torch.rand(x.shape).cuda() - .5
        #y = x
        #x, y = get_batch(128)

        optimizer.zero_grad()
        loss = vae.elbo(x, y, beta=1)
        # loss = vae.recon_loss(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss)

    pred = vae.gen_sample(x)
    #torch.square(pred - y).mean()

    sample_coords = [
        [-0.15, -0.15/2, .1, .1],
        [-0.15, -0.15/2, .2, .2],
        [-0.15, -0.15/2, .2, 0],
        [-0.15, -0.15/2, 0, 0.2],
        [-0.15, -0.15/2, -.2, 0],
        [-0.15, -0.15/2, 0, -0.2],
    ]
    snapper = Snapshot(work_dir if cfg.save_video else None)
    for j, coord in tqdm.tqdm(enumerate(sample_coords)):
        x = torch.tensor(coord).cuda().float()[None]
        for ind in range(3):
            goal = vae.gen_sample(x).cpu().detach().numpy()[0]
            #goal = np.array(coord) + goal
            env = dmc.make(cfg.task, seed=cfg.seed, goal=goal)
            env.reset()
            with env.physics.reset_context():
                env.physics.set_state(np.r_[coord[:2], np.zeros(2)])
            snapper.snapshot(env, f"j{j}_snapshot{ind}.jpg")



if __name__ == "__main__":
    main()
