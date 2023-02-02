from torch import nn

from normal import *
from networks import *

torch.set_default_dtype(torch.float64)

class E2C(nn.Module):
    def __init__(self, obs_dim, z_dim, u_dim, env='planar'):
        super(E2C, self).__init__()
        enc, dec, trans = load_config(env)  # initialize the corresponding class encoder, decoder and transition

        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.u_dim = u_dim

        self.encoder = enc(obs_dim=obs_dim, z_dim=z_dim)
        # self.encoder.apply(init_weights)
        self.decoder = dec(z_dim=z_dim, obs_dim=obs_dim)
        # self.decoder.apply(init_weights)
        self.trans = trans(z_dim=z_dim, u_dim=u_dim)
        # self.trans.apply(init_weights)

        self.dynamics = self.trans.dynamics
        self.net = self.trans.net  # network to output the last layer before predicting A_t, B_t and o_t
        self.fc_A = self.trans.fc_A
        self.fc_B = self.trans.fc_B
        self.fc_o = self.trans.fc_o

    def encode(self, x):
        """
        :param x:
        :return: mean and log variance of q(z | x)
        """
        return self.encoder(x)

    def decode(self, z):
        """
        :param z:
        :return: bernoulli distribution p(x | z)
        """
        return self.decoder(z)

    def transition(self, z_bar, q_z, u):
        """
        :param z_bar:
        :param q_z:
        :param u:
        :return: samples z_hat_next and Q(z_hat_next)
        """
        return self.trans(z_bar, q_z, u)

    def reparam(self, mean, logvar):
        sigma = (logvar / 2).exp()
        epsilon = torch.randn_like(sigma)
        return mean + torch.mul(epsilon, sigma)

    def forward(self, x, u, x_next):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)

        x_recon = self.decode(z)

        z_next, q_z_next_pred = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)

        mu_next, logvar_next = self.encode(x_next)
        q_z_next = NormalDistribution(mean=mu_next, logvar=logvar_next)

        return x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next

    def predict(self, x, u):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)

        z_next, q_z_next_pred = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)
        return x_next_pred
    
    def state_trans(self, z_bar_t, u_t):
        # to calculate the state transition: z_next = A*z_bar_t + B*u_t + ot
        h_t = self.net(z_bar_t) 
        B_t = self.fc_B(h_t)
        o_t = self.fc_o(h_t)

        v_t, r_t = self.fc_A(h_t).chunk(2, dim=1)
        v_t = torch.unsqueeze(v_t, dim=-1)
        r_t = torch.unsqueeze(r_t, dim=-2)

        A_t = torch.eye(self.z_dim).repeat(z_bar_t.size(0), 1, 1) + torch.bmm(v_t, r_t)
        B_t = B_t.view(-1, self.z_dim, self.u_dim)

        z_next = A_t.bmm(z_bar_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + o_t

        return z_next
