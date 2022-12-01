#!/usr/bin/env python3

import argparse
import torch
import torch.distributions as dist
import torch.nn as nn


def get_joint_dist(x, y, p):
    mean = torch.tensor([x.mean, y.mean])

    x_var = x.stddev ** 2
    y_var = y.stddev ** 2
    cross_cov = p * x.stddev * y.stddev
    covariance = torch.tensor([[x_var, cross_cov], [cross_cov, y_var]])

    return dist.MultivariateNormal(mean, covariance)

def construct_true_posterior(cfg):
    xdist = dist.Normal(cfg.mu_x, cfg.std_x)
    ydist = dist.Normal(cfg.mu_y, cfg.std_y)
    joint = get_joint_dist(xdist, ydist, cfg.p)
    return lambda z, num_samples: sample_xy_conditioned_on(joint, z, cfg.coef_x, cfg.coef_y, num_samples)

def get_data_dist(joint, coef_x, coef_y):
    coefs = torch.tensor([coef_x, coef_y])
    mean = torch.matmul(coefs, joint.mean)
    var = torch.matmul(torch.matmul(coefs, joint.covariance_matrix), coefs.t())

    return dist.Normal(mean, torch.sqrt(var))

def rotation_matrix(theta):
    s = torch.sin(theta)
    c = torch.cos(theta)
    return torch.tensor([[c, -s], [s, c]])

def rotate(x, theta):
    rot = rotation_matrix(theta)
    return torch.matmul(rot, x)

def rotate_dist(d, theta):
    # rotate distribution by multiply the random variable by rotation matrix
    rot = rotation_matrix(theta)
    mean = torch.matmul(rot, d.mean)
    cov = torch.matmul(torch.matmul(rot, d.covariance_matrix), rot.t())
    return dist.MultivariateNormal(mean, cov)

def condition_dist(d, value):
    # condition using Mark Schmidt's conditional equations
    dmu = d.mean
    dcov = d.covariance_matrix
    mu = dmu[0] + dcov[0, 1] / dcov[1, 1] * (value - dmu[1])
    cov = dcov[0, 0] - dcov[0, 1] * dcov[1, 0] / dcov[1, 1]
    return dist.Normal(mu, torch.sqrt(cov))

def score_sample_under_true_posterior(x, y, value, coef_x, coef_y, horizontal_dist):
    assert (torch.abs(coef_x*x + coef_y*y - value) < 1e-5).all()

    # angle that x+y=value makes with the horizontal
    theta = torch.atan2(torch.tensor(coef_x), torch.tensor(coef_y))

    # rotate x, y onto the horizontal
    xy = torch.tensor([x, y])
    p = rotate(xy, theta)

    # score x coordinate of p
    px = p[0]
    return horizontal_dist.log_prob(px)

def sample_xy_conditioned_on(joint_dist, value, coef_x, coef_y, num_samples):
    # angle that x+y=value makes with the horizontal
    theta = torch.atan2(torch.tensor(coef_x), torch.tensor(coef_y))

    # rotate joint_dist so that the line coef_x*x + coef_y*y = value is parallel to the horizontal axis
    rot_dist = rotate_dist(joint_dist, theta)

    # condition the rotated distribution on y=value
    horizontal_dist = condition_dist(rot_dist, value)

    # find closest point to origin so that we can rotate it onto the vertical axis
    A = torch.tensor([[coef_x, coef_y], [coef_y, -coef_x]])
    closest_point_to_origin = torch.matmul(A.inverse(), torch.tensor([value, 0.]))
    yval = rotate(closest_point_to_origin, theta)  # y coordinate
    assert (yval[0] < 1e-5).all()

    # sample point
    h = horizontal_dist.rsample([num_samples]).squeeze()  # x coordinate
    p = torch.stack([h, yval[1].repeat(num_samples)])

    # rotate that point back onto the line x+y=value
    x, y = rotate(p, -theta)

    assert (torch.abs(coef_x*x + coef_y*y - value) < 1e-5).all()

    return x, y, horizontal_dist.log_prob(h)


class InferenceCompilation:
    def __init__(self, cfg):
        self.cfg = cfg

        self.latent_transform = torch.nn.Sequential(
            torch.nn.Linear(1, cfg.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.hidden_size, cfg.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.hidden_size, 2),
        )

        # generate true posterior p(x,y|z) and data distribution p(z)
        xdist = dist.Normal(cfg.mu_x, cfg.std_x)
        ydist = dist.Normal(cfg.mu_y, cfg.std_y)
        joint = get_joint_dist(xdist, ydist, cfg.p)

        self.data_dist = get_data_dist(joint, cfg.coef_x, cfg.coef_y)
        self.true_posterior = construct_true_posterior(cfg)
        self.horizontal_dist = self._construct_horizontal_dist(cfg, joint)

        self._initialize_optimizer()

    def _construct_horizontal_dist(self, cfg, joint):
        # angle that x+y=value makes with the horizontal
        theta = torch.atan2(torch.tensor(cfg.coef_x), torch.tensor(cfg.coef_y))

        # rotate joint_dist so that the line coef_x*x + coef_y*y = value is parallel to the horizontal axis
        rot_dist = rotate_dist(joint, theta)

        # condition the rotated distribution on y=value
        return lambda value: condition_dist(rot_dist, value)

    def _initialize_optimizer(self):
        self.optimizer = torch.optim.Adam(self.latent_transform.parameters(), lr=self.cfg.lr)

    def _clip_gradients(self):
        nn.utils.clip_grad_norm_(self.latent_transform.parameters(), 1)

    def _project_to_simplex(self, xys):
        # https://home.ttic.edu/~wwang5/papers/SimplexProj.pdf
        N, D = xys.shape
        sorted_xys, _ = torch.sort(xys)
        tmp = torch.matmul((sorted_xys.cumsum(dim=1)-1), torch.diag(1./torch.arange(1, D+1)))

        v = xys - torch.index_select(tmp, 0, torch.sum(sorted_xys > tmp, dim=1))
        return torch.relu(v)

    def infer(self):
        for _ in range(cfg.num_data_samples):
            z = self.data_dist.rsample([1])
            params = self.latent_transform(z)
            # ddist = dist.Dirichlet(params)
            ddist = dist.Normal(params[0], torch.exp(params[1]))

            x, y, true_scores = self.true_posterior(z, cfg.num_posterior_samples)

            # xy_probs = self._project_to_simplex(torch.stack(xys))
            proposal_scores = ddist.log_prob(x)
            kl = (true_scores - proposal_scores).sum()

            self.optimizer.zero_grad()
            kl.backward()
            self._clip_gradients()
            self.optimizer.step()
        return self

    def proposal(self):
        def _proposal(z):
            params = self.latent_transform(z)
            ddist = dist.Normal(params[0], torch.exp(params[1]))
            return Proposal(ddist, z, self.cfg.coef_x, self.cfg.coef_y)
        return _proposal


class Proposal:
    def __init__(self, ddist, z, coef_x, coef_y):
        self.ddist = ddist
        self.z = z
        self.coef_x = coef_x
        self.coef_y = coef_y

    def sample(self, n=1):
        x = self.ddist.rsample([n])
        y = (self.z - self.coef_x * x) / self.coef_y
        score = self.ddist.log_prob(x)
        return (x, y), score


def get_args():
    parser = argparse.ArgumentParser(description='Parser')

    # Analytical Parameters
    parser.add_argument('--mu_x', type=float, default=0., help='mean of x')
    parser.add_argument('--std_x', type=float, default=1., help='standard devation of x')
    parser.add_argument('--mu_y', type=float, default=0., help='mean of y')
    parser.add_argument('--std_y', type=float, default=1., help='standard devation of y')
    parser.add_argument('--p', type=float, default=0., help='correlation between x and y')
    parser.add_argument('--coef_x', type=float, default=3., help='coefficient on x')
    parser.add_argument('--coef_y', type=float, default=2., help='coefficient on y')
    parser.add_argument('--b', type=float, default=7., help='what coef_x * x + coef_y * y is equal to')

    # Inference Compilation Parameters
    parser.add_argument('--num_data_samples', type=int, default=100, help='number of observations')
    parser.add_argument('--num_posterior_samples', type=int, default=100, help='number of posterior samples per observation')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of neural network for proposal')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Adam optimizer')

    # parse
    args, _ = parser.parse_known_args()

    # return it all
    return args, parser

def trial_sample(cfg):
    true_posterior = construct_true_posterior(cfg)
    x, y, score = true_posterior(torch.tensor(cfg.b), 1)
    print("{} * x + {} * y = {} * {} + {} * {} = {}".format(cfg.coef_x, cfg.coef_y, cfg.coef_x, x, cfg.coef_y, y, cfg.coef_x*x + cfg.coef_y*y))
    print("score of sample is log p(x={}, y={} | z={}) = {}".format(x, y, cfg.b, score))


if __name__ == "__main__":
    cfg, _ = get_args()

    ic = InferenceCompilation(cfg)
    horizontal_dist = ic.horizontal_dist
    ic.infer()
    with torch.no_grad():
        for _ in range(10):
            proposal = ic.proposal()
            z = ic.data_dist.sample([1])
            print('sampled z: ', z)
            ddist = proposal(z)
            (x, y), score = ddist.sample()
            estimated_z = cfg.coef_x*x + cfg.coef_y*y
            print("{} * x + {} * y = {} * {} + {} * {} = {}".format(cfg.coef_x, cfg.coef_y, cfg.coef_x, x, cfg.coef_y, y, estimated_z))
            print("score of sample under proposal is log p(x={}, y={} | z={}) = {}".format(x, y, estimated_z, score))

            true_score = score_sample_under_true_posterior(x, y, z, cfg.coef_x, cfg.coef_y, horizontal_dist(z))
            print("score of sample under true posterior is {}".format(true_score))
