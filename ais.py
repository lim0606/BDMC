from __future__ import print_function

import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import grad as torchgrad
from . import hmc
from . import utils


def ais_trajectory(model,
                   loader,
                   forward=True,
                   schedule=np.linspace(0., 1., 500),
                   n_sample=100,
                   log_likelihood_fn='bernoulli-vae',
                   device=torch.device("cpu"),
                   _epsilon=0.01,
                   ):
  """Compute annealed importance sampling trajectories for a batch of data. 
  Could be used for *both* forward and reverse chain in BDMC.

  Args:
    model (vae.VAE): VAE model
    loader (iterator): iterator that returns pairs, with first component
      being `x`, second would be `z` or label (will not be used)
    forward (boolean): indicate forward/backward chain
    schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^t`
    n_sample (int): number of importance samples

  Returns:
      A list where each element is a torch.autograd.Variable that contains the 
      log importance weights for a single batch of data
  """

  if log_likelihood_fn == 'bernoulli-vae':
    def log_f_i(z, data, t, log_likelihood_fn=utils.log_bernoulli):
      """Unnormalized density for intermediate distribution `f_i`:
          f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
      =>  log f_i = log p(z) + t * log p(x|z)
      """
      zeros = torch.zeros(B, model.latent_dim).to(device)
      log_prior = utils.log_normal(z, zeros, zeros)
      _, x_logits = model.decode(z)
      #x_logits = x_logits.detach()
      log_likelihood = log_likelihood_fn(x_logits, data)
      return log_prior + log_likelihood.mul_(t)

  elif log_likelihood_fn == 'normal-vae':
    def log_f_i(z, data, t, log_likelihood_fn=utils.log_normal):
      zeros = torch.zeros(B, model.latent_dim).to(device)
      log_prior = utils.log_normal(z, zeros, zeros)
      _, x_mus, x_logvars = model.decode(z)
      #x_mus = x_mus.detach()
      #x_logvars = x_logvars.detach()
      log_likelihood = log_likelihood_fn(data, x_mus, x_logvars)
      return log_prior + log_likelihood.mul_(t)

  elif log_likelihood_fn == 'normal-fit':
    def log_f_i(z, data, t, log_likelihood_fn=utils.log_normal):
      zeros = torch.zeros(B, model.latent_dim).to(device)
      log_prior = utils.log_normal(z, zeros, zeros)
      _, x_mus, x_logvars = model.decode._forward_x(z)
      #x_mus = x_mus.detach()
      #x_logvars = x_logvars.detach()
      log_likelihood = log_likelihood_fn(data, x_mus, x_logvars)
      return log_prior + log_likelihood.mul_(t)

  else:
    raise NotImplementedError

  logws = []
  #for i, (batch, post_z) in enumerate(loader):
  for i, (batch, post_z) in tqdm(enumerate(loader), total=len(loader)):
    B = batch.size(0) * n_sample
    batch = batch.to(device)
    batch = utils.safe_repeat(batch, n_sample)

    with torch.no_grad():
      epsilon = torch.ones(B).to(device).mul_(_epsilon)
      accept_hist = torch.zeros(B).to(device)
      logw = torch.zeros(B).to(device)

    # initial sample of z
    if forward:
      current_z = torch.randn(B, model.latent_dim).to(device)
    else:
      current_z = utils.safe_repeat(post_z, n_sample).to(device)
    current_z = current_z.requires_grad_()

    #for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
    for j, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:]), 1):
      # update log importance weight
      log_int_1 = log_f_i(current_z, batch, t0)
      log_int_2 = log_f_i(current_z, batch, t1)
      logw += log_int_2 - log_int_1
      #print(log_int_2 - log_int_1)

      # resample velocity
      current_v = torch.randn(current_z.size()).to(device)

      def U(z):
        return -log_f_i(z, batch, t1)

      def grad_U(z):
        # grad w.r.t. outputs; mandatory in this case
        grad_outputs = torch.ones(B).to(device)
        # torch.autograd.grad default returns volatile
        grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
        # clip by norm
        max_ = B * model.latent_dim * 100.
        grad = torch.clamp(grad, -max_, max_)
        grad.requires_grad_()
        return grad

      def normalized_kinetic(v):
        zeros = torch.zeros(B, model.latent_dim).to(device)
        return -utils.log_normal(v, zeros, zeros)

      z, v = hmc.hmc_trajectory(current_z, current_v, U, grad_U, epsilon)
      current_z, epsilon, accept_hist = hmc.accept_reject(
          current_z, current_v,
          z, v,
          epsilon,
          accept_hist, j,
          U, K=normalized_kinetic,
          device=device,
          )
      #print(z, v)

    logw = utils.log_mean_exp(logw.view(n_sample, -1).transpose(0, 1)).detach()
    if not forward:
      logw = -logw
    logws.append(logw)
    #print('Last batch stats %.4f' % (logw.mean().cpu().numpy()))

  return logws
