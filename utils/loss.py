import numpy as np
import random
import torch
import torch.nn.functional as F

def cross_entropy_loss(logit, target):

    return F.binary_cross_entropy_with_logits(logit, target, reduction='mean') / logit.size(0)

def l2_loss(pred_traj, pred_traj_gt):

    seq_len, batch, _ = pred_traj.size()
    loss = (pred_traj_gt - pred_traj)**2

    return torch.sum(loss) / (seq_len * batch)

def kld_loss_normal(mean, log_var):

    '''
    KLD = -0.5 * (log(var1) - (var1 + mu1^2) + 1 )
    '''

    kld = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    return torch.sum(kld, dim=1).mean()

def kld_loss(mean1, log_var1, mean2, log_var2):

    '''
    KLD = -0.5 * (log(var1/var2) - (var1 + (mu1 - mu2)^2)/var2 + 1 )
        = -0.5 * (A - B + 1)

    A = log(var1) - log(var2)
    B = (var1 + (mu1 - mu2)^2) / var2

    prior ~ N(mean2, var2)
    posterior ~ N(mean1, var1)
    '''

    A = log_var1 - log_var2
    B = log_var1.exp() + (mean1 - mean2).pow(2)
    kld = -0.5 * (A - B.div(log_var2.exp() + 1e-10) + 1)

    return torch.sum(kld, dim=1).mean()

def calc_ED_error(o, k, best_k, pred_trajs, future_traj, ADE_k, FDE_k):

    if (k == 1):
        error_ADE = np.sqrt(np.sum((pred_trajs[0, :, o, :2] - future_traj[:, o, :2]) ** 2, axis=1))
        error_FDE = np.sqrt(np.sum((pred_trajs[0, :, o, :2] - future_traj[:, o, :2]) ** 2, axis=1))
        return error_ADE, error_FDE[-1]

    elif (k <= best_k):
        minADE_idx = np.argmin(np.array(ADE_k[:k]))
        minFDE_idx = np.argmin(np.array(FDE_k[:k]))
        error_ADE = np.sqrt(np.sum((pred_trajs[minADE_idx, :, o, :2] - future_traj[:, o, :2]) ** 2, axis=1))
        error_FDE = np.sqrt(np.sum((pred_trajs[minFDE_idx, :, o, :2] - future_traj[:, o, :2]) ** 2, axis=1))
        return error_ADE, error_FDE[-1]

    else:
        return 0, 0

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """

    y_fake = torch.ones_like(scores_fake)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """

    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.0)
    y_fake = torch.zeros_like(scores_fake)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    loss = loss_real + loss_fake

    return loss
