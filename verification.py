import torch
import numpy as np
from collections import Counter

def cosine_similarity(F):
    """
    Compute a N x N matrix of normalized dot products between all pairs 
    of examples in a batch.

    Inputs:
    - F: N x D tensor; each row is the feature representation of an example in a batch of N examples
    
    Returns:
    - cos_dist_matrix: N x N tensor; each element i, j in the matrix is 
      the cosine distance between F[i] and F[j].
    """
    cos_dist_matrix = None

    F_norm = F / torch.linalg.norm(F, dim=1, keepdim=True)
    cos_dist_matrix = torch.matmul(F_norm, F_norm.T)
    return cos_dist_matrix


def verif_loss(X, y, tau, device='cpu'):
  ''' 
  Contrastive loss for verification. 
  Inputs:
  - X: N x d tensor; batch of N feature vectors of length d
  - y: N vector; class labels
  - tau: temperature

  Returns:
  - verification loss over the batch
  '''
  N, d = X.shape

  # check that every label appears at least 2x
  c = Counter(y) 
  n = 1
  elt = c.most_common()[:-n-1:-1]

  # similarity matrix
  sim_matrix = cosine_similarity(X)

  # exponential with temperature
  exponential = torch.exp(sim_matrix / tau)
    
  # discard diagonal
  diag_mask = (torch.ones_like(exponential, device=device) - torch.eye(N, device=device)).to(device).bool()
  exponential = exponential.masked_select(diag_mask).view(N, N-1)

  # Y: each row is y.T, used for further masking operations
  Y = torch.cat([y.T,]*N).view(N,-1)
  # print('Y')
  # print(Y)
  # print()

  # negative-mask: keep all examples such that label != current label y_i
  neg_mask = (Y != Y.T).type(torch.int) # y is a col-vector to be broacasted
  neg_mask = neg_mask.masked_select(diag_mask).view(N, N-1)
  neg_exp = exponential * neg_mask
  neg_den = torch.sum(neg_exp, dim=1, keepdim=False) # [N, 1]
  # print('neg_den')
  # print(neg_den)
  # print()

  # positive-mask: keep the first instance where label=current label y_i
  pos_mask = (Y == Y.T).type(torch.int) # all those where label = y_i
  pos_mask = (Y == Y.T).type(torch.int) # all those where label = y_i
  pos_mask = pos_mask.masked_select(diag_mask).view(N, N-1)
  # print('pos_mask')
  # print(pos_mask)
  # print()

  # if no positive pair for i, then row pos_mask[i] is all zeros and shouldn't be counted
  no_pos_pair_mask = torch.sum(pos_mask, dim=1)
  no_pos_pair_mask = (no_pos_pair_mask > 0.0).type(torch.int)
  # print('no pos pair')
  # print(no_pos_pair_mask)
  # print()

  pos_indices = pos_mask.argmax(axis=1) # index of fist instance
  norm = no_pos_pair_mask / torch.sum(no_pos_pair_mask)
  # print('norm')
  # print(norm)
  # print()

  num = exponential.gather(1, pos_indices.view(-1, 1)).squeeze()
  # print('num')
  # print(num)
  # print()

  # softmax denominator also contains the positive example
  den = neg_den + num
  p = num/den
    
  # per image loss
  loss_i = -torch.log(p) * norm

  # batch loss
  loss = torch.sum(loss_i)

  return loss
  
