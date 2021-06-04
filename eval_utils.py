import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from LabNet.verification import *
import seaborn as sns
import matplotlib.pylab as plt
from LabNet.data import *

def val_far(d, dist, true_matrix):
  ''' 
  VAL and FAR metrics given
  Inputs:
  - d: threshold
  - dist: distance/similarity matrix
  - true_matrix: actual same/diff identity matrix
  '''
  # N = true_matrix.shape[0]
  # dist = np.random.rand(N, N)
  # true_matrix = np.random.rand(N, N)

  N = true_matrix.shape[0]
  mask = ~np.eye(N,dtype=bool)
  true_matrix = true_matrix[mask].reshape(N, -1)
 
  num_same = np.sum(true_matrix)
  num_diff = N*(N-1) - num_same


  pred = (dist > d).astype(int)
  pred = pred[mask].reshape(N, -1)
  assert pred.shape == true_matrix.shape, 'oops'



  true_accepts = np.sum(true_matrix * pred)
  val = true_accepts / num_same
    
  false_accepts = np.sum((1-true_matrix) * pred)
  far = false_accepts / num_diff
  return val, far, pred


def val_solve(tgt_far, F, classes):
  '''
  Given a target FAR, return corresponding VAL
  '''
  N = len(classes)
  
  # similarity matrix given features F
  # dist = cosine_similarity(F).cpu().numpy() 
  dist = cosine_similarity(F).detach().cpu().numpy() 

  # true identity matching matrix given classes
  true_matrix = (classes == classes.repeat(N).view(N,N).T).cpu().numpy().astype(int)

  val, far = np.zeros(9), np.zeros(9)
  for i, d in enumerate(np.linspace(0.9, 0.1, 9)):
    val[i], far[i], _ = val_far(d, dist, true_matrix)
  
  val_star = np.interp(tgt_far, far, val)
  return val_star, val, far


def display_similarity_heatmap(model, imgs):
  '''
  Display a similarity heatmap for verification task
  Inputs:
  - model: feature-extractor type model
  - imgs: some images
  '''
  with torch.no_grad():
    F = model(imgs)
  print('F.shape: {}'.format(F.shape))

  dist = cosine_similarity(F).cpu().numpy()
  print('dist.shape: {}'.format(dist.shape))

  

  ax = sns.heatmap(dist, linewidth=0.05)
  # plt.axis('off')
  plt.show()


def display_matrix_heatmap(mat):
  ''' 
  Heatmap of a matrix
  '''
  ax = sns.heatmap(mat, linewidth=0.5)
  plt.show()


def verifier_saliency_maps(X, X_tgt, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X, X_tgt: Input images; Tensor of shape (1, 3, H, W)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (, H, W) giving the saliency maps for the input
    images.
    """
    
    # prepare model/input
    model.eval()
    X.requires_grad_()

    # features
    F1 = model(X)
    F2 = model(X_tgt)
    F1 = F1.view(-1)
    F2 = F2.view(-1)
    
    F1_norm = F1 / torch.linalg.norm(F1, dim=0, keepdim=True)
    F2_norm = F2 / torch.linalg.norm(F2, dim=0, keepdim=True)
    cos_dist_loss = torch.dot(F1_norm, F2_norm)

    cos_dist_loss.backward()

    saliency = torch.max(torch.abs(X.grad), dim=1).values
    return saliency


def show_verifier_saliency_maps(X1, X2, y, model, idx_to_class):
    # Compute saliency maps for images in X
    saliency = verifier_saliency_maps(X1, X2, model)

    # Convert to numpy array and show images and saliency maps together.
    saliency = saliency.cpu().numpy()
    X1 = X1.detach().cpu().numpy()
    X2 = X2.cpu().numpy()
    y = y.cpu().numpy()

        #
    plt.subplot(1, 3, 1)
    plt.imshow(array_to_img(X2[0]))
    plt.axis('off')        
    plt.title('Target')
    #
    plt.subplot(1, 3, 2)
    plt.imshow(array_to_img(X1[0]))
    plt.axis('off')
    plt.title(idx_to_class[y[0]])
    #
    plt.subplot(1, 3, 3)
    plt.imshow(saliency[0], cmap=plt.cm.hot)
    plt.axis('off')
    plt.title('Mochi-saliency')
    plt.gcf().set_size_inches(12, 5)
    plt.show()


def classifier_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    scores = model(X)
    correct_class_scores = scores.gather(1, y.view(-1,1)).squeeze()
    dummy_loss = torch.sum(correct_class_scores)
    dummy_loss.backward()

    saliency = torch.max(torch.abs(X.grad), dim=1).values
    return saliency


def show_classifier_saliency_maps(X, y, model, idx_to_class):
    # Compute saliency maps for images in X
    saliency = classifier_saliency_maps(X, y, model)

    # Convert to numpy array and show images and saliency maps together.
    saliency = saliency.cpu().numpy()
    X = X.detach().cpu().numpy()
    y = y.cpu().numpy()

    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(array_to_img(X[i]))
        plt.axis('off')
        plt.title(idx_to_class[y[i]] + ': ' + str(i))
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(8, 5)
    plt.show()


def array_to_img(inp):
    inp = inp.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1) 
    return inp


def show_images(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)