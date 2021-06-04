from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from thop import profile, clever_format

models_dir = 'LabNet/models/'

def set_as_fc1_classifier(model, num_classes, p=0.5):
  ''' Sets the last FC layer and freezes all earlier layers.
    
    Inputs:
    - model: VGG-16
    - num_classes: number of outputs
  '''
  for param in model.parameters():
    param.requires_grad = False

  model.classifier[5] = nn.Dropout(p=p, inplace=False)
  num_ftrs = model.classifier[6].in_features
  model.classifier[6] = nn.Linear(num_ftrs, num_classes) # requires gradients automatically
  model.eval() 
  return None


def set_as_fc2_classifier(model, h_size, num_classes, p=0.5):
  ''' Sets the 2-last FC layer and freezes all earlier layers.
    
    Inputs:
    - model: VGG-16
    - h_size: hidden FC layer size
    - num_classes: number of outputs
    - p: dropout probability
  '''
  # freeze all layers
  for param in model.parameters():
    param.requires_grad = False

  # as we learn the next layer from scratch, best to set dropout prob
  model.classifier[2] = nn.Dropout(p=p, inplace=False)

  # unfreeze penultimate FC layer and maybe change output size
  if h_size != 4096:
    num_ftrs = model.classifier[3].in_features #4096
    model.classifier[3] = nn.Linear(num_ftrs, h_size)
  else:
    for param in model.classifier[3].parameters():
      param.requires_grad = True

  # set dropout probabilities
  model.classifier[5] = nn.Dropout(p=p, inplace=False)

  # change final output size to num_classes, this unfreezes grads
  model.classifier[6] = nn.Linear(h_size, num_classes)
  model.eval()
  return None


def set_as_fc2_feature_model(model, h_size, p=0.5):
  ''' Removes top layers, only retaining up to the classifier[3] layer
      This is utilized as a feature extractor of size h_size
    
    Inputs:
    - model: VGG-16
    - h_size: hidden FC layer size
    - num_classes: number of outputs
    - p: dropout probability

    Outputs: None. The model is modified inplace
  '''
  dummy_num_classes = 2

  # in VGG-16: reset classifier[3] output to h_size, and set dropout
  set_as_fc2_classifier(model, h_size, h_size, p) 

  # model.classifier[7] = nn.Linear(h_size, h_size)
  # discard subsequent layers
  # remove_class_layer(model)
  model.eval()
  return None


def remove_class_layer(model):
  '''
  Resulting model to be used as a feature extractor. Assumes model is built from VGG-16
  Inputs:
  - model: a model

  Returns:
  - model: the feature extractor model
  '''
  # retain layers up to [3], so as to have h_size feature representation
  model.classifier = nn.Sequential(*list(model.classifier.children())[:4])
  return model


def get_params_to_update(model, verbose=True):
  params_to_update = model.parameters()
  params_to_update = []
  if verbose : print("Params to learn:")
  for name, param in model.named_parameters():
    if param.requires_grad == True:
      params_to_update.append(param)
      if verbose: print("\t", name)
  print()    
  return params_to_update


def get_num_trainable_params(model):
  '''
  Inputs:
  - model
  Return:
  - the number of trainable parameters in the model
  '''
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  num_params = sum([np.prod(p.size()) for p in model_parameters])
  return num_params


def profile_model(model, verbose=False, device='cpu'):
  flops, params = profile(model, inputs=(torch.randn(1, 3, 244, 244).to(device),), verbose=verbose)
  flops, params = clever_format([flops, params])  
  return flops, params


def load_model(name, device='cpu'):
  path = models_dir + name 
  model = torch.load(path, map_location=torch.device(device))
  model.eval()
  model.requires_grad_(False)
  return model
