from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from thop import profile, clever_format
from LabNet.eval_utils import *


def train_verification_model(model, tau, dataloaders, loss_fct, optimizer, num_epochs=2, device='cpu', verbose=False):
  '''
  General purpose train function   
  '''
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  train_loss_history = []
  val_loss_history = []
  train_acc_history = []
  val_acc_history = []  

  for epoch in range(num_epochs):
    if verbose: print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    if verbose: print('-' * 10)

    # Run validation 1x per epoch
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  
      else:
        model.eval()   

      running_loss = 0.0
      running_corrects = 0

      iterations_per_epoch = len(dataloaders['train']) # num of batches
      num_iterations = num_epochs * iterations_per_epoch
      num_images_in_epoch = 0

      # Iterate over data (1 epoch)
      for t, (inputs, labels) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          loss = loss_fct(outputs, labels, tau, device)

          _, preds = torch.max(outputs, 1)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

          # statistics
          curr_batch_size = inputs.size(0)
          running_loss += loss.item() * curr_batch_size 
          # acc = torch.sum(preds == labels) 
          # acc = acc.cpu()
          # running_corrects += acc
          tgt_far = 0.3
          VAL, _, _ = val_solve(tgt_far, outputs, labels)         

          if phase=='train':
            train_loss_history.append(loss.item())
            train_acc_history.append(VAL)
          else:
            val_loss_history.append(loss.item())
            val_acc_history.append(VAL)

        # Maybe print training loss 
        if phase=='train' and verbose :
          print('(Iteration %d / %d) loss: %f acc: %f ' 
          % (t + 1 + epoch*iterations_per_epoch, num_iterations, train_loss_history[-1], train_acc_history[-1]))
        num_images_in_epoch += curr_batch_size

      epoch_loss = running_loss / num_images_in_epoch

      if verbose: print('{} Loss: {:.4f} '.format(phase, epoch_loss))

      # deep copy the model
      epoch_acc = train_acc_history[-1] # best we've got 
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print()

  time_elapsed = time.time() - since
  if verbose: print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  model.eval()
  return model, best_acc, train_loss_history, val_loss_history, train_acc_history



def train_model(model, dataloaders, loss_fct, optimizer, num_epochs=2, device='cpu', verbose=False):
  '''
  Classifier: general purpose train function 
  '''
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  train_loss_history = []
  train_acc_history = []
  val_loss_history = []
  val_acc_history = []

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Run validation 1x per epoch
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  
      else:
        model.eval()   

      running_loss = 0.0
      running_corrects = 0

      iterations_per_epoch = len(dataloaders['train']) # num of batches
      num_iterations = num_epochs * iterations_per_epoch
      num_images_in_epoch = 0

      # Iterate over data (1 epoch)
      for t, (inputs, labels) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          loss = loss_fct(outputs, labels)

          _, preds = torch.max(outputs, 1)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

          # statistics
          curr_batch_size = inputs.size(0)
          running_loss += loss.item() * curr_batch_size 
          acc = torch.sum(preds == labels) 
          acc = acc.cpu()
          running_corrects += acc

          if phase=='train':
            train_loss_history.append(loss.item())
            train_acc_history.append(acc / curr_batch_size)
          else:
            val_loss_history.append(loss.item())
            val_acc_history.append(acc / curr_batch_size)

        # Maybe print training loss 
        if phase=='train' and verbose :
          print('(Iteration %d / %d) loss: %f acc: %f' 
          % (t + 1 + epoch*iterations_per_epoch, num_iterations, train_loss_history[-1], train_acc_history[-1]))
        num_images_in_epoch += curr_batch_size

      epoch_loss = running_loss / num_images_in_epoch
      epoch_acc = running_corrects.double() / num_images_in_epoch
      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  model.eval()
  return model, best_acc, train_loss_history, val_loss_history, train_acc_history, val_acc_history