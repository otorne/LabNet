from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, TensorDataset, random_split
import random
import torch
import os
from collections import Counter

# globs 
data_dir = 'DataSet/Train'
ImageNet_mean = [0.485, 0.456, 0.406]
ImageNet_std = [0.229, 0.224, 0.225]

def train_transform(seed=None):
  '''
  Return a composition of standard data augmentations for training.
  '''
  if seed:
    random.seed(seed)
    torch.random.manual_seed(seed)
  
  train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2))
      ,transforms.RandomHorizontalFlip(p=0.5)
      ,transforms.ToTensor() # in [0, 1]
      ,transforms.Normalize(ImageNet_mean, ImageNet_std)# using ImageNet stats
      ,transforms.RandomErasing(p=0.2, scale=(0.0, 0.25), ratio=(0.3, 3.3))
      ,transforms.RandomRotation(degrees=10)      
      # ,transforms.Grayscale(num_output_channels=3) # grayscale is for visual debugging only !!!
      ]) 
  return train_transform


def test_transform():
  """
    Normalize as per training.
  """
  test_transform = transforms.Compose([
      transforms.Resize((224,224))
      ,transforms.ToTensor()
      ,transforms.Normalize(ImageNet_mean, ImageNet_std)
      # ,transforms.Grayscale(num_output_channels=3) # grayscale is for visual debugging only !!!
      ])
  return test_transform


def do_train_transform(img, size=224):
    transform = train_transform()
    return transform(img)


def preprocess(img, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ImageNet_mean, std=ImageNet_std),
        transforms.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=1.0 / ImageNet_std),
        T.Normalize(mean=-ImageNet_mean, std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)


class DatasetFromSubset(Dataset):
  """
   Required so we can use individual transforms on Subsets
  """
  def __init__(self, subset, transform=None):
      self.subset = subset
      self.transform = transform

  def __getitem__(self, index):
      x, y = self.subset[index]
      if self.transform:
          x = self.transform(x)
      return x, y

  def __len__(self):
      return len(self.subset)


def create_datasets(train_split = 1.0, data_dir='DataSet/Train'):
  """
    Make datasets from Image folder
  """
  assert (train_split <= 1.0), "wrong split sizes"
  seed=None
  
  full_dataset = datasets.ImageFolder(data_dir)
  N = len(full_dataset)
  N_train = int(N * train_split)
  N_val = N - N_train

  if seed:
    generator=torch.Generator().manual_seed(seed)
  else:
    generator=torch.Generator()

  train_subset, val_subset = torch.utils.data.random_split(full_dataset, [N_train, N_val], generator)

  # required to set custom transform per subset  
  #train_set = DatasetFromSubset(train_subset, train_transform())
  train_set = DatasetFromSubset(train_subset, test_transform())
  val_set = DatasetFromSubset(val_subset, test_transform())

  ds = {}
  ds['full'] = full_dataset
  ds['train'] = train_set
  ds['val'] = val_set
  return ds


def get_sampler_weights(data, verbose=False):
  """
  For sampling from unbalanced datasets.
  (ugly piece of code)
  """                 
  _, class_to_idx, _ = find_classes()   
  nclasses = len(class_to_idx)
  tmp_loader = torch.utils.data.DataLoader(data, batch_size=1)

  count = [0] * nclasses                                                      
  for img, label in tmp_loader: 
    count[label.item()] += 1 

  weight_per_class = [0.] * nclasses                                      
  N = float(sum(count))                                                   
  for i in range(nclasses):                                                   
     weight_per_class[i] = N/float(count[i])

  weight = [0] * len(data)                                              
  for idx, val in enumerate(data):                                          
     weight[idx] = weight_per_class[val[1]]                                  
    
  return torch.DoubleTensor(weight)


def find_classes(dir=data_dir):
  classes = os.listdir(dir)
  classes.sort()
  class_to_idx = {classes[i]: i for i in range(len(classes))}
  idx_to_class = {i: classes[i] for i in range(len(classes))}
  return classes, class_to_idx, idx_to_class