import pickle
import torch
import numpy as np

from PIL import Image

from torch.utils.data import Dataset

class DatasetFromBatchPath(Dataset):
    def __init__(self, split_path, cifar_path, transform):
        super(DatasetFromBatchPath, self).__init__()
        
        with open(split_path, 'rb') as split_file:
          self.split_dict = pickle.load(split_file, encoding='bytes')
        
        self.batches = {
                        'data_batch_1':{}, 
                        'data_batch_2':{},
                        'data_batch_3':{},
                        'data_batch_4':{},
                        'data_batch_5':{},
                       }
        
        for key in self.batches:
            with open(cifar_path + key, 'rb') as batch_file:
              self.batches[key] = pickle.load(batch_file, encoding='bytes')
            
        self.transform = transform
    def __len__(self):
        length = 0
        
        for key in self.split_dict:
            length += len(self.split_dict[key])
            
        return length
        
    def __getitem__(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
                    
        batch_file = self.split_dict[ids%10][int(ids/10)][0]
        img_index = self.split_dict[ids%10][int(ids/10)][1]
        
        img = self.batches[batch_file][b'data'][img_index]
        img = np.transpose(np.reshape(img,(3, 32,32)), (1,2,0))
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        
        label = ids%10
        
        return img,label