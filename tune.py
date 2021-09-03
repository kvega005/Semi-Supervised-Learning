import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import pickle

from torchvision import models, datasets
from collections import defaultdict
from tqdm import tqdm

from modules import *

CIFAR_DIRECTORY = 'cifar-10-batches-py/'
SPLIT_FOLDER = 'cifar-10-splits/'
MODEL_FOLDER = 'models/'

PARAMS = {
    'image_size':224,
    'learning_rate':3e-3,
    'batch_size': 32,
    'num_epochs':25,
    'checkpoint_epochs': 10,
    'num_workers': 8,
    'device': 'cuda'
}

if __name__ == "__main__":
    torch.cuda.set_device(2)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_percentage_train", required=True, type=str)
    parser.add_argument("--split_percentage_tune", required=True, type=str)
    
    args = parser.parse_args()
    
    print("Tune split:", args.split_percentage_tune)
    
    SPLIT_PATH_TRAIN = SPLIT_FOLDER + args.split_percentage_tune + "_percent_split"
    MODEL_PATH = MODEL_FOLDER + f"training-{args.split_percentage_train}-final.pt"
    
    train_dataset = DatasetFromBatchPath(
        SPLIT_PATH_TRAIN, 
        CIFAR_DIRECTORY, 
        TransformsSimCLR(size=PARAMS['image_size']).test_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=PARAMS['batch_size'],
    )
    test_dataset = datasets.CIFAR10(
        CIFAR_DIRECTORY,
        train=False,
        download=True,
        transform=TransformsSimCLR(size=PARAMS['image_size']).test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=PARAMS['batch_size'],
        drop_last=True,
    )
    
    resnet = models.resnet50(pretrained=False)
    resnet.load_state_dict(torch.load(MODEL_PATH, map_location=PARAMS['device']))
    resnet = resnet.to(PARAMS['device'])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=resnet.parameters(), lr=PARAMS['learning_rate'])
    
    for epoch in range(PARAMS['num_epochs']):
        metrics = defaultdict(list)
        
        print(f"Epoch {epoch}/{PARAMS['num_epochs']}:")
        
        for (h, y) in tqdm(train_loader):
            h = h.to(PARAMS['device'])
            y = y.to(PARAMS['device'])

            outputs = resnet(h)

            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # calculate accuracy and save metrics
            accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            metrics["Loss/train"].append(loss.item())
            metrics["Accuracy/train"].append(accuracy)
            
    metrics = defaultdict(list)
    for (h, y) in tqdm(test_loader):
        h = h.to(PARAMS['device'])
        y = y.to(PARAMS['device'])

        outputs = resnet(h)

        # calculate accuracy and save metrics
        accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
        metrics["Accuracy/test"].append(accuracy)
        
    f = open("result_%s_%s.txt"%(args.split_percentage_train, args.split_percentage_tune),"w")
    f.write(f"Final test performance: " + "\t".join([f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))
    f.close()