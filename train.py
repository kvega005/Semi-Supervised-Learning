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

PARAMS = {
    'image_size':224,
    'learning_rate':3e-2,
    'batch_size': 32,
    'num_epochs':100,
    'checkpoint_epochs': 10,
    'num_workers': 8,
    'device': 'cuda'
}

if __name__ == "__main__":
    torch.cuda.set_device(2)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_percentage", required=True, type=str)
    args = parser.parse_args()
    
    print("Train split:", args.split_percentage)
    
    SPLIT_PATH_TRAIN = SPLIT_FOLDER + args.split_percentage + "_percent_split"

    train_dataset = DatasetFromBatchPath(SPLIT_PATH_TRAIN, CIFAR_DIRECTORY, TransformsSimCLR(224))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=PARAMS['batch_size'],
    )

    resnet = models.resnet50(pretrained=False)
    
    model = BYOL(resnet, image_size = PARAMS["image_size"], hidden_layer="avgpool")
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = PARAMS["learning_rate"])

    global_step = 0

    for epoch in range(PARAMS["num_epochs"]):
        metrics = defaultdict(list)
        
        print(f"Epoch {epoch}/{PARAMS['num_epochs']}:")
        
        for (View_1,View_2),_ in tqdm(train_loader):
            View_1 = View_1.cuda(non_blocking=True)
            View_2 = View_2.cuda(non_blocking=True)

            loss = model(View_1, View_2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_moving_average()  # update moving average of target encoder

            metrics["Loss/train"].append(loss.item())
            global_step += 1

        torch.save(resnet.state_dict(), f"./models/training-{args.split_percentage}-final.pt")