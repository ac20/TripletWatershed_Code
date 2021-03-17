"""
The idea is to use the watershed layer for classification. Recall that we 
golbablize the watershed first and then train the neural network.
"""

import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset

from Datasets.HyperspectralDatasets import Hyperspectral_Dataset
from Datasets.WatershedDataset import watershed_augment_dataset
from Models.modelA import modelA
from WatershedLabels import get_watershed_labels
from utils import AverageMeter, hash_param, EarlyStopping
from Evaluate import evaluate_model_watershed, mean_average_precision

import argparse
from tqdm import tqdm


class tripletDataset(Dataset):
    def __init__(self, base_dataset, size_dataset=60000):
        self.base_dataset = base_dataset
        self.labels = base_dataset.labels
        self.list_labels = np.array(np.unique(self.labels))
        self.size_dataset = size_dataset

        self.labels_choose = self.labels

    def __getitem__(self, idx):
        label_pos, label_neg = np.random.choice(self.list_labels, size=2, replace=False)
        anc_idx, pos_idx = np.random.choice(np.where(self.labels_choose == label_pos)[0], size=2, replace=True)
        neg_idx = np.random.choice(np.where(self.labels_choose == label_neg)[0], size=1, replace=False)[0]

        Xanc, _ = self.base_dataset.__getitem__(anc_idx)
        Xpos, _ = self.base_dataset.__getitem__(pos_idx)
        Xneg, _ = self.base_dataset.__getitem__(neg_idx)

        return Xanc, Xpos, Xneg

    def __len__(self):
        return self.size_dataset


def train_all_layers(model, dataset_input, optimizer, scheduler, **param):
    """Train the network parameters except last layer.
    """

    triplet_dataset = tripletDataset(dataset_input, size_dataset=512*100)
    dataloader = DataLoader(triplet_dataset, batch_size=512)
    loss_fn = nn.TripletMarginLoss(reduction='mean')
    optimizer.zero_grad()
    track_loss = AverageMeter('loss')
    tqdm_tot = len(dataloader)
    for batch_no, (Xanc, Xpos, Xneg) in tqdm(enumerate(dataloader), total=tqdm_tot):
        Xanc = Xanc.to(param['device'])
        Xpos = Xpos.to(param['device'])
        Xneg = Xneg.to(param['device'])

        yanc = model.forward_rep(Xanc)
        ypos = model.forward_rep(Xpos)
        yneg = model.forward_rep(Xneg)

        loss = loss_fn(yanc, ypos, yneg)
        loss.backward()
        track_loss.update(loss.item(), 1)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return track_loss.average


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to work on", choices=['indianpines', 'paviaU', 'ksc'], default='indianpines')
    parser.add_argument("--seed", type=int, help="Set the seed for train/test split of dataset", default=42)
    parser.add_argument("--train_size", type=float, help="Train Size", default=0.1)
    parser.add_argument("--embed_dim", type=int, help="Embedding Dimension", default=64)
    parser.add_argument("--semi_supervised", help="To use semi-supervised split", action="store_true")
    args = parser.parse_args()
    print(args)

    param = {}
    param['n_epochs'] = 15
    param['device'] = torch.device("cuda")
    param['embed_dim'] = args.embed_dim
    if args.semi_supervised:
        param['semi_supervised'] = True

    param['dataset'] = args.dataset
    param['seed'] = args.seed
    param['train_size'] = args.train_size

    base_dataset = Hyperspectral_Dataset(**param)
    watershed_dataset = watershed_augment_dataset(base_dataset, **param)

    param['number_features'] = base_dataset.number_features
    param['number_labels'] = base_dataset.number_labels

    model = modelA(**param).to(param['device'])

    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=50)
    optimizer.zero_grad()

    # A fake forward pass to set the values of batch_norm
    model.train()
    dataloader = DataLoader(watershed_dataset, batch_size=256, shuffle=True)
    for batch_no, (X, y) in enumerate(dataloader):
        X = X.to(param['device'])
        model.forward_rep(X)

    # Keep track of the best results
    best_score = 0.0

    for epoch in range(param['n_epochs']):
        model.eval()
        new_labels, watershed_acc = get_watershed_labels(model, base_dataset, **param)
        watershed_dataset.update_watershed_labels(new_labels)

        # Train the network using the watershed labels and triplet loss
        model.train()
        loss = train_all_layers(model, watershed_dataset, optimizer, scheduler, **param)
        print("-- Epoch {:02d} out of {:02d} -- Training Loss : {:0.4f}".format(epoch+1, param['n_epochs'], loss))

    # Save the model and Print Final EValuation Metric
    key = hash_param(param)
    torch.save(model.state_dict(), "./dump/weights_model_"+str(key)+".pth")
    res = evaluate_model_watershed(model, base_dataset, **param)
    res_map = mean_average_precision(model, base_dataset, **param)
    print("-------------------------------------------------------------")
    print("-------------------------- RESULTS --------------------------")
    print("-------------------------------------------------------------")
    print('Train OA: {: 0.4f} AA: {: 0.4f} Kappa: {: 0.4f}'.format(res[0], res[1], res[2]))
    print(' Test OA: {: 0.4f} AA: {: 0.4f} Kappa: {: 0.4f}'.format(res[3], res[4], res[5]))
    print("-------------------------------------------------------------")
    print("Mean Average Precision : {:0.4f}".format(res_map))
