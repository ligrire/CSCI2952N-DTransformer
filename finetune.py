import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, Dataset, DataLoader
from DTransformer import ViT, ViTWrapper, DTransformer
from torch.optim import lr_scheduler
from utils import save_loss, cifar10_dataset, save_checkpoint
from tqdm import tqdm
import numpy as np
from random import randint
import random
import warmup_scheduler


def train(channel_size_lst, div_indices, start_index, end_index, name, dataset_path,batch_size, num_workers, dim, depth, mlp_dim, heads, finetune_epochs, warm_up_epoch, finetune_lr, **_):
    model = build_model(channel_size_lst, div_indices, start_index, end_index, name, dim, depth, mlp_dim, heads)
    train_dataset, val_dataset = cifar10_dataset(dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=finetune_lr)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epochs)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=warm_up_epoch, after_scheduler=base_scheduler)
    optimizer.zero_grad()
    optimizer.step()
    train_accuracies = []
    val_accuracies = []
    for epoch in range(finetune_epochs):    
        scheduler.step(epoch + 1)
        train_epoch(model, train_dataloader, optimizer, loss_fn, train_accuracies)
        val_epoch(model, val_dataloader, loss_fn, val_accuracies)
    save_loss(name, [train_accuracies, val_accuracies])
    save_checkpoint(f'finetune-checkpoints/{name}.pt', model)


def train_epoch(model, train_dataloader, optimizer, loss_fn, train_accuracies):
    # step variable
    epoch_total_loss = 0
    epoch_steps = 0
    sample_size = 0
    correct = 0
    
    patch_num = 64
    #set train mode
    model.train()
    for batch_idx, (inputs, labels) in tqdm(enumerate(train_dataloader)):
        # backprop
        optimizer.zero_grad()
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        sample_size += labels.size(0)
        
        # update step
        epoch_total_loss += loss.item()
        epoch_steps += 1
        # change learning rate


    epoch_loss = epoch_total_loss / epoch_steps
    accuracy = correct / sample_size

    print(f"learning_rate ${optimizer.param_groups[0]['lr']}")
    print(f"train_epoch_loss ${epoch_loss}")
    print(f"training_accuracy ${accuracy}")
    train_accuracies.append(accuracy)
    return epoch_steps

def val_epoch(model, val_dataloader, loss_fn, val_accuracies):
    # step_variable
    epoch_total_loss = 0
    epoch_steps = 0
    sample_size = 0
    correct = 0
    
    # set eval mode
    model.eval()
    for batch_idx, (inputs, labels) in tqdm(enumerate(val_dataloader, 0)):
        with torch.no_grad():
            # inference
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            sample_size += labels.size(0)
            epoch_total_loss += loss.item()
            epoch_steps += 1
    epoch_loss = epoch_total_loss / epoch_steps
    accuracy = correct / sample_size
    val_accuracies.append(accuracy)
    print(f"validation_loss ${epoch_loss}")
    print(f"accuracy ${accuracy}")


def build_model(channel_size_lst, div_indices, start_index, end_index, name, dim, depth, mlp_dim, heads):
    vit = ViT(image_size=32, patch_size=4, dim=dim, depth=depth, mlp_dim=mlp_dim, heads=heads)
    d = DTransformer(vit, dim, channel_size_lst=channel_size_lst, div_indices = div_indices, start_index=start_index, end_index=end_index)
    d.load_state_dict(torch.load(f'pretrain-checkpoints/{name}.pt'))
    model = ViTWrapper(vit, dim, 10)
    model.cuda()
    return model
