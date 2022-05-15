import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader
from DTransformer import ViT, DTransformer
from torch.optim import lr_scheduler

from tqdm import tqdm
import numpy as np
from random import randint
import random
import warmup_scheduler
from resnet import *
from utils import save_checkpoint, save_loss, cifar10_dataset

def train(name, channel_size_lst, div_indices, start_index, end_index, dataset_path, batch_size, num_workers, lr, epochs, warm_up_epoch, teacher_model, dim, depth, mlp_dim, heads, **_):
    model, teacher = build_model(channel_size_lst, div_indices, start_index, end_index, teacher_model, dim, depth, mlp_dim, heads)
    train_dataset, _ = cifar10_dataset(dataset_path)
    train_dataset, val_dataset = random_split(train_dataset, [47500, 2500])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr)
    base_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=epochs)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=warm_up_epoch, after_scheduler=base_scheduler)
    train_losses = []
    val_losses = []
    optimizer.zero_grad()
    optimizer.step()
    best_loss = 100000
    for epoch in range(epochs):    
        scheduler.step(epoch + 1)
        train_epoch(model, train_dataloader, optimizer, teacher, train_losses)
        epoch_loss = val_epoch(model, val_dataloader, teacher, val_losses)
        if epoch_loss < best_loss:
            save_checkpoint(f'pretrain-checkpoints/{name}.pt', model)
            best_loss = epoch_loss
    save_loss(name, [train_losses, val_losses])

def train_epoch(model, train_dataloader, optimizer, teacher, train_losses):
    # step variable
    epoch_total_loss = 0
    epoch_steps = 0
    
    #set train mode
    model.train()
    for batch_idx, (inputs, labels) in tqdm(enumerate(train_dataloader)):
        # backprop
        optimizer.zero_grad()
        inputs, labels = inputs.cuda(), labels.cuda()
        loss = model.pretrain_loss(inputs, teacher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # update step
        epoch_total_loss += loss.item()
        epoch_steps += 1


    epoch_loss = epoch_total_loss / epoch_steps

    print(f"learning_rate ${optimizer.param_groups[0]['lr']}")
    print(f"train_epoch_loss ${epoch_loss}")
    train_losses.append(epoch_loss)
    return epoch_steps


def val_epoch(model, val_dataloader, teacher, val_losses):
    # step_variable
    epoch_total_loss = 0
    epoch_steps = 0
    
    # set eval mode
    model.eval()
    for batch_idx, (inputs, labels) in tqdm(enumerate(val_dataloader, 0)):
        with torch.no_grad():
            # inference
            inputs, labels = inputs.cuda(), labels.cuda()
            loss = model.pretrain_loss(inputs, teacher)

            epoch_total_loss += loss.item()
            epoch_steps += 1
    epoch_loss = epoch_total_loss / epoch_steps
    print(f"validation_loss ${epoch_loss}")
    val_losses.append(epoch_loss)
    return epoch_loss

def build_model(channel_size_lst, div_indices, start_index, end_index, teacher_model, dim, depth, mlp_dim, heads):
    vit = ViT(image_size=32, patch_size=4, dim=dim, depth=depth, mlp_dim=mlp_dim, heads=heads)
    if teacher_model == 'resnet20':
        teacher = nn.DataParallel(resnet20())
        teacher_path = 'resnet20-12fca82f.th'
    elif teacher_model == 'resnet32':
        teacher = nn.DataParallel(resnet32())
        teacher_path = 'resnet32-d509ac18.th'
    elif teacher_model == 'resnet56':
        teacher = nn.DataParallel(resnet56())
        teacher_path = 'resnet56-4bfd9763.th'
    elif teacher_model == 'resnet110':
        teacher = nn.DataParallel(resnet110())
        teacher_path = 'resnet110-1d1ed7c2.th'
    else:
        raise Exception('teacher_model is not defined')
    teacher.load_state_dict(torch.load(f'resnet-checkpoints/{teacher_path}')['state_dict'])
    teacher = teacher.module
    teacher.eval()
    teacher.cuda()
    model = DTransformer(vit, dim, channel_size_lst=channel_size_lst, div_indices = div_indices, start_index=start_index, end_index=end_index)
    model.cuda()
    return model, teacher


