# set up the environment and install any missing packages:
#!pip install torch torchvision numpy scipy matplotlib pandas pillow tqdm MLclf

import torch
from torch import nn, utils
import torch.nn.functional as F

from torch.utils.data import DataLoader, BatchSampler
from torchvision import models, datasets, transforms

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

from config import config
from utils import (inspect_batch, test_evaluate, training_plot, setup_dataset, 
                   inspect_task, evaluate_model, evaluate_model_prototypes, 
                   get_batch_acc, logger)

import wandb
from copy import deepcopy
import time
import logging
import pdb
import random
from torch.utils.data import Sampler

from config import *

torch.manual_seed(config['misc']['random_seed'])
np.random.seed(config['misc']['random_seed'])
random.seed(config['misc']['random_seed'])
torch.cuda.manual_seed_all(config['misc']['random_seed'])
torch.cuda.manual_seed(config['misc']['random_seed'])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(config['misc']['device'] if torch.cuda.is_available() else 'cpu')

### Classification head
class TaskHead(nn.Module):
    def __init__(self, input_size: int,
                 projection_size: int,
                 num_classes: int,
                 dropout: float=0.,
                 device=device):
        super().__init__()
        
        self.projection = nn.Linear(input_size, projection_size)
        self.classifier = nn.Linear(projection_size, num_classes)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.relu = nn.ReLU()
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.projection(self.relu(self.dropout(x)))
        x = self.classifier(self.relu(self.dropout(x)))
        return x

### Multi-task model
class MultitaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, device=device):
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleDict()
        self.relu = nn.ReLU()
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor, task_id: int):
        task_id = str(int(task_id))
        assert task_id in self.task_heads, f"no head exists for task id {task_id}"
        chosen_head = self.task_heads[task_id]
        x = self.relu(self.backbone(x))
        x = chosen_head(x)
        return x

    def add_task(self, task_id: int, head: nn.Module):
        self.task_heads[str(task_id)] = head

    @property
    def num_task_heads(self):
        return len(self.task_heads)


torch.manual_seed(0)
torch.cuda.empty_cache()

### dataset hyperparameters:
VAL_FRAC = 0.1
TEST_FRAC = 0.1
BATCH_SIZE = 512
dataset = "Split-CIFAR100"
NUM_TASKS = 10

### training hyperparameters:
EPOCHS_PER_TIMESTEP = 15
lr     = 1e-4
l2_reg = 1e-6
freeze_backbone = config['model']['frozen_backbone']

# EWC hyperparameters:
lambda_ewc = 50.0  # Adjust as needed for regularization strength
ewc_params = []    # Will store (params, fisher) after each task

os.makedirs('results', exist_ok=True)
num = time.strftime("%m%d-%H%M%S")
results_dir = 'results/' + num + '-EWC'
os.makedirs(results_dir, exist_ok=True)

logger = logger(results_dir)

logger.log('Starting training with EWC...')
logger.log(f'Training hyperparameters: EPOCHS_PER_TIMESTEP={EPOCHS_PER_TIMESTEP}, lr={lr}, l2_reg={l2_reg}, lambda_ewc={lambda_ewc}')
logger.log(f'Training on device: {device}')

data = setup_dataset(dataset, data_dir='./data', num_tasks=NUM_TASKS, val_frac=VAL_FRAC, test_frac=TEST_FRAC, batch_size=BATCH_SIZE)
timestep_tasks = data['timestep_tasks']
timestep_task_classes = data['timestep_task_classes']
final_test_loader = data['final_test_loader']
task_metadata = data['task_metadata']
task_test_sets = data['task_test_sets']
images_per_class = data['images_per_class']

backbone_name = 'resnet50'
task_head_projection_size = 256
hyper_hidden_features = 256
hyper_hidden_layers = 4

backbone = models.resnet50(pretrained=True)
backbone.num_features = backbone.fc.in_features

if freeze_backbone:
    for param in backbone.parameters():
        param.requires_grad = False

model = MultitaskModel(backbone.to(device))

#logger.log(f'Model architecture: {model}')
logger.log(f"Model initialized with backbone_config={backbone}, task_head_projection_size={task_head_projection_size}, hyper_hidden_features={hyper_hidden_features}, hyper_hidden_layers={hyper_hidden_layers}")

opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
loss_fn = nn.CrossEntropyLoss()

plot_training = True
show_progress = True
verbose = True

metrics = { 'train_losses': [],
              'train_accs': [],
              'val_losses': [],
                'val_accs': [],
             'epoch_steps': [],
            'CL_timesteps': [],
            'best_val_acc': 0.0,
           'steps_trained': 0,}

prev_test_accs = []

print("Starting training")


def count_optimizer_parameters(optimizer: torch.optim.Optimizer) -> None:
    total_params = 0
    unique_params = set()
    print("\n=== Optimizer Parameter Groups ===")
    for idx, param_group in enumerate(optimizer.param_groups):
        num_params = len(param_group['params'])
        num_params_in_group = sum(p.numel() for p in param_group['params'])
        print(f"Parameter Group {idx + 1}: {num_params} parameters, Total Parameters: {num_params_in_group}")
        total_params += num_params_in_group
        total_optimized_params = sum(p.numel() for p in unique_params if p.requires_grad)
        for p in param_group['params']:
            unique_params.add(p)
    print(f"Total Optimized Parameters: {total_params} ({sum(p.numel() for p in unique_params)} unique)")
    print(f"Total Optimized Parameters: {total_optimized_params} ({len(unique_params)} unique)")
    wandb.log({'total_params': total_params})
    wandb.log({'total_optimized_params': total_optimized_params})
    print("===================================\n")


def compute_fisher(model, data_loader, device, sample_size=2000):
    # Compute the Fisher Information matrix for the parameters by collecting gradients of the log-likelihood for samples.
    model.eval()
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}

    data_iter = iter(data_loader)
    count = 0
    for x, y, t_id in data_iter:
        x, y = x.to(device), y.to(device)
        pred = model(x, t_id[0])
        log_likelihood = F.log_softmax(pred, dim=1)
        # we only take the diagonal of the fisher, so we can just pick the labels
        # for simplicity: sample the predicted labels (like in the original EWC paper)
        chosen_log_likelihood = log_likelihood[range(len(y)), y]
        loss = - chosen_log_likelihood.mean()

        model.zero_grad()
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.pow(2)

        count += len(y)
        if count >= sample_size:
            break

    # Normalize by number of samples considered
    for n in fisher:
        fisher[n] = fisher[n] / count

    return fisher

def ewc_loss(model, ewc_params, lambda_ewc):
    # Add EWC penalty for all previously learned tasks, ewc_params is a list of (old_params, fisher)
    loss = 0.0
    for old_params, fisher in ewc_params:
        for (n, p) in model.named_parameters():
            if p.requires_grad and n in old_params:
                diff = p - old_params[n]
                loss += (fisher[n] * diff.pow(2)).sum()
    return (lambda_ewc/2.0) * loss

frozen = 'frozen' if freeze_backbone else ''
with wandb.init(project='HyperCMTL', name=f'EWC_Baseline-{dataset}-{backbone_name}{frozen}', group="CorrectSplit") as run:
    count_optimizer_parameters(opt)
    for t, (task_train, task_val) in timestep_tasks.items():
        task_train.num_classes = len(timestep_task_classes[t])
        print(f'task_train.num_classes: {task_train.num_classes}')
        logger.log(f"Training on task id: {t}  (classification between: {task_metadata[t]})")
        if t not in model.task_heads:
            task_head = TaskHead(input_size=1000, projection_size=64, num_classes=task_train.num_classes).to(device)
            model.add_task(t, task_head)
            opt.add_param_group({'params': task_head.parameters()})
            count_optimizer_parameters(opt)

        train_loader, val_loader = [utils.data.DataLoader(data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
                                        for data in (task_train, task_val)]
        
        for e in range(EPOCHS_PER_TIMESTEP):
            epoch_train_losses, epoch_train_accs = [], []
            progress_bar = tqdm(train_loader, ncols=100) if show_progress else train_loader
            for batch_idx, batch in enumerate(progress_bar):
                x, y, task_ids = batch
                x, y = x.to(device), y.to(device)
                task_id = task_ids[0]

                opt.zero_grad()
                pred = model(x, task_id)
                hard_loss = loss_fn(pred, y)

                # Add EWC penalty if we have previously learned tasks
                penalty = 0.0
                if len(ewc_params) > 0:
                    penalty = ewc_loss(model, ewc_params, lambda_ewc)
                total_loss = hard_loss + penalty

                wandb.log({'hard_loss': hard_loss.item(), 'ewc_penalty': float(penalty), 'train_loss': total_loss.item(), 'epoch': e, 'task_id': t, 'batch_idx': batch_idx})

                total_loss.backward()
                opt.step()

                epoch_train_losses.append(hard_loss.item())
                epoch_train_accs.append(get_batch_acc(pred, y))
                metrics['steps_trained'] += 1

                if show_progress:
                    progress_bar.set_description((f'E{e} loss:{hard_loss:.2f}, acc:{epoch_train_accs[-1]:>5.1%}'))

            avg_val_loss, avg_val_acc = evaluate_model(model, val_loader, loss_fn, device= device)
            wandb.log({'val_loss': avg_val_loss, 'val_accuracy': avg_val_acc, 'epoch': e, 'task_id': t})

            metrics['epoch_steps'].append(metrics['steps_trained'])
            metrics['train_losses'].extend(epoch_train_losses)
            metrics['train_accs'].extend(epoch_train_accs)
            metrics['val_losses'].append(avg_val_loss)
            metrics['val_accs'].append(avg_val_acc)
            
            if show_progress:
                print((f'E{e} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
                       f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%}'))

            if avg_val_acc > metrics['best_val_acc']:
                metrics['best_val_acc'] = avg_val_acc

        metrics['CL_timesteps'].append(metrics['steps_trained'])
        if plot_training and len(metrics['val_losses']) > 0:
            training_plot(metrics, show_timesteps=True)

        if verbose:
            print(f'Best validation accuracy: {metrics["best_val_acc"]:.2%}\n')
        metrics['best_val_acc'] = 0.0   

        test_accs = test_evaluate(model, task_test_sets[:t+1],
                                  task_test_sets=task_test_sets, 
                                  model_name=f'EWC at t={t}', 
                                  prev_accs=prev_test_accs,
                                  verbose=True,
                                  task_metadata=task_metadata,
                                  device=device)
        wandb.log({'mean_test_acc': np.mean(test_accs), 'task_id': t})
        prev_test_accs.append(test_accs)

        # Compute Fisher and store parameters for EWC after finishing this task
        fisher = compute_fisher(model, train_loader, device)
        old_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        ewc_params.append((old_params, fisher))

    final_avg_test_acc = np.mean(test_accs)
    print(f'Final average test accuracy: {final_avg_test_acc:.2%}')
    wandb.summary['final_avg_test_acc'] = final_avg_test_acc
