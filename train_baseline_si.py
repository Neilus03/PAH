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

# SI hyperparameters:
lambda_si = 50.0  # Adjust as needed
xi = 0.1           # small dampening term to avoid division by zero

# Data structures for SI
# si_omega will accumulate across tasks
si_omega = {}      # importance weights for each param
si_old_params = {} # parameter values after learning each task
si_w = {}           # per-task importance accumulators

os.makedirs('results', exist_ok=True)
num = time.strftime("%m%d-%H%M%S")
results_dir = 'results/' + num + '-SI'
os.makedirs(results_dir, exist_ok=True)

logger = logger(results_dir)

logger.log('Starting training with SI...')
logger.log(f'Training hyperparameters: EPOCHS_PER_TIMESTEP={EPOCHS_PER_TIMESTEP}, lr={lr}, l2_reg={l2_reg}, lambda_si={lambda_si}')
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
        for p in param_group['params']:
            unique_params.add(p)
    print(f"Total Optimized Parameters: {total_params} ({sum(p.numel() for p in unique_params)} unique)")
    print("===================================\n")


def init_si_buffers(model):
    # Initialize si_w and si_old_params if not already initialized
    for n, p in model.named_parameters():
        if p.requires_grad:
            if n not in si_omega:
                si_omega[n] = torch.zeros_like(p, device=device)
            if n not in si_old_params:
                si_old_params[n] = p.detach().clone()
            si_w[n] = torch.zeros_like(p, device=device)

def update_si_after_step(model, old_params, gradients):
    # Update si_w using parameter changes and gradients
    # w += Δθ * (-g),  where g is gradient and Δθ = θ_current - θ_old
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            delta = p.detach() - old_params[n]
            # gradient was computed as dL/dθ, so negative gradient is direction of improvement
            si_w[n] += delta * (-gradients[n])  # accumulate importance
    # After update, store new old_params for next step
    for n, p in model.named_parameters():
        if p.requires_grad:
            old_params[n] = p.detach().clone()

def si_penalty(model):
    # Compute SI penalty: sum over parameters of Ω * (θ - θ_old)^2 / 2
    # where Ω = si_omega[n] from previous tasks.
    loss = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad:
            diff = p - si_old_params[n]
            loss += (si_omega[n] * diff.pow(2)).sum() / 2.0
    return lambda_si * loss

def finalize_task_si(model):
    # After finishing a task, update si_omega and si_old_params
    # Ω_i = si_omega[i] + w[i]/((θ_i - θ_old_i)^2 + ξ)
    for n, p in model.named_parameters():
        if p.requires_grad:
            diff = p.detach() - si_old_params[n]
            si_omega[n] += si_w[n] / (diff.pow(2) + xi)
            # Update si_old_params to current param values
            si_old_params[n] = p.detach().clone()
            # Reset w
            si_w[n].zero_()
            
def ensure_si_buffers_exist_for_all_params(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            if n not in si_omega:
                si_omega[n] = torch.zeros_like(p, device=device)
            if n not in si_old_params:
                si_old_params[n] = p.detach().clone()
            if n not in si_w:
                si_w[n] = torch.zeros_like(p, device=device)


frozen = 'frozen' if freeze_backbone else ''
with wandb.init(project='HyperCMTL', name=f'SI_Baseline-{dataset}-{backbone_name}{frozen}', group="CorrectSplit") as run:
    count_optimizer_parameters(opt)
    # Initialize SI buffers before training on the first task
    init_si_buffers(model)

    for t, (task_train, task_val) in timestep_tasks.items():
        task_train.num_classes = len(timestep_task_classes[t])
        print(f'task_train.num_classes: {task_train.num_classes}')
        logger.log(f"Training on task id: {t}  (classification between: {task_metadata[t]})")
        if t not in model.task_heads:
            task_head = TaskHead(input_size=1000, projection_size=64, num_classes=task_train.num_classes).to(device)
            model.add_task(t, task_head)
            opt.add_param_group({'params': task_head.parameters()})
            count_optimizer_parameters(opt)
            # Now ensure that SI buffers include new params
            ensure_si_buffers_exist_for_all_params(model)

        train_loader, val_loader = [utils.data.DataLoader(data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
                                        for data in (task_train, task_val)]
        
        # For SI, track old params at start of this task
        task_old_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

        for e in range(EPOCHS_PER_TIMESTEP):
            epoch_train_losses, epoch_train_accs = [], []
            progress_bar = tqdm(train_loader, ncols=100) if show_progress else train_loader
            for batch_idx, batch in enumerate(progress_bar):
                x, y, task_ids = batch
                x, y = x.to(device), y.to(device)
                task_id = task_ids[0]

                # Store current params for delta computation
                pre_update_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                
                opt.zero_grad()
                pred = model(x, task_id)
                hard_loss = loss_fn(pred, y)

                # SI penalty
                penalty = 0.0
                if t > 0:  # Only after first task do we have meaningful si_omega
                    penalty = si_penalty(model)
                total_loss = hard_loss + penalty

                wandb.log({'hard_loss': hard_loss.item(), 'si_penalty': float(penalty), 'train_loss': total_loss.item(), 'epoch': e, 'task_id': t, 'batch_idx': batch_idx})

                total_loss.backward()
                
                # Store gradients for SI update
                grads = {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
                
                opt.step()

                # Update SI w
                update_si_after_step(model, pre_update_params, grads)

                epoch_train_losses.append(hard_loss.item())
                epoch_train_accs.append(get_batch_acc(pred, y))
                metrics['steps_trained'] += 1

                if show_progress:
                    progress_bar.set_description((f'E{e} loss:{hard_loss:.2f}, acc:{epoch_train_accs[-1]:>5.1%}'))

            avg_val_loss, avg_val_acc = evaluate_model(model, val_loader, loss_fn)
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
                                  model_name=f'SI at t={t}', 
                                  prev_accs=prev_test_accs,
                                  verbose=True,
                                  task_metadata=task_metadata)
        wandb.log({'mean_test_acc': np.mean(test_accs), 'task_id': t})
        prev_test_accs.append(test_accs)

        # Finalize SI after finishing this task
        finalize_task_si(model)

    final_avg_test_acc = np.mean(test_accs)
    print(f'Final average test accuracy: {final_avg_test_acc:.2%}')
    wandb.summary['final_avg_test_acc'] = final_avg_test_acc
