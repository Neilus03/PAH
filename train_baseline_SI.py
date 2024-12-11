#/home/ndelafuente/TSR/train/Split-CIFAR100/SI_baseline.py
# FILE TO TRAIN SI BASELINE on Split-CIFAR100

import torch
from torch import nn, utils
import torch.nn.functional as F
import wandb 
from copy import deepcopy

from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import sys
import random
import time

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from networks.backbones import ResNet50, MobileNetV2, EfficientNetB0, ViT, ResNet18
from networks.networks_baseline import MultitaskModel_Baseline, TaskHead_Baseline, TaskHead_simple, MultitaskModel_Baseline_notaskid

from utils import *


# ------------------ Synaptic Intelligence (SI) Auxiliary Functions ------------------ #

def initialize_si_structures(model, device):
    # Omega stores importance for each parameter accumulated over tasks
    Omega = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    return Omega

def zero_like_model_params(model, device):
    return {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}

def get_params_dict(model):
    return {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

def si_penalty(model, old_params, Omega, si_lambda):
    # Compute SI penalty: sum over i Omega_i * (p_i - old_p_i)^2 / 2
    loss = torch.tensor(0.0, device=model.device)
    for n, p in model.named_parameters():
        if p.requires_grad and n in old_params:
            loss += (Omega[n] * (p - old_params[n]).pow(2)).sum() / 2
    return si_lambda * loss

def update_omega(Omega, W, old_params, model, initial_params, epsilon=0.00001):
    # After finishing a task:
    # Omega_i += W_i / ((param_final_i - param_init_i)^2 + epsilon)
    for n, p in model.named_parameters():
        if p.requires_grad and n in W:
            param_diff = p.data - initial_params[n]
            denominator = param_diff.pow(2) + epsilon
            Omega[n] += W[n] / denominator

def accumulate_w(W, old_params, model):
    # W_i is accumulated over the course of training the task: Δp_i * g_i
    # Here:
    # Before calling this, gradients are computed (after backward).
    # Then we do an optimizer step.
    # We need to track param change: Δp_i = p_new - p_old
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            old_val = old_params[n]
            new_val = p.data
            delta_p = new_val - old_val
            g_i = p.grad.data
            W[n] += delta_p * g_i
    # After accumulation, update old_params to reflect new values
    for n, p in model.named_parameters():
        if p.requires_grad:
            old_params[n] = p.data.clone()

# ------------------ Main Training Script ------------------ #

config = config_load(sys.argv[1])['config']
seed_everything(config['misc']['seed'])

device = torch.device(config["misc"]["device"] if torch.cuda.is_available() else "cpu")

num = time.strftime("%Y%m%d-%H%M%S")
name_run = f"{config['logging']['name']}-{num}"
results_dir = os.path.join(config["logging"]["results_dir"], name_run)
os.makedirs(results_dir, exist_ok=True)

logger = logger(results_dir)

# Log initial info
logger.log(f"Starting training for {config['logging']['name']}")
logger.log(f"Configuration: {config}")
logger.log(f"Device: {device}")
logger.log(f"Random seed: {config['misc']['seed']}")

#Load dataset
if config["dataset"]["dataset"] != "TinyImageNet":
    data = setup_dataset(dataset_name = config["dataset"]["dataset"],
                        data_dir = config["dataset"]["data_dir"], 
                        num_tasks=config["dataset"]["NUM_TASKS"],
                        val_frac=config["dataset"]["VAL_FRAC"],
                        test_frac=config["dataset"]["TEST_FRAC"],
                        batch_size=config["dataset"]["BATCH_SIZE"])
else:
    data = setup_tinyimagenet(data_dir = config["dataset"]["data_dir"], 
                        num_tasks=config["dataset"]["NUM_TASKS"],
                        val_frac=config["dataset"]["VAL_FRAC"],
                        test_frac=config["dataset"]["TEST_FRAC"],
                        batch_size=config["dataset"]["BATCH_SIZE"])

backbone_dict = {
    'resnet50': ResNet50,
    'mobilenetv2': MobileNetV2,
    'efficientnetb0': EfficientNetB0,
    'vit': ViT,
    'resnet18': ResNet18
}

backbone_name = config["model"]["backbone"]

backbone = backbone_dict[backbone_name](device=device, pretrained=True)
logger.log(f"Using backbone: {backbone_name}")

if config["model"]["frozen_backbone"] == True:
    for param in backbone.parameters():
        param.requires_grad = False

# Create model
baseline_si = MultitaskModel_Baseline_notaskid(backbone, device)

logger.log(f"Model created!")
logger.log(f"Model initialized with freeze_backbone={config['model']['frozen_backbone']}, config={config['model']}")

# Initialize optimizer and loss
optimizer = setup_optimizer(
                model=baseline_si,
                lr=config["training"]["lr"],
                l2_reg=config["training"]["l2_reg"],
                optimizer=config["training"]["optimizer"]
            )
loss_fn = nn.CrossEntropyLoss()

# Track metrics for plotting
metrics = { 'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': [],
            'epoch_steps': [], 
            'CL_timesteps': [],
            'best_val_acc': 0.0,
            'steps_trained': 0,
            'soft_losses': [],
           }

prev_test_accs = []
logger.log(f"Starting training for {config['logging']['name']}")

# SI parameters and structures
Omega = initialize_si_structures(baseline_si, device)
old_params = get_params_dict(baseline_si) # initial old params (before first task)
si_lambda = config["training"]["si_lambda"]
epsilon = config["training"]["si_epsilon"]

task_train_num_classes = len(data['timestep_task_classes'][0])
task_head = TaskHead_simple(input_size=baseline_si.backbone.num_features, 
                                num_classes=task_train_num_classes,
                                device=device)
# Add task head to model
baseline_si.add_task(0, task_head)
optimizer.add_param_group({'params': task_head.parameters()})
logger.log(f"Task head added. same task head will be used for all tasks")

with wandb.init(project='HyperCMTL', entity='pilligua2', name=f'{name_run}', config=config, group=config['logging']['group']) as run:
    #Outer loop for each task, in sequence
    for t, (task_train, task_val) in data['timestep_tasks'].items():
        task_train.num_classes = len(data['timestep_task_classes'][t])
        logger.log(f"Task {t}: {task_train.num_classes} classes\n: {data['task_metadata'][t]}")
            
        # Build training and validation dataloaders
        train_loader, val_loader = [utils.data.DataLoader(d,
                                        batch_size=config["dataset"]["BATCH_SIZE"],
                                        shuffle=True) for d in (task_train, task_val)]
        
        # Store initial params at start of the task
        initial_params = get_params_dict(baseline_si)
        # Initialize W for this task
        W = zero_like_model_params(baseline_si, device)
        
        # After setting initial_params and W, we also set "old_params" to track param changes during updates
        temp_old_params = get_params_dict(baseline_si)
        
        #Inner loop for training epochs over the current task
        for e in range(config['training']['epochs_per_timestep']):
            epoch_train_losses, epoch_train_accs = [], []
            epoch_soft_losses = [] # Not used for SI, but keep structure
            
            progress_bar = tqdm(train_loader, ncols=100, total=len(train_loader), desc=f"Task {t}, Epoch {e}") if config["logging"]["show_progress"] else train_loader
            
            #Training loop
            for batch_idx, batch in enumerate(progress_bar):
                x, y, task_ids = batch
                x, y = x.to(device), y.to(device)
                task_id = 0
                
                optimizer.zero_grad()
                
                pred = baseline_si(x, task_id)
                hard_loss = loss_fn(pred, y)
                
                # SI penalty if not the first task
                penalty = 0.0
                if t > 0:
                    penalty = si_penalty(baseline_si, old_params, Omega, si_lambda)
                
                total_loss = hard_loss + penalty
                
                wandb.log({'hard_loss': hard_loss.item(), 'si_penalty': float(penalty), 'train_loss': total_loss.item(), 'epoch': e, 'task_id': t, 'batch_idx': batch_idx})
                
                total_loss.backward()
                
                # Before step, store old param values
                pre_update_params = {n: p.data.clone() for n, p in baseline_si.named_parameters() if p.requires_grad}
                
                optimizer.step()
                
                # After step, accumulate W
                for n, p in baseline_si.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        delta_p = p.data - pre_update_params[n]
                        Omega[n] = torch.zeros_like(p.data)
                        g_i = p.grad.data
                        W[n] += delta_p * g_i
                
                #Track metrics
                epoch_train_losses.append(total_loss.item())
                epoch_train_accs.append(get_batch_acc(pred, y))
                epoch_soft_losses.append(0.0)  
                metrics['steps_trained'] += 1
                
                if config["logging"]["show_progress"]:
                    progress_bar.set_description(f"Task {t}, Epoch {e}, Loss: {total_loss.item():.4f}, Acc: {epoch_train_accs[-1]:.4f}")
                    
            #Evaluate model on current task's validation set after each epoch
            avg_val_loss, avg_val_acc, time_elapsed = evaluate_model_timed(multitask_model=baseline_si,
                                                                          val_loader=val_loader,  
                                                                          loss_fn=loss_fn,
                                                                          device=device)
          
            wandb.log({'val_loss': avg_val_loss, 'val_acc': avg_val_acc, 'epoch': e, 'task_id': t, 'time': time_elapsed})
            
            #Update metrics
            metrics['epoch_steps'].append(metrics['steps_trained'])
            metrics['train_losses'].extend(epoch_train_losses)
            metrics['train_accs'].extend(epoch_train_accs)
            metrics['soft_losses'].extend(epoch_soft_losses)
            metrics['val_losses'].append(avg_val_loss)
            metrics['val_accs'].append(avg_val_acc)
            
            if config["logging"]["show_progress"]:
                logger.log((f'E{e} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
                            f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%} in {time_elapsed:.2f}s'))
                
            if avg_val_acc > metrics['best_val_acc']:
                metrics['best_val_acc'] = avg_val_acc
                logger.log(f"New best validation accuracy: {avg_val_acc:.4f}")
                
        #For plotting
        metrics['CL_timesteps'].append(metrics['steps_trained'])
        
        #If plotting is enabled, plot training curves
        if config["logging"]["plot_training"] and len(metrics['val_losses']) > 0:
            training_plot(metrics, show_timesteps=True)
            
        if config["logging"]["verbose"]:
            logger.log(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
            logger.log(f"Final epoch completed in {time_elapsed:.2f}s")   
        metrics['best_val_acc'] = 0.0
            
        #Evaluate the model on all previous tasks
        metrics_test = test_evaluate_metrics(
                            multitask_model=baseline_si,
                            selected_test_sets=data['task_test_sets'][:t+1],
                            task_test_sets=data['task_test_sets'],
                            model_name=f'SI at t={t}',
                            prev_accs=prev_test_accs,
                            verbose=True,
                            task_metadata=data['task_metadata'],
                            device=device
                            )
            
        wandb.log({**metrics_test, 'task_id': t})
        prev_test_accs.append(metrics_test['task_test_accs'])
        
        # After finishing training this task, update Omega
        baseline_si.eval()
        final_params = get_params_dict(baseline_si)
        update_omega(Omega, W, old_params, baseline_si, initial_params, epsilon=epsilon)
        
        # Set old_params to the parameters at the end of this task
        old_params = final_params
            
    #Log final metrics
    logger.log(f"Task {t} completed!")
    logger.log(f'final metrics: {metrics_test}')
    wandb.summary.update(metrics_test)
