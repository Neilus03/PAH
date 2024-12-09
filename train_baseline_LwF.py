#/home/ndelafuente/TSR/train/Split-CIFAR100/LwF_baseline.py
#FILE TO TRAIN LwF BASELINE on Split-CIFAR100



import torch
from torch import nn, utils
import torch.nn.functional as F
import wandb 
from copy import deepcopy

from torch.utils.data import DataLoader, BatchSampler
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

from networks.backbones import ResNet50, MobileNetV2, EfficientNetB0 
from networks.networks_baseline import MultitaskModel_Baseline, TaskHead_Baseline

from utils import *


config = config_load(sys.argv[1])["config"]

device = torch.device(config["misc"]["device"] if torch.cuda.is_available() else "cpu")
seed_everything(config['misc']['seed'])

num = time.strftime("%Y%m%d-%H%M%S")
name_run = f"{config['logging']['name']}-{num}"
results_dir = os.path.join(config["logging"]["results_dir"], name_run)
os.makedirs(results_dir, exist_ok=True)

logger = logger(results_dir)

#Log initial info
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
    'efficientnetb0': EfficientNetB0
}
backbone_name = config["model"]["backbone"]

backbone = backbone_dict[backbone_name](device=device, pretrained=True)
logger.log(f"Using backbone: {backbone_name}")

if config["model"]["frozen_backbone"] == True:
    for param in backbone.parameters():
        param.requires_grad = False

#Create model
baseline_lwf = MultitaskModel_Baseline(backbone, device)

logger.log(f"Model created!")
logger.log(f"Model initialized with freeze_backbone={config['model']['frozen_backbone']}, config={config['model']}")

#Initialize previous model
previous_model = None

#Initialize optimizer and loss
optimizer = setup_optimizer(
                model=baseline_lwf,
                lr=config["training"]["lr"],
                l2_reg=config["training"]["l2_reg"],
                optimizer=config["training"]["optimizer"]
            )
loss_fn = nn.CrossEntropyLoss()

#Track metrics for plotting
metrics = { 'train_losses': [],
              'train_accs': [],
              'val_losses': [],
                'val_accs': [],
             'epoch_steps': [], # used for plotting val loss at the correct x-position
            'CL_timesteps': [], # used to draw where each new timestep begins
            'best_val_acc': 0.0,
           'steps_trained': 0,
             'soft_losses': [], # distillation loss
          }

prev_test_accs = []

logger.log(f"Starting training for {config['logging']['name']}")

#Training loop
with wandb.init(project='HyperCMTL', entity='pilligua2', name=f'{name_run}', config=config) as run:
    #count_optimizer_parameters(optimizer, logger)
    
    # Outer loop for each task, in sequence
    for t, (task_train, task_val) in data['timestep_tasks'].items():
        task_train.num_classes = len(data['timestep_task_classes'][t])
        logger.log(f"Task {t}: {task_train.num_classes} classes\n: {data['task_metadata'][t]}")
        
        if t not in baseline_lwf.task_heads:
            task_head = TaskHead_Baseline(input_size=baseline_lwf.backbone.num_features, 
                                          projection_size=config["model"]["task_head_projection_size"],
                                          num_classes=task_train.num_classes,
                                          device=device)
            #Add task head to model
            baseline_lwf.add_task(t, task_head)
            optimizer.add_param_group({'params': task_head.parameters()})
            logger.log(f"Task head added for task {t}")
            
        #Build training and validation dataloaders
        train_loader, val_loader = [utils.data.DataLoader(data,
                                        batch_size=config["dataset"]["BATCH_SIZE"],
                                        shuffle=True) for data in (task_train, task_val)]
        
        #Inner loop for training epochs over the current task
        for e in range(config['training']['epochs_per_timestep']):
            epoch_train_losses, epoch_train_accs = [], []
            epoch_soft_losses = []
            
            #progress bar
            progress_bar = tqdm(train_loader, ncols=100, total= len(train_loader), desc=f"Task {t}, Epoch {e}") if config["logging"]["show_progress"] else train_loader
            num_batches = len(train_loader)
            
            #Training loop
            for batch_idx, batch in enumerate(progress_bar):
                #Get data from batch
                x, y, task_ids = batch
                x, y = x.to(device), y.to(device)
                task_id = task_ids[0]
                
                #Zero gradients
                optimizer.zero_grad()
                
                #Get predictions from model
                pred = baseline_lwf(x, task_id)
                
                #Calculate loss
                hard_loss = loss_fn(pred, y)
                
                #If previous model is available, calculate distillation loss
                soft_loss = 0.0
                if previous_model is not None:
                    for old_task_id in range(t):
                        with torch.no_grad():
                            old_pred = previous_model(x, old_task_id)
                        new_prev_pred = baseline_lwf(x, old_task_id)
                        soft_loss += distillation_output_loss(new_prev_pred, old_pred, config["training"]["temperature"]).mean()
                
                #Combine losses
                total_loss = hard_loss + config["training"]["stability"] * soft_loss
                
                #Backpropagate loss
                total_loss.backward()
                optimizer.step()
                
                accuracy_batch = get_batch_acc(pred, y)
                
                wandb.log({'hard_loss': hard_loss.item(), 'soft_loss': float(soft_loss), 
                           'train_loss': total_loss.item(), 'train_accuracy': accuracy_batch,
                           'epoch': e, 'task_id': t, 'batch_idx': batch_idx})
                
                #Track metrics
                epoch_train_losses.append(total_loss.item())
                epoch_train_accs.append(accuracy_batch)
                epoch_soft_losses.append(soft_loss.item() if isinstance(soft_loss, torch.Tensor) else soft_loss)
                metrics['steps_trained'] += 1
                
                if config["logging"]["show_progress"]:
                    progress_bar.set_description(f"Task {t}, Epoch {e}, Loss: {total_loss.item():.4f}, Acc: {epoch_train_accs[-1]:.4f}")
                    
            #Evaluate model on current task's validation set after each epoch
            avg_val_loss, avg_val_acc, time = evaluate_model_timed(multitask_model=baseline_lwf,
                                                        val_loader=val_loader,  
                                                        loss_fn=loss_fn,
                                                        device = device
                                                        )
          
            wandb.log({'val_loss': avg_val_loss, 'val_acc': avg_val_acc, 'epoch': e, 'task_id': t, 'time': time})
            
            #Update metrics
            metrics['epoch_steps'].append(metrics['steps_trained'])
            metrics['train_losses'].extend(epoch_train_losses)
            metrics['train_accs'].extend(epoch_train_accs)
            metrics['soft_losses'].extend(epoch_soft_losses)
            metrics['val_losses'].append(avg_val_loss)
            metrics['val_accs'].append(avg_val_acc)
            
            if config["logging"]["show_progress"]:
                logger.log((f'E{e} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
                                    f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%} in {time:.2f}s'))
                
            if avg_val_acc > metrics['best_val_acc']:
                metrics['best_val_acc'] = avg_val_acc
                logger.log(f"New best validation accuracy: {avg_val_acc:.4f}")
                
        #For cool plots
        metrics['CL_timesteps'].append(metrics['steps_trained'])
        
        #If plotting is enabled, plot training curves
        if config["logging"]["plot_training"] and len(metrics['val_losses']) > 0:
            training_plot(metrics, show_timesteps=True)
            
        #If verbose, print evaluation results
        if config["logging"]["verbose"]:
            logger.log(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
            logger.log(f"Epoch {e} completed in {time:.2f}s")   
        metrics['best_val_acc'] = 0.0
            
        #Evaluate the model on all previous tasks
        metrics_test = test_evaluate_metrics(
                            multitask_model=baseline_lwf,
                            selected_test_sets=data['task_test_sets'][:t+1],
                            task_test_sets=data['task_test_sets'],
                            model_name=f'LwF at t={t}',
                            prev_accs=prev_test_accs,
                            verbose=True,
                            task_metadata=data['task_metadata'],
                            device=device
                            )
            
        wandb.log({**metrics_test, 'task_id': t})
        prev_test_accs.append(metrics_test['task_test_accs'])
        
        # Store the model for the next task
        previous_model = deepcopy(baseline_lwf)
            
    #Log final metrics
    logger.log(f"Task {t} completed!")
    logger.log(f'final metrics: {metrics_test}')
    wandb.summary.update(metrics_test)
        
