import torch
from torch import nn, utils
import torch.nn.functional as F
import wandb 
from copy import deepcopy

from torch.utils.data import DataLoader, BatchSampler, ConcatDataset
from torchvision import models, datasets, transforms

import json
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

from networks.backbones import ResNet50, AlexNet, MobileNetV2, EfficientNetB0, ViT, ResNet18, ReducedResNet18
from networks.networks_baseline import *

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
    'efficientnetb0': EfficientNetB0,
    'vit': ViT,
    'resnet18': ResNet18,
    'reducedresnet18': ReducedResNet18,
    'alexnet': AlexNet
}

backbone_name = config["model"]["backbone"]
backbone = backbone_dict[backbone_name](device=config["misc"]["device"], pretrained=False)
logger.log(f"Using backbone: {backbone_name}")

# Freeze backbone if specified
if config["model"]["frozen_backbone"]:
    for param in backbone.parameters():
        param.requires_grad = False
       

        
# Combine all tasks into a single dataset for joint training
logger.log("Combining all tasks for joint training...")
train_datasets = [train_set for train_set, _ in data["timestep_tasks"].values()]
val_datasets = [val_set for _, val_set in data["timestep_tasks"].values()]
logger.log(f"Number of training samples per task: {[len(ds) for ds in train_datasets]}")
logger.log(f"Number of validation samples: {[len(ds) for ds in val_datasets]}")
joint_train_dataset = ConcatDataset(train_datasets)
joint_val_dataset = ConcatDataset(val_datasets)
logger.log(f"Number of training samples after concatenation: {len(joint_train_dataset)}")
logger.log(f"Number of validation samples after concatenation: {len(joint_val_dataset)}")
joint_train_loader = DataLoader(joint_train_dataset, batch_size=config["dataset"]["BATCH_SIZE"], shuffle=True)
joint_val_loader = DataLoader(joint_val_dataset, batch_size=config["dataset"]["BATCH_SIZE"], shuffle=False)
logger.log("Joint training and validation datasets created")
logger.log("Number of training samples: {}".format(len(joint_train_dataset)))
logger.log("Number of validation samples: {}".format(len(joint_val_dataset)))


# Create joint model
num_classes = sum([len(classes) for classes in data["timestep_task_classes"].values()])
logger.log(f"Number of classes in total: {num_classes}")
joint_model = MultitaskModel_Baseline(backbone, device=config["misc"]["device"])
logger.log("Joint model created")

# Optimizer and loss function
optimizer = setup_optimizer(
    model=joint_model,
    lr=config["training"]["lr"],
    l2_reg=config["training"]["l2_reg"],
    optimizer=config["training"]["optimizer"]
)

loss_fn = nn.CrossEntropyLoss()
logger.log("Optimizer and loss function created")

joint_task_head = TaskHead_simple(
    input_size=backbone.num_features, num_classes=num_classes, device=config["misc"]["device"]
)
logger.log("Task head created")
joint_model.add_task(0, joint_task_head)
optimizer.add_param_group({"params": joint_task_head.parameters()})
logger.log("Task head added to joint model")


#Track metrics for plotting
metrics = { 'train_losses': [],
              'train_accs': [],
              'val_losses': [],
                'val_accs': [],
             'epoch_steps': [], # used for plotting val loss at the correct x-position
            'CL_timesteps': [], # used to draw where each new timestep begins
            'best_val_acc': 0.0,
           'steps_trained': 0,
          }

prev_test_accs = []

# Training loop
logger.log(f"Starting joint training for {config['logging']['name']}...")
with wandb.init(project='HyperCMTL', entity='pilligua2', name=f'{name_run}', config=config, group=config['logging']['group']) as run:
    for epoch in range(config["training"]["epochs_per_timestep"]):
        joint_model.train()
        epoch_train_losses, epoch_train_accs = [], []
        
        progress_bar = tqdm(joint_train_loader,ncols=100,total= len(joint_train_loader), desc=f"Epoch {epoch + 1}/{config['training']['epochs_per_timestep']}") if config["logging"]["show_progress"] else joint_train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = joint_model(x, 0)  # Single task ID for joint training
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            accuracy_batch = get_batch_acc(pred, y)
            
            wandb.log({"train_loss": loss.item(), "train_acc": accuracy_batch, "epoch": epoch, "batch_idx": batch_idx})
            
            epoch_train_losses.append(loss.item())
            epoch_train_accs.append(accuracy_batch)
            metrics['steps_trained'] += 1
            
            if config["logging"]["show_progress"]:
                    progress_bar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {epoch_train_accs[-1]:.4f}")
            
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        logger.log(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        avg_val_loss, avg_val_acc, time = evaluate_model_timed(
                                                multitask_model=joint_model,
                                                val_loader=joint_val_loader,
                                                loss_fn=loss_fn,
                                                device=config["misc"]["device"],
                                                joint_training=True
                                            )
        wandb.log({"val_loss": avg_val_loss, "val_acc": avg_val_acc, "epoch": epoch, "time": time})
        logger.log(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Epoch: {epoch + 1}/{config['training']['epochs_per_timestep']}, Time: {time}")

        #Update metrics
        metrics['epoch_steps'].append(metrics['steps_trained'])
        metrics['train_losses'].extend(epoch_train_losses)
        metrics['train_accs'].extend(epoch_train_accs)
        metrics['val_losses'].append(avg_val_loss)
        metrics['val_accs'].append(avg_val_acc)
        
        if config["logging"]["show_progress"]:
            logger.log((f'E{epoch} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
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
        logger.log(f"Epoch {epoch} completed in {time:.2f}s")   
    metrics['best_val_acc'] = 0.0
    
    
    #DE AQUÍ PARA ABAJO AUN NO ESTÁ ACABADO, ASI no se si va-
    
    
    # Evaluate joint model on individual task test sets
    logger.log("Evaluating joint model on individual tasks...")
    joint_model_accuracies = []
    
    for t, test_set in enumerate(data["task_test_sets"]):
        test_loader = DataLoader(test_set, batch_size=config["dataset"]["BATCH_SIZE"], shuffle=False)
        _, acc, _ = evaluate_model_timed(joint_model, test_loader, device=config["misc"]["device"], joint_training=True)
        
        joint_model_accuracies.append(acc)
        
        wandb.log({f"task {t} accuracy": acc})
        logger.log(f"Task {t} - Accuracy: {acc:.4f}")
        

    # Save joint model accuracies
    joint_model_accuracies_path = os.path.join(results_dir, "joint_model_accuracies.json")
    with open(joint_model_accuracies_path, "w") as f:
        json.dump(joint_model_accuracies, f)
    logger.log(f"Joint model accuracies saved to {joint_model_accuracies_path}")
    wandb.log({"joint_model_accuracies": joint_model_accuracies})