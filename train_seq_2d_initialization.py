# set up the environment and install any missing packages:
#!pip install torch torchvision numpy scipy matplotlib pandas pillow tqdm MLclf

# PyTorch for building and training neural networks
import torch
from torch import nn, utils
import torch.nn.functional as F

# DataLoader for creating training and validation dataloaders
from torch.utils.data import DataLoader

from collections import OrderedDict

# Torchvision for datasets and transformations
from torchvision import models, datasets, transforms

# Numpy for numerical operations
import numpy as np

# Matplotlib for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# Pandas for data manipulation
import pandas as pd

# PIL for image processing
from PIL import Image

# TQDM for progress bars
from tqdm import tqdm

# OS for operating system operations
import os

# Functions from utils to help with training and evaluation
from utils import *

# Import the HyperCMTL_seq model architecture
from networks.hypernetwork import HyperCMTL_seq, HyperCMTL_seq_simple_2d
from networks.backbones import ResNet50, MobileNetV2, EfficientNetB0

# Import the wandb library for logging metrics and visualizations
import wandb

### Learning without Forgetting:
from copy import deepcopy # Deepcopy for copying models

# time and logging for logging training progress
import time
import logging
import pdb
import sys

config = config_load(sys.argv[1])["config"]

device = torch.device(config["misc"]["device"] if torch.cuda.is_available() else "cpu")
seed_everything(config['misc']['seed'])

num = time.strftime("%Y%m%d-%H%M%S")
name_run = f"{config['logging']['name']}-{num}"
results_dir = os.path.join(config["logging"]["results_dir"], name_run)
os.makedirs(results_dir, exist_ok=True)

logger = logger(results_dir)

# Log initial information
logger.log(f"Starting training for {config['logging']['name']}")
logger.log(f"Configuration: {config}")
logger.log(f"Device: {device}")
logger.log(f"Random seed: {config['misc']['seed']}")

### Define preprocessing transform and load a batch to inspect it:
data = setup_dataset_prototype(
    dataset_name=config['dataset']['dataset'], 
    data_dir=config["dataset"]["data_dir"], 
    num_tasks=config['dataset']['NUM_TASKS'], 
    val_frac=config['dataset']['VAL_FRAC'], 
    test_frac=config['dataset']['TEST_FRAC'], 
    batch_size=config['dataset']['BATCH_SIZE']
)

num_tasks = len(data['task_metadata'])
num_classes_per_task = len(data['task_metadata'][0])

backbone_dict = {
    'resnet50': ResNet50,
    'mobilenetv2': MobileNetV2,
    'efficientnetb0': EfficientNetB0
}

logger.log(f"Using backbone: {config['model']['backbone']}")

# Initialize the model with the new configurations
model = HyperCMTL_seq_simple_2d(
    num_tasks=num_tasks,
    num_classes_per_task=num_classes_per_task,
    model_config=config['model'],
    device=device
).to(device)

logger.log(f"Model created!")
logger.log(f"Model initialized with freeze_backbone={config['model']['frozen_backbone']}, config={config['model']}")

prototypes_inititalization = torch.zeros((10, 4000))
fig, ax = plt.subplots(num_tasks, num_classes_per_task, figsize=(num_classes_per_task*3, (num_tasks)*3))
for t in range(num_tasks):
    prototypes = torch.mean(data['task_prototypes'][t][:, :, 6:26, 6:26], dim=1, keepdim=True)
    prototypes_inititalization[t] = prototypes.reshape(-1)

    for i in range(len(data['task_metadata'][t])):
        ax[t][i].imshow(prototypes[i].permute(1, 2, 0), cmap='gray')
        ax[t][i].axis('off')
        ax[t][i].set_title(data['task_metadata'][t][i])

plt.savefig(results_dir + '/prototypes.png')
plt.close()
model.initialize_embeddings(prototypes_inititalization.to(device))

# Initialize the previous model
previous_model = None

# Initialize optimizer and loss function:
loss_fn = nn.CrossEntropyLoss()
tv_loss_fn = TotalVariationLoss()
opt = torch.optim.AdamW(model.get_optimizer_list())

# track metrics for plotting training curves:
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
prev_test_accs_prot = []

logger.log(f"Starting training for {config['logging']['name']}")

with wandb.init(project='HyperCMTL', name=f'{name_run}', config=config) as run:
    wandb.log({"all prototypes": wandb.Image(results_dir + '/prototypes.png')})

    # outer loop over each task, in sequence
    for t, (task_train, task_val) in data['timestep_tasks'].items():
        task_train.num_classes = len(data['timestep_task_classes'][t])
        logger.log(f"Task {t}: {task_train.num_classes} classes\n: {data['task_metadata'][t]}")
        
        # build train and validation loaders for the current task:
        train_loader, val_loader = [utils.data.DataLoader(data,
                                        batch_size=config['dataset']['BATCH_SIZE'],
                                        shuffle=True)
                                        for data in (task_train, task_val)]

        eval_data = evaluate_model_2d(model, val_loader, loss_fn = loss_fn, device = device, task_metadata = data['task_metadata'], task_id=t, wandb_run = run.id)
        avg_val_loss, avg_val_acc, avg_val_loss_prot, avg_val_acc_prot, time = eval_data

        # inner loop over the current task:
        for e in range(config['training']['epochs_per_timestep']):
            epoch_train_losses, epoch_train_accs = [], []
            epoch_soft_losses = []

            progress_bar = tqdm(train_loader, ncols=100) if config["logging"]["show_progress"] else train_loader
            num_batches = len(train_loader)
            
            for batch_idx, batch in enumerate(progress_bar):
                #Get data from batch
                x, y, task_ids = batch
                x, y = x.to(device), y.to(device)
                task_id = task_ids[0]

                # zero the gradients
                opt.zero_grad()

                # get the predictions from the model
                pred, pred_prototypes = model(x, task_id)
                prototypes = model.get_prototypes()
                y_prototypes = torch.arange(len(data['task_metadata'][int(task_id)]), device=device, dtype=torch.int64)
                
                hard_loss = loss_fn(pred, y)
                prototypes_loss = loss_fn(pred_prototypes, y_prototypes) * config['training']['weight_hard_loss_prototypes']
                
                # print(prototypes.shape)
                smoothness_loss = tv_loss_fn(prototypes) * config['training']['weight_smoothness_loss']

                #if previous model exists, calculate distillation loss
                soft_loss = torch.tensor(0.0).to(device)
                if previous_model is not None:
                    for old_task_id in range(t):
                        with torch.no_grad():
                    
                            old_pred, old_pred_prot = previous_model(x, old_task_id)
                        new_prev_pred, new_prev_pred_prot = model(x, old_task_id)
                        soft_loss += distillation_output_loss(new_prev_pred, old_pred, config['training']['temperature']).mean().to(device)
                        soft_loss += distillation_output_loss(new_prev_pred_prot, old_pred_prot, config['training']['temperature']).mean().to(device) * config['training']['weight_soft_loss_prototypes']

                soft_loss *= config['training']['stability']
                total_loss = hard_loss + soft_loss + prototypes_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                accuracy_batch = get_batch_acc(pred, y)
                
                wandb.log({'hard_loss': hard_loss.item(), 'soft_loss': soft_loss.item(), 
                           'train_loss': total_loss.item(), 'prototype_loss': prototypes_loss.item(),
                           'smoothness_loss': smoothness_loss.item(),
                           'epoch': e, 'task_id': t, 'batch_idx': batch_idx, 'train_accuracy': accuracy_batch})

                # track loss and accuracy:
                epoch_train_losses.append(hard_loss.item())
                epoch_train_accs.append(accuracy_batch)
                epoch_soft_losses.append(soft_loss.item() if isinstance(soft_loss, torch.Tensor) else soft_loss)
                metrics['steps_trained'] += 1

                if config["logging"]["show_progress"]:
                    progress_bar.set_description((f'E{e} batch loss:{hard_loss:.2f}, batch acc:{accuracy_batch:>5.1%}'))

            # evaluate after each epoch on the current task's validation set:
            eval_data = evaluate_model_2d(model, val_loader, loss_fn = loss_fn, device = device, task_metadata = data['task_metadata'], task_id=t, wandb_run = run.id)
            avg_val_loss, avg_val_acc, avg_val_loss_prot, avg_val_acc_prot, time = eval_data

            wandb.log({'val_loss': avg_val_loss, 'val_accuracy': avg_val_acc, 
                       'epoch': e, 'task_id': t, 'val_prot_loss': avg_val_loss_prot, 
                       'val_prot_accuracy': avg_val_acc_prot, 'time': time})

            ### update metrics:
            metrics['epoch_steps'].append(metrics['steps_trained'])
            metrics['train_losses'].extend(epoch_train_losses)
            metrics['train_accs'].extend(epoch_train_accs)
            metrics['val_losses'].append(avg_val_loss)
            metrics['val_accs'].append(avg_val_acc)
            metrics['soft_losses'].extend(epoch_soft_losses)

            if config["logging"]["show_progress"]:
                # log end-of-epoch stats:
                logger.log((f'E{e} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
                                    f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%}'))

            if avg_val_acc > metrics['best_val_acc']:
                metrics['best_val_acc'] = avg_val_acc

        # this one is important for nice plots:
        metrics['CL_timesteps'].append(metrics['steps_trained'])

        # plot training curves only if validation losses exist
        if config["logging"]["plot_training"] and len(metrics['val_losses']) > 0:
            training_plot(metrics, show_timesteps=True, results_dir = results_dir + f'/training-t{t}.png')

        if config["logging"]["verbose"]:
            logger.log(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
            logger.log(f"Epoch {e} completed in {time:.2f}s")   
        metrics['best_val_acc'] = 0.0
        
        # evaluate on all tasks:
        metrics_test = test_evaluate_2d(
                        multitask_model=model, 
                        selected_test_sets=data['task_test_sets'][:t+1],  
                        task_test_sets=data['task_test_sets'], 
                        prev_accs = prev_test_accs,
                        prev_accs_prot = prev_test_accs_prot,
                        show_taskwise_accuracy=True, 
                        baseline_taskwise_accs = None, 
                        model_name= 'HyperCMTL_seq + LwF', 
                        verbose=True, 
                        batch_size=config['dataset']['BATCH_SIZE'],
                        results_dir=results_dir,
                        task_id=t,
                        task_metadata=data['task_metadata'],
                        wandb_run = run.id,
                        device=device
                        )
        
        wandb.log({**metrics_test, 'task_id': t})
        
        prev_test_accs.append(metrics_test['task_test_accs'])
        prev_test_accs_prot.append(metrics_test['task_test_accs_prot'])

        #store the current model as the previous model
        previous_model = model.deepcopy()


    #Log final metrics
    logger.log(f"Task {t} completed!")
    logger.log(f'final metrics: {metrics_test}')
    wandb.summary.update(metrics_test)