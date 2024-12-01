# set up the environment and install any missing packages:
#!pip install torch torchvision numpy scipy matplotlib pandas pillow tqdm MLclf

# PyTorch for building and training neural networks
import torch
from torch import nn, utils
import torch.nn.functional as F

# DataLoader for creating training and validation dataloaders
from torch.utils.data import DataLoader

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
from utils import inspect_batch, test_evaluate, training_plot, setup_dataset_prototype, inspect_task, distillation_output_loss, evaluate_model, get_batch_acc, logger, evaluate_model_prototypes

# Import the HyperCMTL model architecture
from hypernetwork import HyperCMTL, HyperCMTL_prototype, HyperCMTL_prototype_attention

# Import the wandb library for logging metrics and visualizations
import wandb

### Learning without Forgetting:
from copy import deepcopy # Deepcopy for copying models

# time and logging for logging training progress
import time
import logging
import pdb

import random
from torch.utils.data import Sampler

from collections import defaultdict
import random
from torch.utils.data import Sampler

from dataset import Dataset_Prototypes
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# class MinimumClassSampler(Sampler):
#     def __init__(self, dataset, num_classes, batch_size):
#         self.class_indices = self.group_indices_by_class(dataset)
#         # print(self.class_indices)
#         self.num_classes = num_classes
#         self.batch_size = batch_size

#     def group_indices_by_class(self, dataset):
#         class_indices = defaultdict(list)
#         for idx, (_, label, _) in enumerate(dataset):  # Assuming dataset returns (x, y, task_id)
#             class_indices[label.item()].append(idx)
#         return class_indices

#     def __iter__(self):
#         while True:
#             yield self.generate_batch()

#     def generate_batch(self):
#         # Ensure at least one example per class
#         batch = [random.choice(self.class_indices[c]) for c in range(self.num_classes)]

#         # Fill the rest of the batch randomly
#         all_indices = sum(self.class_indices.values(), [])
#         remaining = self.batch_size - self.num_classes
#         batch += random.sample(all_indices, remaining)

#         random.shuffle(batch)  # Shuffle for randomness
#         return batch

#     def __len__(self):
#         return min(len(v) for v in self.class_indices.values())



torch.manual_seed(0)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### dataset hyperparameters:
VAL_FRAC = 0.1
TEST_FRAC = 0.1
BATCH_SIZE = 64
dataset = "Split-CIFAR100" # "Split-MNIST" or "Split-CIFAR100" or "TinyImageNet"
NUM_TASKS = 5 if dataset == 'Split-MNIST' else 10

### training hyperparameters:
EPOCHS_PER_TIMESTEP = 7
lr     = 1e-4  # initial learning rate
l2_reg = 1e-6  # L2 weight decay term (0 means no regularisation)
temperature = 2.0  # temperature scaling factor for distillation loss
stability = 5 #`stability` term to balance this soft loss with the usual hard label loss for the current classification task.

os.makedirs('results', exist_ok=True)
# num = str(len(os.listdir('results/'))).zfill(3)
num = time.strftime("%m%d-%H%M%S")
results_dir = 'results/' + num + '-HyperCMTL'
os.makedirs(results_dir, exist_ok=True)

logger = logger(results_dir)

# Log initial information
logger.log('Starting training...')
logger.log(f'Training hyperparameters: EPOCHS_PER_TIMESTEP={EPOCHS_PER_TIMESTEP}, lr={lr}, l2_reg={l2_reg}, temperature={temperature}, stability={stability}')
logger.log(f'Training on device: {device}')

### Define preprocessing transform and load a batch to inspect it:
data = setup_dataset_prototype(dataset, data_dir='./data', num_tasks=NUM_TASKS, val_frac=VAL_FRAC, test_frac=TEST_FRAC, batch_size=BATCH_SIZE)

timestep_tasks = data['timestep_tasks']
final_test_loader = data['final_test_loader']
task_metadata = data['task_metadata']
task_test_sets = data['task_test_sets']
images_per_class = data['images_per_class']
classes_per_task = data['classes_per_task']

train_split = 6
val_split = 2
test_split = 2

images_train = {k:[im for i, im in enumerate(images_per_class[k]['images']) if i < train_split] for k in images_per_class.keys() }
images_val = {k:[im for i, im in enumerate(images_per_class[k]['images']) if i >= train_split and i < train_split + val_split] for k in images_per_class.keys() }
images_test = {k:[im for i, im in enumerate(images_per_class[k]['images']) if i >= train_split + val_split] for k in images_per_class.keys() }

train_prototypes = Dataset_Prototypes(images_train, classes_per_task)
val_prototypes = Dataset_Prototypes(images_val, classes_per_task)
test_prototypes = Dataset_Prototypes(images_test, classes_per_task)

train_prototypes_dataloader = DataLoader(train_prototypes, batch_size=1, shuffle=True)
val_prototypes_dataloader = DataLoader(val_prototypes, batch_size=1, shuffle=True)
test_prototypes_dataloader = DataLoader(test_prototypes, batch_size=1, shuffle=True)

# More complex model configuration
backbone = 'resnet50'                  # ResNet50 backbone. others: ['mobilenetv2', 'efficientnetb0', 'vit'] #vit not yet working
task_head_projection_size = 256          # Even larger hidden layer in task head
hyper_hidden_features = 256             # Larger hypernetwork hidden layer size
hyper_hidden_layers = 4                 # Deeper hypernetwork

# Initialize the model with the new configurations
model = HyperCMTL_prototype_attention(
    num_instances=len(task_metadata),
    backbone=backbone,
    task_head_projection_size=task_head_projection_size,
    task_head_num_classes=len(task_metadata[0]),
    hyper_hidden_features=hyper_hidden_features,
    hyper_hidden_layers=hyper_hidden_layers,
    device=device,
    std=0.01
).to(device)

# Log the model architecture and configuration
logger.log(f'Model architecture: {model}')

logger.log(f"Model initialized with backbone_config={backbone}, task_head_projection_size={task_head_projection_size}, hyper_hidden_features={hyper_hidden_features}, hyper_hidden_layers={hyper_hidden_layers}")

# Initialize the previous model
previous_model = None

# Initialize optimizer and loss function:
opt = torch.optim.AdamW(model.get_optimizer_list())
loss_fn = nn.CrossEntropyLoss()

### metrics and plotting:
plot_training = True   # show training plots after each timestep
show_progress = True   # show progress bars and end-of-epoch metrics
verbose       = True   # output extra info to console

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

print("Starting training")

with wandb.init(project='HyperCMTL', name=f'HyperCMTL-{dataset}-{backbone}') as run:

    inputs_hyper_x_task = {}
    # outer loop over each task, in sequence
    for (t, (task_train, task_val)), prototypes in zip(timestep_tasks.items(), train_prototypes_dataloader):
        prototypes = prototypes.to(device)[0]
        logger.log(f"Training on task id: {t}  (classification between: {task_metadata[t]})")

        #if verbose:
            #inspect_task(task_train=task_train, task_metadata=task_metadata)

        # build train and validation loaders for the current task:
        train_loader, val_loader = [
            DataLoader(data, BATCH_SIZE)
            for data in (task_train, task_val)
        ]

        # inner loop over the current task:
        for e in range(EPOCHS_PER_TIMESTEP):
            epoch_train_losses, epoch_train_accs = [], []
            epoch_soft_losses = []

            progress_bar = tqdm(train_loader, ncols=100) if show_progress else train_loader
            num_batches = len(train_loader)
            for batch_idx, batch in enumerate(progress_bar):
                x, y, task_ids = batch
                x, y = x.to(device), y.to(device)
                task_id = task_ids[0]
                
                if task_id not in inputs_hyper_x_task:
                    inputs_hyper_x_task[task_id.item()] = prototypes
                
                # get the predictions from the model
                pred = model(x, prototypes).squeeze(0)
                # logger.log('pred shape', pred.shape, 'y shape', y.shape)
                # input_hypernet_tensor = torch.tensor(input_hypernet, device=device)
                # y_no_hyper = y[~torch.isin(torch.arange(y.size(0), device=device), input_hypernet_tensor)]
                    
                hard_loss = loss_fn(pred, y)

                #if previous model exists, calculate distillation loss
                soft_loss = torch.tensor(0.0).to(device)
                if previous_model is not None:
                    for old_task_id in range(t):
                        with torch.no_grad():
                            # print(inputs_hyper_x_task[old_task_id].shape, x.shape)
                            # x_plus_samples = torch.concat([inputs_hyper_x_task[old_task_id].unsqueeze(1), x], dim=0)
                            # print(inputs_hyper_x_task)
                            old_pred = previous_model(x, inputs_hyper_x_task[old_task_id])
                        new_prev_pred = model(x, inputs_hyper_x_task[old_task_id])
                        soft_loss += distillation_output_loss(new_prev_pred, old_pred, temperature).mean().to(device)
                    
                total_loss = hard_loss + stability * soft_loss
                
                # zero the gradients
                opt.zero_grad()
                total_loss.backward()
                opt.step()

                accuracy_batch = get_batch_acc(pred, y)
                
                wandb.log({'hard_loss': hard_loss.item(), 'soft_loss': soft_loss.item(), 'train_loss': total_loss.item(), 'epoch': e, 'task_id': t, 'batch_idx': batch_idx, 'train_accuracy': accuracy_batch})

                # track loss and accuracy:
                epoch_train_losses.append(hard_loss.item())
                epoch_train_accs.append(accuracy_batch)
                epoch_soft_losses.append(soft_loss.item() if isinstance(soft_loss, torch.Tensor) else soft_loss)
                metrics['steps_trained'] += 1

                if show_progress:
                    # show loss/acc of this batch in progress bar:
                    progress_bar.set_description((f'E{e} batch loss:{hard_loss:.2f}, batch acc:{epoch_train_accs[-1]:>5.1%}'))

            # print("Total datasets", len(val_prototypes_dataloader.dataset))
            prototypes_val = val_prototypes_dataloader.dataset[t].to(device)
            # evaluate after each epoch on the current task's validation set:
            avg_val_loss, avg_val_acc = evaluate_model_prototypes(model, val_loader, loss_fn, prototypes = prototypes_val, device = device)

            wandb.log({'val_loss': avg_val_loss, 'val_accuracy': avg_val_acc, 'epoch': e, 'task_id': t})

            ### update metrics:
            metrics['epoch_steps'].append(metrics['steps_trained'])
            metrics['train_losses'].extend(epoch_train_losses)
            metrics['train_accs'].extend(epoch_train_accs)
            metrics['val_losses'].append(avg_val_loss)
            metrics['val_accs'].append(avg_val_acc)
            metrics['soft_losses'].extend(epoch_soft_losses)

            if show_progress:
                # log end-of-epoch stats:
                logger.log((f'E{e} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
                                    f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%}'))

            if avg_val_acc > metrics['best_val_acc']:
                metrics['best_val_acc'] = avg_val_acc

        # this one is important for nice plots:
        metrics['CL_timesteps'].append(metrics['steps_trained'])

        # plot training curves only if validation losses exist
        if plot_training and len(metrics['val_losses']) > 0:
            training_plot(metrics, show_timesteps=True, results_dir = results_dir + f'/training-t{t}.png')

        if verbose:
            logger.log(f'Best validation accuracy: {metrics["best_val_acc"]:.2%}\n')
        metrics['best_val_acc'] = 0.0
        
        # evaluate on all tasks:
        test_accs = test_evaluate(
                            multitask_model=model, 
                            selected_test_sets=task_test_sets[:t+1],  
                            task_test_sets=task_test_sets, 
                            prev_accs = prev_test_accs,
                            show_taskwise_accuracy=True, 
                            baseline_taskwise_accs = None, 
                            model_name= 'HyperCMTL + LwF', 
                            verbose=True, 
                            batch_size=BATCH_SIZE,
                            results_dir=results_dir,
                            task_id=t,
                            task_metadata=task_metadata,
                            using_prototypes=True,
                            prototypes_dataloader=test_prototypes_dataloader
                            )
        
        prev_test_accs.append(test_accs)

        #store the current model as the previous model
        previous_model = model.deepcopy(device = device)
        #torch.cuda.empty_cache()

        final_avg_test_acc = np.mean(test_accs)
        logger.log(f'Final average test accuracy: {final_avg_test_acc:.2%}')
        wandb.log({'val_accuracy': final_avg_test_acc, 'epoch': e, 'task_id': t})
        wandb.summary['final_avg_test_acc'] = final_avg_test_acc