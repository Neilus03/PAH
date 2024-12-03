# set up the environment and install any missing packages:
#!pip install torch torchvision numpy scipy matplotlib pandas pillow tqdm MLclf

# PyTorch for building and training neural networks
import torch
from torch import nn, utils
import torch.nn.functional as F

# DataLoader for creating training and validation dataloaders
from torch.utils.data import DataLoader, BatchSampler

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
from utils import inspect_batch, test_evaluate, training_plot, setup_dataset, inspect_task, distillation_output_loss, evaluate_model, evaluate_model_prototypes, get_batch_acc, logger

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


### as before, define a classification head that we can attach to the backbone:
class TaskHead(nn.Module):
    def __init__(self, input_size: int, # number of features in the backbone's output
                 projection_size: int,  # number of neurons in the hidden layer
                 num_classes: int,      # number of output neurons
                 dropout: float=0.,     # optional dropout rate to apply
                 device='cuda'):
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
        # assume x is already unactivated feature logits,
        # e.g. from resnet backbone
        x = self.projection(self.relu(self.dropout(x)))
        x = self.classifier(self.relu(self.dropout(x)))

        return x

class MultitaskModel(nn.Module):
    def __init__(self, backbone: nn.Module,
                 device="cuda"):
        super().__init__()

        self.backbone = backbone

        # a dict mapping task IDs to the classification heads for those tasks:
        self.task_heads = nn.ModuleDict()
        # we must use a nn.ModuleDict instead of a base python dict,
        # to ensure that the modules inside are properly registered in self.parameters() etc.

        self.relu = nn.ReLU()
        self.device = device
        self.to(device)

    def forward(self,
                x: torch.Tensor,
                task_id: int):

        task_id = str(int(task_id))
        # nn.ModuleDict requires string keys for some reason,
        # so we have to be sure to cast the task_id from tensor(2) to 2 to '2'

        assert task_id in self.task_heads, f"no head exists for task id {task_id}"

        # select which classifier head to use:
        chosen_head = self.task_heads[task_id]

        # activated features from backbone:
        x = self.relu(self.backbone(x))
        # task-specific prediction:
        x = chosen_head(x)

        return x

    def add_task(self,
                 task_id: int,
                 head: nn.Module):
        """accepts an integer task_id and a classification head
        associated to that task.
        adds the head to this baseline_lwf_model's collection of task heads."""
        self.task_heads[str(task_id)] = head

    @property
    def num_task_heads(self):
        return len(self.task_heads)
    
    
### and a baseline_lwf_model that contains a backbone plus multiple class heads,
### and performs task-ID routing at runtime, allowing it to perform any learned task:
class MultitaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, 
                 device='cuda'):
        super().__init__()

        self.backbone = backbone

        # a dict mapping task IDs to the classification heads for those tasks:
        self.task_heads = nn.ModuleDict()        
        # we must use a nn.ModuleDict instead of a base python dict,
        # to ensure that the modules inside are properly registered in self.parameters() etc.

        self.relu = nn.ReLU()
        self.device = device
        self.to(device)

    def forward(self, 
                x: torch.Tensor, 
                task_id: int):
        
        task_id = str(int(task_id))
        # nn.ModuleDict requires string keys for some reason,
        # so we have to be sure to cast the task_id from tensor(2) to 2 to '2'
        
        assert task_id in self.task_heads, f"no head exists for task id {task_id}"
        
        # select which classifier head to use:
        chosen_head = self.task_heads[task_id]

        # activated features from backbone:
        x = self.relu(self.backbone(x))
        # task-specific prediction:
        x = chosen_head(x)

        return x

    def add_task(self, 
                 task_id: int, 
                 head: nn.Module):
        """accepts an integer task_id and a classification head
        associated to that task.
        adds the head to this baseline_lwf_model's collection of task heads."""
        self.task_heads[str(task_id)] = head

    @property
    def num_task_heads(self):
        return len(self.task_heads)

torch.manual_seed(0)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### dataset hyperparameters:
VAL_FRAC = 0.1
TEST_FRAC = 0.1
BATCH_SIZE = 128
dataset = "TinyImageNet" # "Split-MNIST" or "Split-CIFAR100" or "TinyImageNet"
NUM_TASKS = 5 if dataset == 'Split-MNIST' else 10

### training hyperparameters:
EPOCHS_PER_TIMESTEP = 15
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
data = setup_dataset(dataset, data_dir='./data', num_tasks=NUM_TASKS, val_frac=VAL_FRAC, test_frac=TEST_FRAC, batch_size=BATCH_SIZE)

timestep_tasks = data['timestep_tasks']
timestep_task_classes = data['timestep_task_classes']
final_test_loader = data['final_test_loader']
task_metadata = data['task_metadata']
task_test_sets = data['task_test_sets']
images_per_class = data['images_per_class']

# More complex model configuration
backbone_name = 'resnet50'                  # ResNet50 backbone. others: ['mobilenetv2', 'efficientnetb0', 'vit'] #vit not yet working
task_head_projection_size = 256          # Even larger hidden layer in task head
hyper_hidden_features = 256             # Larger hypernetwork hidden layer size
hyper_hidden_layers = 4                 # Deeper hypernetwork

patience = 5  # Number of epochs to wait for improvement
best_val_acc = 0.0
patience_counter = 0

backbone = models.resnet50(pretrained=True)
backbone.num_features = backbone.fc.in_features

baseline_lwf_model = MultitaskModel(backbone.to(device))


# Log the model architecture and configuration
logger.log(f'Model architecture: {baseline_lwf_model}')

logger.log(f"Model initialized with backbone_config={backbone}, task_head_projection_size={task_head_projection_size}, hyper_hidden_features={hyper_hidden_features}, hyper_hidden_layers={hyper_hidden_layers}")

# Initialize the previous model
previous_model = None

# Initialize optimizer and loss function:
opt = torch.optim.Adam(baseline_lwf_model.parameters(), lr=lr, weight_decay=l2_reg)
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


with wandb.init(project='LwF_Baseline', name=f'LwF_Baseline-{dataset}-{backbone_name}') as run:
    wandb.watch(baseline_lwf_model, log='all', log_freq=100)

    # outer loop over each task, in sequence
    for t, (task_train, task_val) in timestep_tasks.items():
        task_train.num_classes = len(timestep_task_classes[t])
        print(f'task_train.num_classes: {task_train.num_classes}')
        logger.log(f"Training on task id: {t}  (classification between: {task_metadata[t]})")
        if t not in baseline_lwf_model.task_heads:
            task_head = TaskHead(input_size=1000, projection_size=64, num_classes=task_train.num_classes).to(device)
            baseline_lwf_model.add_task(t, task_head)

    
        # build train and validation loaders for the current task:
        train_loader, val_loader = [utils.data.DataLoader(data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
                                        for data in (task_train, task_val)]
    

        # inner loop over the current task:
        for e in range(EPOCHS_PER_TIMESTEP):
            epoch_train_losses, epoch_train_accs = [], []
            epoch_soft_losses = []
            
            progress_bar = tqdm(train_loader, ncols=100) if show_progress else train_loader
            num_batches = len(train_loader)
            for batch_idx, batch in enumerate(progress_bar):
                #Get data from batch
                x, y, task_ids = batch
                x, y = x.to(device), y.to(device)
                task_id = task_ids[0]
                
                # zero the gradients
                opt.zero_grad()

                # get the predictions from the model
                pred = baseline_lwf_model(x, task_id)
                hard_loss = loss_fn(pred, y)
                
                #if previous model exists, calculate distillation loss
                soft_loss = 0.0
                if previous_model is not None:
                    for old_task_id in range(t):
                        with torch.no_grad():
                            old_pred = previous_model(x, old_task_id)
                        new_prev_pred = baseline_lwf_model(x, old_task_id)
                        soft_loss += distillation_output_loss(new_prev_pred, old_pred, temperature).mean()
                
                #add the distillation loss to the total loss
                total_loss = hard_loss + stability * soft_loss
                
                #backpropagate the loss
                total_loss.backward()
                opt.step()
                
                # track loss and accuracy:
                epoch_train_losses.append(hard_loss.item())
                epoch_train_accs.append(get_batch_acc(pred, y))
                epoch_soft_losses.append(soft_loss.item() if isinstance(soft_loss, torch.Tensor) else soft_loss)
                metrics['steps_trained'] += 1
                
                if show_progress:
                    # show loss/acc of this batch in progress bar:
                    progress_bar.set_description((f'E{e} batch loss:{hard_loss:.2f}, batch acc:{epoch_train_accs[-1]:>5.1%}'))

            # evaluate after each epoch on the current task's validation set:
            avg_val_loss, avg_val_acc = evaluate_model(baseline_lwf_model, val_loader, loss_fn)
            
            ### update metrics:
            metrics['epoch_steps'].append(metrics['steps_trained'])
            metrics['train_losses'].extend(epoch_train_losses)
            metrics['train_accs'].extend(epoch_train_accs)
            metrics['val_losses'].append(avg_val_loss)
            metrics['val_accs'].append(avg_val_acc)
            metrics['soft_losses'].extend(epoch_soft_losses)
            
            if show_progress:
                # print end-of-epoch stats:
                print((f'E{e} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
                                    f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%}'))

            if avg_val_acc > metrics['best_val_acc']:
                metrics['best_val_acc'] = avg_val_acc
            
        # this one is important for nice plots:
        metrics['CL_timesteps'].append(metrics['steps_trained']) 

        # plot training curves only if validation losses exist
        if plot_training and len(metrics['val_losses']) > 0:
            training_plot(metrics, show_timesteps=True)

        if verbose:
            print(f'Best validation accuracy: {metrics["best_val_acc"]:.2%}\n')
        metrics['best_val_acc'] = 0.0   

        # evaluate on all tasks:
        test_accs = test_evaluate(baseline_lwf_model, task_test_sets[:t+1],
                                  task_test_sets=task_test_sets, 
                                model_name=f'LwF at t={t}', 
                                prev_accs = prev_test_accs,
                                #baseline_taskwise_accs = baseline_taskwise_test_accs, 
                                verbose=True,
                                task_metadata=task_metadata)
        prev_test_accs.append(test_accs)
        
        #store the current baseline_lwf_model as the previous baseline_lwf_model
        previous_model = deepcopy(baseline_lwf_model)

    final_avg_test_acc = np.mean(test_accs)
    print(f'Final average test accuracy: {final_avg_test_acc:.2%}')
   