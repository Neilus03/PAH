# set up the environment and install any missing packages:
#!pip install torch torchvision numpy scipy matplotlib pandas pillow tqdm MLclf

# PyTorch for building and training neural networks
import PIL.Image
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
from PIL import Image, ImageDraw
import PIL

# TQDM for progress bars
from tqdm import tqdm

# OS for operating system operations
import os

# Functions from utils to help with training and evaluation
from utils import inspect_batch, test_evaluate, training_plot, setup_dataset, inspect_task, distillation_output_loss, evaluate_model, get_batch_acc, logger, evaluate_model_2d, test_evaluate_2d

# Import the HyperCMTL_seq model architecture
from networks.hypernetwork import HyperCMTL_seq, HyperCMTL_seq_simple_2d

# Import the wandb library for logging metrics and visualizations
import wandb

### Learning without Forgetting:
from copy import deepcopy # Deepcopy for copying models

# time and logging for logging training progress
import time
import logging
import pdb

torch.manual_seed(0)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### dataset hyperparameters:
VAL_FRAC = 0.1
TEST_FRAC = 0.1
BATCH_SIZE = 128
dataset = "Split-CIFAR100" # "Split-MNIST" or "Split-CIFAR100" or "TinyImageNet"
NUM_TASKS = 5 if dataset == 'Split-MNIST' else 10

### training hyperparameters:
EPOCHS_PER_TIMESTEP = 15
lr     = 1e-4  # initial learning rate
l2_reg = 1e-6  # L2 weight decay term (0 means no regularisation)
temperature = 2.0  # temperature scaling factor for distillation loss
stability = 3 #`stability` term to balance this soft loss with the usual hard label loss for the current classification task.
weight_hard_loss_prototypes = 0.2
weight_soft_loss_prototypes = 0.05

os.makedirs('results', exist_ok=True)
# num = str(len(os.listdir('results/'))).zfill(3)
num = time.strftime("%m%d-%H%M%S")
results_dir = 'results/' + num + '-HyperCMTL_seq'
os.makedirs(results_dir, exist_ok=True)

logger = logger(results_dir)

# Log initial information
logger.log('Starting training...')
logger.log(f'Training hyperparameters: EPOCHS_PER_TIMESTEP={EPOCHS_PER_TIMESTEP}, lr={lr}, l2_reg={l2_reg}, temperature={temperature}, stability={stability}')
logger.log(f'Training on device: {device}')

### Define preprocessing transform and load a batch to inspect it:
data = setup_dataset(dataset, data_dir='./data', num_tasks=NUM_TASKS, val_frac=VAL_FRAC, test_frac=TEST_FRAC, batch_size=BATCH_SIZE)

timestep_tasks = data['timestep_tasks']
final_test_loader = data['final_test_loader']
task_metadata = data['task_metadata']
task_test_sets = data['task_test_sets']

# More complex model configuration
backbone = 'resnet50'                  # ResNet50 backbone. others: ['mobilenetv2', 'efficientnetb0', 'vit'] #vit not yet working
task_head_projection_size = 512          # Even larger hidden layer in task head
hyper_hidden_features = 256             # Larger hypernetwork hidden layer size
hyper_hidden_layers = 4                 # Deeper hypernetwork

# Initialize the model with the new configurations
model = HyperCMTL_seq_simple_2d(
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
prev_test_accs_prot = []

print("Starting training")

config = {'EPOCHS_PER_TIMESTEP': EPOCHS_PER_TIMESTEP, 'lr': lr, 
          'l2_reg': l2_reg, 'temperature': temperature, 'stability': stability, 
          'weight_hard_loss_prototypes': weight_hard_loss_prototypes, 
          'weight_soft_loss_prototypes': weight_soft_loss_prototypes, 
          'backbone': backbone, 'color' : 'RGB'}

# with wandb.init(project='HyperCMTL', name=f'HyperCMTL_seq-learned_emb-{dataset}-{backbone}') as run:
    # wandb.config.update(config)
    # wandb.watch(model, log='all', log_freq=100)

    # outer loop over each task, in sequence
    # for t, (task_train, task_val) in timestep_tasks.items():
    # t = 0
    # (task_train, task_val) = timestep_tasks[t]
    # logger.log(f"Training on task id: {t}  (classification between: {task_metadata[t]})")

    # # build train and validation loaders for the current task:
    # train_loader, val_loader = [utils.data.DataLoader(data,
    #                                 batch_size=BATCH_SIZE,
    #                                 shuffle=True)
    #                                 for data in (task_train, task_val)]

    # # inner loop over the current task:
    # for e in range(EPOCHS_PER_TIMESTEP):
    #     epoch_train_losses, epoch_train_accs = [], []
    #     epoch_soft_losses = []

    #     progress_bar = tqdm(train_loader, ncols=100) if show_progress else train_loader
    #     num_batches = len(train_loader)
    #     for batch_idx, batch in enumerate(progress_bar):
    #         #Get data from batch
    #         x, y, task_ids = batch
    #         x, y = x.to(device), y.to(device)
    #         task_id = task_ids[0]

    #         # zero the gradients
    #         opt.zero_grad()

    #         # get the predictions from the model
    #         pred, pred_prototypes = model(x, task_id)
    #         y_prototypes = torch.arange(len(task_metadata[int(task_id)]), device=device, dtype=torch.int64)
            
    #         hard_loss = loss_fn(pred, y)
    #         prototypes_loss = loss_fn(pred_prototypes, y_prototypes)

    #         #if previous model exists, calculate distillation loss
    #         soft_loss = torch.tensor(0.0).to(device)
    #         total_loss = hard_loss 
            
    #         total_loss.backward()
    #         opt.step()

    #         accuracy_batch = get_batch_acc(pred, y)
            
    #         wandb.log({'hard_loss': hard_loss.item(), 'soft_loss': (soft_loss*stability).item(), 
    #                     'train_loss': total_loss.item(), 'prototype_loss': prototypes_loss.item(),
    #                     'epoch': e, 'task_id': t, 'batch_idx': batch_idx, 'train_accuracy': accuracy_batch})

    #         # track loss and accuracy:
    #         epoch_train_losses.append(hard_loss.item())
    #         epoch_train_accs.append(accuracy_batch)
    #         epoch_soft_losses.append(soft_loss.item() if isinstance(soft_loss, torch.Tensor) else soft_loss)
    #         metrics['steps_trained'] += 1

    #         if show_progress:
    #             # show loss/acc of this batch in progress bar:
    #             progress_bar.set_description((f'E{e} batch loss:{hard_loss:.2f}, batch acc:{epoch_train_accs[-1]:>5.1%}'))

    #     # evaluate after each epoch on the current task's validation set:
    #     avg_val_loss, avg_val_acc, avg_val_loss_prot, avg_val_acc_prot = evaluate_model_2d(model, val_loader, loss_fn, device = device, task_metadata = task_metadata, task_id=t, wandb_run = run.id)

    #     wandb.log({'val_loss': avg_val_loss, 'val_accuracy': avg_val_acc, 'epoch': e, 'task_id': t, 'val_prot_loss': avg_val_loss_prot, 'val_prot_accuracy': avg_val_acc_prot})

    #     ### update metrics:
    #     metrics['epoch_steps'].append(metrics['steps_trained'])
    #     metrics['train_losses'].extend(epoch_train_losses)
    #     metrics['train_accs'].extend(epoch_train_accs)
    #     metrics['val_losses'].append(avg_val_loss)
    #     metrics['val_accs'].append(avg_val_acc)
    #     metrics['soft_losses'].extend(epoch_soft_losses)

    #     if show_progress:
    #         # log end-of-epoch stats:
    #         logger.log((f'E{e} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
    #                             f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%}'))

    #     if avg_val_acc > metrics['best_val_acc']:
    #         metrics['best_val_acc'] = avg_val_acc

    #     # this one is important for nice plots:
    #     metrics['CL_timesteps'].append(metrics['steps_trained'])

    #     # plot training curves only if validation losses exist
    #     if plot_training and len(metrics['val_losses']) > 0:
    #         training_plot(metrics, show_timesteps=True, results_dir = results_dir + f'/training-t{t}.png')

    #     if verbose:
    #         logger.log(f'Best validation accuracy: {metrics["best_val_acc"]:.2%}\n')
    #     metrics['best_val_acc'] = 0.0
        
    # # evaluate on all tasks:
    # test_accs, test_accs_prot = test_evaluate_2d(
    #                         multitask_model=model, 
    #                         selected_test_sets=task_test_sets[:t+1],  
    #                         task_test_sets=task_test_sets, 
    #                         prev_accs = prev_test_accs,
    #                         prev_accs_prot = prev_test_accs_prot,
    #                         show_taskwise_accuracy=True, 
    #                         baseline_taskwise_accs = None, 
    #                         model_name= 'HyperCMTL_seq + LwF', 
    #                         verbose=True, 
    #                         batch_size=BATCH_SIZE,
    #                         results_dir=results_dir,
    #                         task_id=t,
    #                         task_metadata=task_metadata,
    #                         wandb_run = run.id
    #                         )
        
    # wandb.log({'mean_test_acc': np.mean(test_accs), 'task_id': t, 'mean_test_acc_prot': np.mean(test_accs_prot)})

    # prev_test_accs.append(test_accs)
    # prev_test_accs_prot.append(test_accs_prot)

    # #store the current model as the previous model
    # previous_model = model.deepcopy()
    # #torch.cuda.empty_cache()
t = 0
for name, param in model.named_parameters():
    if "emb" not in name:
        param.requires_grad = False
    
print("Not frozen parameters:", [param.name for param in model.parameters() if param.requires_grad])
all_losses = []
images = []
for pretrain_epochs in range(1000):
    z = model.hyper_emb(torch.LongTensor([t]).to(model.device))
    prototypes = z.view(model.task_head_num_classes, 1, 20, 20)
    prototypes = prototypes.repeat(1, 3, 1, 1)        
    prototypes = prototypes.clip(0, 1)
    # fig, ax = plt.subplots(prototypes.size(0), 1, figsize=(5, 5*prototypes.size(0)))
    # ax = ax.flatten()
    # for i in range(prototypes.size(0)):
        # ax[i].imshow(prototypes[i].cpu().detach().numpy().transpose(1, 2, 0))
        # ax[i].axis('off')
    # plt.tight_layout()
    image = (prototypes[0].cpu().detach().numpy().transpose(1, 2, 0))
    # put all values below 0 to 0 and above 1 to 1
    print(image.max(), image.min())
    image = PIL.Image.fromarray(image.astype(np.uint8))
    
    pred_y = model(prototypes, t)[0]

    true_y = torch.arange(len(task_metadata[t]), device=device, dtype=torch.int64)
    
    loss_prototypes = loss_fn(pred_y, true_y)
    # put a text in the left top corner with the loss of this prototype
    ImageDraw.Draw(image).text((0, 0), f'Loss: {loss_prototypes.item():.2f}', fill='white')
    
    print(loss_prototypes.item())
    all_losses.append(loss_prototypes.item())

    images.append(image)
    mean_loss = np.mean(all_losses[-50:])
    if mean_loss < 0.001:
        break

    # wandb.log({'extra_loss_prototypes': loss_prototypes.item(), 'epoch': e, 'task_id': t, 'batch_idx': batch_idx, 'inner_epoch': pretrain_epochs})
    opt.zero_grad()
    loss_prototypes.backward()
    opt.step()

    # create a gif from all the images
images[0].save(results_dir + f'/prototypes-t{t}.gif', save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

# final_avg_test_acc = np.mean(test_accs)
# logger.log(f'Final average test accuracy: {final_avg_test_acc:.2%}')
# wandb.log({'val_accuracy': final_avg_test_acc, 'epoch': e, 'task_id': t})
# wandb.summary['final_avg_test_acc'] = final_avg_test_acc