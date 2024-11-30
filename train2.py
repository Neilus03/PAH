import torch
from torch import nn, utils
import torch.nn.functional as F
from torchvision import models, datasets, transforms

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
from utils2 import inspect_batch, test_evaluate, training_plot, inspect_task, distillation_output_loss, evaluate_model, get_batch_acc, logger
from TSR.hypernetwork2 import HyperCMTL2
import wandb
from dataset import build_task_datasets

### Learning without Forgetting:
from copy import deepcopy
import torch
torch.manual_seed(0)

# Assuming HyperCMTL, timestep_tasks, BATCH_SIZE, distillation_output_loss, get_batch_acc, evaluate_model, training_plot, test_evaluate are defined elsewhere in the notebook
import numpy as np
from tqdm import tqdm
import time
import logging

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_FRAC = 0.1
TEST_FRAC = 0.05
BATCH_SIZE = 256

### training hyperparameters:
EPOCHS_PER_TIMESTEP = 5
lr     = 1e-4  # initial learning rate
l2_reg = 1e-6  # L2 weight decay term (0 means no regularisation)
temperature = 2.0  # temperature scaling factor for distillation loss
stability = 1 #`stability` term to balance this soft loss with the usual hard label loss for the current classification task.

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

fmnist = datasets.FashionMNIST(root='data/', download=True)
fmnist.name, fmnist.num_classes = 'Fashion-MNIST', len(fmnist.classes)
logger.log(f'{fmnist.name}: {len(fmnist)} samples')

timestep_task_classes = []
for i, cl in enumerate(fmnist.classes):
    if i == len(fmnist.classes) - 1:
        break
    timestep_task_classes.append([fmnist.classes[i], fmnist.classes[i+1]])
    timestep_task_classes.append([fmnist.classes[i+1], fmnist.classes[i]])
    
timestep_task_classes = {i: timestep_task_classes[i] for i in range(len(timestep_task_classes))}
print(timestep_task_classes)


for i, cl in enumerate(fmnist.classes):
    logger.log(f'{i}: {cl}')
    

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

datasets = build_task_datasets(
    fmnist=fmnist,
    timestep_task_classes=timestep_task_classes,
    preprocess=preprocess,
    VAL_FRAC=0.1,
    TEST_FRAC=0.1,
    BATCH_SIZE=64,
    inspect_task=inspect_task  # Optional
)

timestep_tasks = datasets['timestep_tasks']
final_test_loader = datasets['final_test_loader']
task_test_sets = datasets['task_test_sets']


model = HyperCMTL2(num_instances=4, device=device, std=0.01).to(device)
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

with wandb.init(project='HyperCMTL', name='HyperCMTL') as run:

    # outer loop over each task, in sequence
    inputs_hyper_x_task = {}
    for t, (task_train, task_val) in timestep_tasks.items():
        logger.log(f"Training on task id: {t}  (classification between: {task_train.classes})")

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
                
                idx_sample_class_0 = (y == 0).nonzero()[0].item()
                idx_sample_class_1 = (y == 1).nonzero()[0].item()
                
                input_hypernet = [idx_sample_class_0, idx_sample_class_1]
                if t not in inputs_hyper_x_task:
                    inputs_hyper_x_task[t] = torch.concat([x[idx_sample_class_0], x[idx_sample_class_1]], dim=0)
                
                # zero the gradients
                opt.zero_grad()

                # get the predictions from the model
                pred = model(x, input_hypernet).squeeze(0)
                input_hypernet_tensor = torch.tensor(input_hypernet, device=device)
                y_no_hyper = y[~torch.isin(torch.arange(y.size(0), device=device), input_hypernet_tensor)]
                
                hard_loss = loss_fn(pred, y_no_hyper)

                #if previous model exists, calculate distillation loss
                soft_loss = torch.tensor(0.0).to(device)
                if previous_model is not None:
                    for old_task_id in range(t):
                        with torch.no_grad():
                            # print(inputs_hyper_x_task[old_task_id].shape, x.shape)
                            x_plus_samples = torch.concat([inputs_hyper_x_task[old_task_id].unsqueeze(1), x], dim=0)
                            input_hypernet = [0, 1]
                            old_pred = previous_model(x_plus_samples, input_hypernet)
                        new_prev_pred = model(x_plus_samples, input_hypernet)
                        soft_loss += distillation_output_loss(new_prev_pred, old_pred, temperature).mean().to(device)
                
                total_loss = hard_loss + stability * soft_loss
                
                total_loss.backward()
                opt.step()

                accuracy_batch = get_batch_acc(pred, y_no_hyper)
                
                wandb.log({'hard_loss': hard_loss.item(), 'soft_loss': soft_loss.item(), 'train_loss': total_loss.item(), 'epoch': e, 'task_id': t, 'batch_idx': batch_idx, 'train_accuracy': accuracy_batch})

                # track loss and accuracy:
                epoch_train_losses.append(hard_loss.item())
                epoch_train_accs.append(accuracy_batch)
                epoch_soft_losses.append(soft_loss.item() if isinstance(soft_loss, torch.Tensor) else soft_loss)
                metrics['steps_trained'] += 1

                # if show_progress:
                    # show loss/acc of this batch in progress bar:
                    # progress_bar.set_description((f'E{e} batch loss:{hard_loss:.2f}, batch acc:{epoch_train_accs[-1]:>5.1%}'))

            # evaluate after each epoch on the current task's validation set:
            avg_val_loss, avg_val_acc = evaluate_model(model, val_loader, loss_fn)

            wandb.log({'val_loss': avg_val_loss, 'val_accuracy': avg_val_acc, 'epoch': e, 'task_id': t})

            ### update metrics:
            metrics['epoch_steps'].append(metrics['steps_trained'])
            metrics['train_losses'].extend(epoch_train_losses)
            metrics['train_accs'].extend(epoch_train_accs)
            metrics['val_losses'].append(avg_val_loss)
            metrics['val_accs'].append(avg_val_acc)
            metrics['soft_losses'].extend(epoch_soft_losses)

            # if show_progress:
                # log end-of-epoch stats:
                # logger.log((f'E{e} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
                                    # f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%}'))

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
        test_accs = test_evaluate(model, 
                                task_test_sets[:t+1],
                                task_test_sets,
                                prev_accs = prev_test_accs,
                                model_name=f'LwF at t={t}',
                                show_taskwise_accuracy = True,
                                verbose=True,
                                batch_size=BATCH_SIZE,
                                results_dir = results_dir + f'/evaluation-t{t}.png', 
                                task_id=t)
        
        prev_test_accs.append(test_accs)

        #store the current model as the previous model
        previous_model = model.deepcopy(device = device)

    final_avg_test_acc = np.mean(test_accs)
    logger.log(f'Final average test accuracy: {final_avg_test_acc:.2%}')
    wandb.log({'val_accuracy': final_avg_test_acc, 'epoch': e, 'task_id': t})
    wandb.summary['final_avg_test_acc'] = final_avg_test_acc