import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils as utils
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset, Subset
from torchvision import datasets, transforms
#from tinyimagenet import TinyImageNet
import pandas as pd
import matplotlib as mpl
import wandb
import io
from PIL import Image
import pdb
from tqdm import tqdm
import pdb
import random
import torch
import random
from torch.utils.data import Sampler
# from configs.config import config
import time
import os
from easydict import EasyDict 


def inspect_batch(images, labels=None, predictions=None, class_names=None, title=None,
                  center_title=True, max_to_show=16, num_cols=4, scale=1):
    """
    Plots a batch of images in a grid for manual inspection. Optionally displays ground truth 
    labels and/or model predictions.

    Args:
        images (torch.Tensor or list): Batch of images as a torch tensor or list of tensors. Each
            image tensor should have shape (C, H, W).
        labels (list, optional): Ground truth labels for the images. Defaults to None.
        predictions (torch.Tensor or list, optional): Model predictions for the images. Defaults to None.
        class_names (list or dict, optional): Class names for labels and predictions. Can be a list 
            (index-to-name mapping) or a dict (name-to-index mapping). Defaults to None.
        title (str, optional): Title for the plot. Defaults to None.
        center_title (bool, optional): Whether to center the title. Defaults to True.
        max_to_show (int, optional): Maximum number of images to show. Defaults to 16.
        num_cols (int, optional): Number of columns in the grid. Defaults to 4.
        scale (float, optional): Scale factor for figure size. Defaults to 1.

    Returns:
        None: Displays the grid of images using matplotlib.
    """

    # Ensure max_to_show does not exceed the number of available images
    max_to_show = min(max_to_show, len(images))
    num_rows = int(np.ceil(max_to_show / num_cols))

    # Calculate additional figure height for captions if labels or predictions are provided
    extra_height = 0.2 if (labels is not None or predictions is not None) else 0

    # Determine figure dimensions
    fig_width = 2 * scale * num_cols
    fig_height = (2 + extra_height) * scale * num_rows + (0.3 if title is not None else 0)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, squeeze=False, figsize=(fig_width, fig_height))
    all_axes = [ax for ax_row in axes for ax in ax_row]

       # If class_names are provided, map labels and predictions to class names
    if class_names is not None:
        if labels is not None:
            if isinstance(class_names, dict):
                if isinstance(next(iter(class_names.keys())), str):  # Handle string keys (e.g., mini-ImageNet)
                    labels_to_marks = {v: k for k, v in class_names.items()}
                    labels = [f'{l}:{class_names[labels_to_marks[l]]}' for l in labels]
                else:  # For datasets like CIFAR-10 or Fashion-MNIST
                    labels = [f'{l}:{class_names[l]}' for l in labels]
            else:  # Assume class_names is a list
                labels = [f'{l}:{class_names[l]}' for l in labels]
        if predictions is not None:
            if len(predictions.shape) == 2:  # Handle probability distributions or one-hot vectors
                predictions = predictions.argmax(dim=1)
            predictions = [f'{p}:{class_names[p]}' for p in predictions]

    # Plot each image in the grid
    for b, ax in enumerate(all_axes):
        if b < max_to_show:
            # Rearrange to H*W*C
            img_p = images[b].permute([1, 2, 0])
            # Normalize the image
            img = (img_p - img_p.min()) / (img_p.max() - img_p.min())
            # Convert to numpy
            img = img.cpu().detach().numpy()

            # Display the image
            ax.imshow(img, cmap='gray')
            ax.axis('off')

            # Add title for labels and predictions
            if labels is not None:
                ax.set_title(f'{labels[b]}', fontsize=10 * scale ** 0.5)
            if predictions is not None:
                ax.set_title(f'pred: {predictions[b]}', fontsize=10 * scale ** 0.5)
            if labels is not None and predictions is not None:
                # Indicate correctness of predictions
                if labels[b] == predictions[b]:
                    mark, color = '✔', 'green'
                else:
                    mark, color = '✘', 'red'
                ax.set_title(f'label:{labels[b]}\npred:{predictions[b]} {mark}', color=color, fontsize=8 * scale ** 0.5)
        else:
            ax.axis('off')

    # Add the main title if provided
    if title is not None:
        x, align = (0.5, 'center') if center_title else (0, 'left')
        fig.suptitle(title, fontsize=14 * scale ** 0.5, x=x, horizontalalignment=align)

    # Adjust layout and display the plot
    fig.tight_layout()
    plt.show()
    plt.close()


# Quick function for displaying the classes of a task
def inspect_task(task_train, task_metadata, title=None):
    """
    Displays example images for each class in the task.

    Args:
        task_data (Dataset): The task-specific dataset containing classes and data.
        title (str, optional): Title for the visualization. Default is None.

    Returns:
        None: Displays a grid of example images for each class.
    """
    # Get the number of classes and their names as strings
    num_task_classes = len(task_metadata[0])
    
    task_classes = tuple([str(c) for c in task_metadata[0]])

    # Retrieve one example image for each class
    class_image_examples = [[batch[0] for batch in task_train if batch[1] == c][0] for c in range(num_task_classes)]

    # Display the images in a grid
    inspect_batch(
        class_image_examples,
        labels=task_classes,
        scale=0.7,
        num_cols=num_task_classes,
        title=title,
        center_title=False
    )


def training_plot(metrics,
      title=None, # optional figure title
      alpha=0.05, # smoothing parameter for train loss
      baselines=None, # optional list, or named dict, of baseline accuracies to compare to
      show_epochs=False,    # display boundary lines between epochs
      show_timesteps=False, # display discontinuities between CL timesteps
      results_dir=""
      ):
    """
    Plots training and validation loss/accuracy curves over training steps.

    Args:
        metrics (dict): Dictionary containing the following keys:
            - 'train_losses': List of training losses at each step.
            - 'val_losses': List of validation losses at each epoch.
            - 'train_accs': List of training accuracies at each step.
            - 'val_accs': List of validation accuracies at each epoch.
            - 'epoch_steps': List of training steps corresponding to epoch boundaries.
            - 'CL_timesteps': List of training steps corresponding to Continual Learning timesteps.
            - 'soft_losses' (optional): List of soft losses (e.g., from LwF) at each step.
        title (str, optional): Title for the entire figure. Defaults to None.
        alpha (float, optional): Exponential smoothing factor for curves. Defaults to 0.05.
        baselines (list or dict, optional): Baseline accuracies to plot as horizontal lines. 
            Can be a list of values or a dictionary with names and values. Defaults to None.
        show_epochs (bool, optional): If True, draws vertical lines at epoch boundaries. Defaults to False.
        show_timesteps (bool, optional): If True, draws vertical lines at Continual Learning timestep boundaries. Defaults to False.

    Returns:
        None: Displays the generated plot.
    """
    for metric_name in 'train_losses', 'val_losses', 'train_accs', 'val_accs', 'epoch_steps':
        assert metric_name in metrics, f"{metric_name} missing from metrics dict"

    fig, (loss_ax, acc_ax) = plt.subplots(1,2)

    # determine where to place boundaries, by calculating steps per epoch and epochs per timestep:
    steps_per_epoch = int(np.round(len(metrics['train_losses']) / len(metrics['val_losses'])))
    epochs_per_ts = int(np.round(len(metrics['epoch_steps']) / len(metrics['CL_timesteps'])))

    # if needing to show timesteps, we plot the curves discontinuously:
    if show_timesteps:
        # break the single list of metrics into nested sub-lists:
        timestep_train_losses, timestep_val_losses = [], []
        timestep_train_accs, timestep_val_accs = [], []
        timestep_epoch_steps, timestep_soft_losses = [], []
        prev_ts = 0
        for t, ts in enumerate(metrics['CL_timesteps']):
            timestep_train_losses.append(metrics['train_losses'][prev_ts:ts])
            timestep_train_accs.append(metrics['train_accs'][prev_ts:ts])
            timestep_val_losses.append(metrics['val_losses'][t*epochs_per_ts:(t+1)*epochs_per_ts])
            timestep_val_accs.append(metrics['val_accs'][t*epochs_per_ts:(t+1)*epochs_per_ts])
            timestep_epoch_steps.append(metrics['epoch_steps'][t*epochs_per_ts:(t+1)*epochs_per_ts])
            if 'soft_losses' in metrics:
                timestep_soft_losses.append(metrics['soft_losses'][prev_ts:ts])
            else:
                timestep_soft_losses.append(None)
            prev_ts = ts
    else:
        # just treat this as one timestep, by making lists of size 1:
        timestep_train_losses = [metrics['train_losses']]
        timestep_train_accs = [metrics['train_accs']]
        timestep_val_losses = [metrics['val_losses']]
        timestep_val_accs = [metrics['val_accs']]
        timestep_epoch_steps = [metrics['epoch_steps']]
        if 'soft_losses' in metrics:
            timestep_soft_losses = metrics['soft_losses']
        else:
            timestep_soft_losses = [None]

    # zip up the individual curves at each timestep:
    timestep_metrics = zip(timestep_train_losses,
                          timestep_train_accs,
                          timestep_val_losses,
                          timestep_val_accs,
                          timestep_epoch_steps,
                          metrics['CL_timesteps'],
                          timestep_soft_losses)

    for train_losses, train_accs, val_losses, val_accs, epoch_steps, ts, soft_losses in timestep_metrics:
        ### plot loss:
        smooth_train_loss = pd.Series(train_losses).ewm(alpha=alpha).mean()
        steps = np.arange(ts-len(train_losses), ts)

        # train loss is plotted at every step:
        loss_ax.plot(steps, smooth_train_loss, 'b-', label=f'train loss')
        # but val loss is plotted at every epoch:
        loss_ax.plot(epoch_steps, val_losses, 'r-', label=f'val loss')

        ### plot soft loss if given:
        if soft_losses is not None:
            smooth_soft_loss = pd.Series(soft_losses).ewm(alpha=alpha).mean()
            loss_ax.plot(steps, smooth_soft_loss, 'g-', label=f'soft loss')

        ### plot acc:
        smooth_train_acc = pd.Series(train_accs).ewm(alpha=alpha).mean()

        acc_ax.plot(steps, smooth_train_acc, 'b-', label=f'train acc')
        acc_ax.plot(epoch_steps, val_accs, 'r-', label=f'val acc')


    loss_legend = ['train loss', 'val loss'] if 'soft_loss' not in metrics else ['train loss', 'val loss', 'soft loss']
    acc_legend = ['train acc', 'val acc']

    loss_ax.legend(loss_legend); loss_ax.set_xlabel(f'Training step'); loss_ax.set_ylabel(f'Loss (CXE)')
    acc_ax.legend(acc_legend); acc_ax.set_xlabel(f'Training step'); acc_ax.set_ylabel(f'Accuracy')

    # format as percentage on right:
    acc_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
    acc_ax.yaxis.tick_right()
    acc_ax.yaxis.set_label_position('right')

    # optionally, draw lines at baseline accuracy points:
    if baselines is not None:
        if type(baselines) is list:
            for height in baselines:
                acc_ax.axhline(height, c=[0.8]*3, linestyle=':')
            # rescale y-axis to accommodate baselines if needed:
            plt.ylim([0, max(list(smooth_train_acc) + metrics['val_accs'] + baselines)+0.05])
        elif type(baselines) is dict:
            for name, height in baselines.items():
                acc_ax.axhline(height, c=[0.8]*3, linestyle=':')
                # add text label as well:
                acc_ax.text(0, height+0.002, name, c=[0.6]*3, size=8)
            plt.ylim([0, max(list(smooth_train_acc) + metrics['val_accs'] + [h for h in baselines.values()])+0.05])

    # optionally, draw epoch boundaries
    if show_epochs:
        for ax in (loss_ax, acc_ax):
            for epoch in metrics['epoch_steps']:
                ax.axvline(epoch, c=[0.9]*3, linestyle=':', zorder=1)

    # and/or CL timesteps:
    if show_timesteps:
        for ax in (loss_ax, acc_ax):
            for epoch in metrics['CL_timesteps']:
                ax.axvline(epoch, c=[.7,.7,.9], linestyle='--', zorder=0)


    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(results_dir)
    plt.close()


def get_batch_acc(pred, y):
    """
    Calculates accuracy for a batch of predictions.

    Args:
        pred (torch.Tensor): Predicted logits with shape (batch_size, num_classes).
        y (torch.Tensor): Ground truth labels as integers with shape (batch_size,).

    Returns:
        float: Accuracy as a scalar value.
    """
    return (pred.argmax(axis=1) == y).float().mean().item()

def evaluate_model(multitask_model: nn.Module,  # trained model capable of multi-task classification
                   val_loader: utils.data.DataLoader,  # task-specific data to evaluate on
                   loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                   device = None
                  ):
    """
    Evaluates the model on a validation dataset.

    Args:
        multitask_model (nn.Module): The trained multitask model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        loss_fn (_Loss, optional): Loss function to calculate validation loss. Default is CrossEntropyLoss.

    Returns:
        tuple: Average validation loss and accuracy across all batches.
    """
    with torch.no_grad():
        batch_val_losses, batch_val_accs = [], []

        # Iterate over all batches in the validation DataLoader
        for batch in val_loader:
            vx, vy, task_ids = batch
            vx, vy = vx.to(device), vy.to(device)
        
            # Forward pass with task-specific parameters
            vpred = multitask_model(vx, task_ids[0])

            # Calculate loss and accuracy for the batch
            val_loss = loss_fn(vpred, vy)
            val_acc = get_batch_acc(vpred, vy)

            batch_val_losses.append(val_loss.item())
            batch_val_accs.append(val_acc)

    # Return average loss and accuracy across all batches
    return np.mean(batch_val_losses), np.mean(batch_val_accs)

def evaluate_model_timed(multitask_model: nn.Module,  # trained model capable of multi-task classification
                   val_loader: utils.data.DataLoader,  # task-specific data to evaluate on
                   loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                   device = None
                  ):
    """
    Evaluates the model on a validation dataset.

    Args:
        multitask_model (nn.Module): The trained multitask model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        loss_fn (_Loss, optional): Loss function to calculate validation loss. Default is CrossEntropyLoss.

    Returns:
        tuple: Average validation loss and accuracy across all batches.
    """
    multitask_model.eval()
    with torch.no_grad():
        batch_val_losses, batch_val_accs = [], []

        time_inf = []
        # Iterate over all batches in the validation DataLoader
        for batch in val_loader:
            vx, vy, task_ids = batch
            vx, vy, task_ids = vx.to(device), vy.to(device), task_ids.to(device)

            start_time = time.time()
            
            # Forward pass with task-specific parameters
            vpred = multitask_model(vx, task_ids[0])
            time_inf.append(time.time() - start_time)

            # Calculate loss and accuracy for the batch
            val_loss = loss_fn(vpred, vy)
            val_acc = get_batch_acc(vpred, vy)

            batch_val_losses.append(val_loss.item())
            batch_val_accs.append(val_acc)

    # Return average loss and accuracy across all batches
    multitask_model.train()
    return np.mean(batch_val_losses), np.mean(batch_val_accs), np.mean(time_inf)

def evaluate_model_2d(multitask_model: nn.Module,  # trained model capable of multi-task classification
                   val_loader: utils.data.DataLoader,  # task-specific data to evaluate on
                   device, 
                   loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                   task_metadata = None,
                   task_id = 0,
                   wandb_run = None
                  ):
    """
    Evaluates the model on a validation dataset.

    Args:
        multitask_model (nn.Module): The trained multitask model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        loss_fn (_Loss, optional): Loss function to calculate validation loss. Default is CrossEntropyLoss.

    Returns:
        tuple: Average validation loss and accuracy across all batches.
    """
    with torch.no_grad():
        batch_val_losses, batch_val_accs = [], []
        batch_val_losses_prototypes, batch_val_accs_prototypes = [], []

        fig, ax = plt.subplots(len(task_metadata[int(task_id)]), 1, figsize=(10, 10))
        ax = ax.flatten()
        prototypes = multitask_model.get_prototypes(task_id)
        for i in range(len(task_metadata[int(task_id)])):
            ax[i].imshow(prototypes[i].cpu().detach().numpy().reshape(1, 20, 20).transpose(1, 2, 0), cmap='gray')
            ax[i].axis('off')
        file_name = f'prototypes_{int(task_id)}_{wandb_run}.png'
        plt.savefig(file_name)
        wandb.log({f'prototypes_{int(task_id)}': wandb.Image(file_name), 'task': task_id})
        plt.close()

        time_inf = []
        # Iterate over all batches in the validation DataLoader
        for batch in val_loader:
            vx, vy, task_ids = batch
            vx, vy = vx.to(device), vy.to(device)

            start_time = time.time()

            # Forward pass with task-specific parameters
            vpred, vpred_prototypes = multitask_model(vx, task_ids[0])
            time_inf.append(time.time() - start_time)

            # Calculate loss and accuracy for the batch
            val_loss = loss_fn(vpred, vy)
            vy_prototypes = torch.arange(len(task_metadata[int(task_id)]), device=device, dtype=torch.int64)         
            val_loss_prototypes = loss_fn(vpred_prototypes, vy_prototypes)

            val_acc = get_batch_acc(vpred, vy)
            val_acc_prototypes = get_batch_acc(vpred_prototypes, vy_prototypes)

            batch_val_losses.append(val_loss.item())
            batch_val_accs.append(val_acc)

            batch_val_losses_prototypes.append(val_loss_prototypes.item())
            batch_val_accs_prototypes.append(val_acc_prototypes)

    # Return average loss and accuracy across all batches
    return np.mean(batch_val_losses), np.mean(batch_val_accs) , np.mean(batch_val_losses_prototypes), np.mean(batch_val_accs_prototypes), np.mean(time_inf)


def evaluate_model_prototypes(multitask_model: nn.Module,  # trained model capable of multi-task classification
                   val_loader: torch.utils.data.DataLoader,  # task-specific data to evaluate on
                   prototypes: torch.Tensor,  # prototypes for the current task
                   task_id: int,  # current task id
                   loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                   device: torch.device = None
                  ):
    """
    Evaluates the model on a validation dataset.

    Args:
        multitask_model (nn.Module): The trained multitask model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        prototypes (torch.Tensor): Prototype images for the current task.
        task_id (int): The current task ID.
        loss_fn (_Loss, optional): Loss function to calculate validation loss. Default is CrossEntropyLoss.
        device (torch.device, optional): Device to perform evaluation on. Default is the global 'device'.

    Returns:
        tuple: Average validation loss and accuracy across all batches.
    """
    multitask_model.eval()
    with torch.no_grad():
        batch_val_losses, batch_val_accs = [], []

        # Iterate over all batches in the validation DataLoader
        for batch in val_loader:
            vx, vy, task_ids = batch
            vx, vy = vx.to(device), vy.to(device)
            current_task_id = task_ids[0].item()

            # Ensure the task_id matches
            if current_task_id != task_id:
                raise ValueError(f"Expected task_id {task_id}, but got {current_task_id}")

            # Forward pass with support set and prototypes
            vpred = multitask_model(vx, prototypes, task_id)

            # Calculate loss and accuracy for the batch
            val_loss = loss_fn(vpred, vy)
            val_acc = get_batch_acc(vpred, vy)

            batch_val_losses.append(val_loss.item())
            batch_val_accs.append(val_acc)

    # Return average loss and accuracy across all batches
    return np.mean(batch_val_losses), np.mean(batch_val_accs)

# Evaluate the model on the test sets of all tasks
def test_evaluate_prototypes(multitask_model: nn.Module, 
                  selected_test_sets,  
                  task_test_sets, 
                  task_prototypes,  # Added argument
                  device,
                  prev_accs = None,
                  show_taskwise_accuracy=True, 
                  baseline_taskwise_accs = None, 
                  model_name: str='', 
                  verbose=False, 
                  batch_size=16,
                  results_dir="",
                  task_metadata=None,
                  task_id=0,
                  loss_fn=nn.CrossEntropyLoss(),
                 ):
    """
    Evaluates the model on all selected test sets and optionally displays results.

    Args:
        multitask_model (nn.Module): The trained multitask model to evaluate.
        selected_test_sets (list[Dataset]): List of test datasets for each task.
        task_test_sets (list[Dataset]): List of all test datasets for each task.
        task_prototypes (dict[int, torch.Tensor]): Dictionary mapping task IDs to their prototypes.
        prev_accs (list[list[float]], optional): Previous accuracies for tracking forgetting.
        show_taskwise_accuracy (bool, optional): If True, plots a bar chart of taskwise accuracies.
        baseline_taskwise_accs (list[float], optional): Baseline accuracies for comparison.
        model_name (str, optional): Name of the model to show in plots. Default is ''.
        verbose (bool, optional): If True, prints detailed evaluation results. Default is False.
        batch_size (int, optional): Batch size for evaluation.
        results_dir (str, optional): Directory to save results plots.
        task_metadata (dict[int, dict[int, str]], optional): Metadata for tasks and classes.

    Returns:
        list[float]: Taskwise accuracies for the selected test sets.
    """
    if verbose:
        print(f'{model_name} evaluation on test set of all tasks:'.capitalize())

    task_test_losses = []
    task_test_accs = []

    # Iterate over each task's test dataset
    for t, test_data in enumerate(selected_test_sets):
        # Create a DataLoader for the current task's test dataset
        test_loader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False)  # Typically, shuffle=False for evaluation

        # Retrieve prototypes for the current task
        prototypes = task_prototypes[t].to(device)

        # Evaluate the model on the current task
        task_test_loss, task_test_acc = evaluate_model_prototypes(multitask_model, test_loader, prototypes, t, loss_fn, device=device)

        if verbose:
            class_names = [task_metadata[t][idx] for idx in range(len(task_metadata[t]))]
            print(f'Task {t} ({", ".join(class_names)}): {task_test_acc:.2%}')
            if baseline_taskwise_accs is not None:
                print(f'(Baseline: {baseline_taskwise_accs[t]:.2%})')

        task_test_losses.append(task_test_loss)
        task_test_accs.append(task_test_acc)

    # Calculate average test loss and accuracy across all tasks
    avg_task_test_loss = np.mean(task_test_losses)
    avg_task_test_acc = np.mean(task_test_accs)

    if verbose:
        print(f'\n +++ AVERAGE TASK TEST ACCURACY: {avg_task_test_acc:.2%} +++ ')

    # Plot taskwise accuracy if enabled
    if show_taskwise_accuracy:
        bar_heights = task_test_accs + [0]*(len(task_test_sets) - len(selected_test_sets))
        # display bar plot with accuracy on each evaluation task
        plt.bar(x = range(len(task_test_sets)), height=bar_heights, zorder=1)

        plt.xticks(
        range(len(task_test_sets)),
        [','.join(task_classes.values()) for t, task_classes in task_metadata.items()],
        rotation='vertical'
        )

        plt.axhline(avg_task_test_acc, c=[0.4]*3, linestyle=':')
        plt.text(0, avg_task_test_acc+0.002, f'{model_name} (average)', c=[0.4]*3, size=8)

        if prev_accs is not None:
            # plot the previous step's accuracies on top
            # (will show forgetting in red)
            for p, prev_acc_list in enumerate(prev_accs):
                plt.bar(x = range(len(prev_acc_list)), height=prev_acc_list, fc='tab:red', zorder=0, alpha=0.5*((p+1)/len(prev_accs)))

        if baseline_taskwise_accs is not None:
            for t, acc in enumerate(baseline_taskwise_accs):
                plt.plot([t-0.5, t+0.5], [acc, acc], c='black', linestyle='--')

            # show average as well:
            baseline_avg = np.mean(baseline_taskwise_accs)
            plt.axhline(baseline_avg, c=[0.6]*3, linestyle=':')
            plt.text(0, baseline_avg+0.002, 'baseline average', c=[0.6]*3, size=8)

        plt.ylim([0, 1])
        #plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure to wandb
        file_path = os.path.join(results_dir, f'taskwise_accuracy_task_{task_id}.png')
        plt.savefig(file_path)
        img = Image.open(file_path)
        wandb.log({f'taskwise accuracy': wandb.Image(img), 'task': task_id})

        plt.close()

    return task_test_accs


# Evaluate the model on the test sets of all tasks
def test_evaluate_2d(multitask_model: nn.Module, 
                  selected_test_sets,  
                  task_test_sets, 
                  device,
                  prev_accs,
                  prev_accs_prot,
                  show_taskwise_accuracy=True, 
                  baseline_taskwise_accs = None, 
                  model_name: str='', 
                  verbose=False, 
                  batch_size=16,
                  results_dir="",
                  task_id=0,
                  task_metadata=None,
                  wandb_run = None,
                 ):
    """
    Evaluates the model on all selected test sets and optionally displays results.
    Args:
        multitask_model (nn.Module): The trained multitask model to evaluate.
        selected_test_sets (list[Dataset]): List of test datasets for each task.
        prev_accs (list[list[float]], optional): Previous accuracies for tracking forgetting.
        show_taskwise_accuracy (bool, optional): If True, plots a bar chart of taskwise accuracies.
        baseline_taskwise_accs (list[float], optional): Baseline accuracies for comparison.
        model_name (str, optional): Name of the model to show in plots. Default is ''.
        verbose (bool, optional): If True, prints detailed evaluation results. Default is False.
    Returns:
        list[float]: Taskwise accuracies for the selected test sets.
    """
    metrics = {}
    
    print(f'{model_name} evaluation on test set of all tasks:'.capitalize())

    task_test_accs = []
    task_test_accs_prot = []
    task_test_times = []

    # Iterate over each task's test dataset
    for t, test_data in enumerate(selected_test_sets):
        # Create a DataLoader for the current task's test dataset
        test_loader = utils.data.DataLoader(test_data,
                                       batch_size=batch_size,
                                       shuffle=True)

        # Evaluate the model on the current task
        task_test_loss, task_test_acc, task_test_loss_prot, task_test_acc_prot, time = evaluate_model_2d(multitask_model, test_loader, task_metadata=task_metadata, task_id=task_id, wandb_run=wandb_run, device=device)

        print(f'{task_metadata[t]}: {task_test_acc:.2%}, prototypes acc: {task_test_acc_prot:.2%} in {time:.2f} s')

        task_test_accs.append(task_test_acc)
        task_test_accs_prot.append(task_test_acc_prot)
        task_test_times.append(time)

    metrics['task_test_accs'] = task_test_accs        
    AA = np.mean(task_test_accs)
    metrics['AA'] = AA
    
    metrics['task_test_accs_prot'] = task_test_accs_prot
    AA_prot = np.mean(task_test_accs_prot)
    metrics['AA_prot'] = AA_prot
    
    if t > 0:
        FM, BWT = compute_FM_BWT(task_test_accs, prev_accs)
        metrics['FM'] = FM
        metrics['BWT'] = BWT
    else:
        FM = 0
        BWT = 0
        metrics['FM'] = FM
        metrics['BWT'] = BWT
    
    Num_params = sum(p.numel() for p in multitask_model.parameters())
    Time_inf = np.mean(task_test_times)
    metrics['Num_params'] = Num_params
    metrics['Time_inf'] = Time_inf
    
    print(f'\n +++ AA: {AA:.2%}, AA_prot: {AA_prot:.2%}, FM: {FM:.2%}, BWT: {BWT:.2%}, Num_params: {Num_params}, Time_inf: {Time_inf:.2f} +++ ')

    # Plot taskwise accuracy if enabled
    bar_heights = task_test_accs + [0]*(len(task_test_sets) - len(selected_test_sets))
    plt.bar(x = range(len(task_test_sets)), height=bar_heights, zorder=1)

    plt.xticks(range(len(task_test_sets)), [','.join(task_classes.values()) for t, task_classes in task_metadata.items()], rotation='vertical')

    plt.axhline(AA, c=[0.4]*3, linestyle=':')
    plt.text(0, AA+0.002, f'{model_name} (average)', c=[0.4]*3, size=8)

    if prev_accs is not None:
        for p, prev_acc_list in enumerate(prev_accs):
            plt.bar(x = range(len(prev_acc_list)), height=prev_acc_list, fc='tab:red', zorder=0, alpha=0.5*((p+1)/len(prev_accs)))

    plt.ylim([0, 1])
    #plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure to wandb
    file_path = os.path.join(results_dir, f'taskwise_accuracy_task_{task_id}.png')
    plt.savefig(file_path)
    img = Image.open(file_path)
    wandb.log({f'taskwise accuracy': wandb.Image(img), 'task': task_id})

    plt.close()

    print(f'\n +++ AVERAGE TASK TEST ACCURACY PROTOTYPES: {AA:.2%} +++ ')

    # Plot taskwise accuracy if enabled
    bar_heights = task_test_accs_prot + [0]*(len(task_test_sets) - len(selected_test_sets))
    plt.bar(x = range(len(task_test_sets)), height=bar_heights, zorder=1)

    plt.xticks(range(len(task_test_sets)), [','.join(task_classes.values()) for t, task_classes in task_metadata.items()], rotation='vertical')

    plt.axhline(AA_prot, c=[0.4]*3, linestyle=':')

    for p, prev_acc_list in enumerate(prev_accs_prot):
        plt.bar(x = range(len(prev_acc_list)), height=prev_acc_list, fc='tab:red', zorder=0, alpha=0.5*((p+1)/len(prev_accs)))

    plt.ylim([0, 1])

    # Save figure to wandb
    file_path = os.path.join(results_dir, f'taskwise_accuracy_task_{task_id}_prot.png')
    plt.savefig(file_path)
    img = Image.open(file_path)
    wandb.log({f'taskwise accuracy prototypes': wandb.Image(img), 'task': task_id})

    plt.close()

    return metrics


def compute_FM_BWT(task_test_accs, prev_accs):
    """
    Computes the Forgetting Measure (FM) and Backward Transfer (BWT) metrics.

    FM is the average difference between the max accuracy on each task and the current accuracy.
    BWT is the average difference between the prev accuracy and the current accuracy. 

    Args:
        task_test_accs (list[float]): Taskwise accuracies for the selected test sets.
        prev_accs (list[list[float]]): Previous accuracies for tracking forgetting.

    Returns:
        tuple: Average forgetting and backward transfer across all tasks.
    """
    # Calculate the forgetting metric
    acc_per_task = []
    for prev_acc in prev_accs:
        acc_per_task.append([])
        for i, acc in enumerate(prev_acc):
            acc_per_task[i].append(acc)
    
    forgetting = [max(accs) - acc for accs, acc in zip(acc_per_task, task_test_accs)]
    FM = np.mean(forgetting)
    
    # Calculate the Backward Transfer metric
    backward_transfer = [acc - prev for prev, acc in zip(prev_accs[-1], task_test_accs)]
    BWT = np.mean(backward_transfer)
    
    return FM, BWT

# Evaluate the model on the test sets of all tasks
def test_evaluate(multitask_model: nn.Module, 
                  selected_test_sets,  
                  task_test_sets, 
                  prev_accs = None,
                  show_taskwise_accuracy=True, 
                  baseline_taskwise_accs = None, 
                  model_name: str='', 
                  verbose=False, 
                  batch_size=16,
                  results_dir="",
                  task_id=0,
                  task_metadata=None,
                  device=None
                 ):
    """
    Evaluates the model on all selected test sets and optionally displays results.
    Args:
        multitask_model (nn.Module): The trained multitask model to evaluate.
        selected_test_sets (list[Dataset]): List of test datasets for each task.
        prev_accs (list[list[float]], optional): Previous accuracies for tracking forgetting.
        show_taskwise_accuracy (bool, optional): If True, plots a bar chart of taskwise accuracies.
        baseline_taskwise_accs (list[float], optional): Baseline accuracies for comparison.
        model_name (str, optional): Name of the model to show in plots. Default is ''.
        verbose (bool, optional): If True, prints detailed evaluation results. Default is False.
    Returns:
        list[float]: Taskwise accuracies for the selected test sets.
    """
    if verbose:
        print(f'{model_name} evaluation on test set of all tasks:'.capitalize())

    task_test_losses = []
    task_test_accs = []

    # Iterate over each task's test dataset
    for t, test_data in enumerate(selected_test_sets):
        # Create a DataLoader for the current task's test dataset
        test_loader = utils.data.DataLoader(test_data,
                                       batch_size=batch_size,
                                       shuffle=True)

        # Evaluate the model on the current task
        task_test_loss, task_test_acc = evaluate_model(multitask_model, test_loader, device=device)

        if verbose:
            print(f'{task_metadata[t]}: {task_test_acc:.2%}')
            if baseline_taskwise_accs is not None:
                print(f'(Baseline: {baseline_taskwise_accs[t]:.2%})')

        task_test_losses.append(task_test_loss)
        task_test_accs.append(task_test_acc)

    # Calculate average test loss and accuracy across all tasks
    avg_task_test_loss = np.mean(task_test_losses)
    avg_task_test_acc = np.mean(task_test_accs)

    print(f'\n +++ AVERAGE TASK TEST ACCURACY: {avg_task_test_acc:.2%} +++ ')

    # Plot taskwise accuracy if enabled
    if show_taskwise_accuracy:
        bar_heights = task_test_accs + [0]*(len(task_test_sets) - len(selected_test_sets))
        # display bar plot with accuracy on each evaluation task
        plt.bar(x = range(len(task_test_sets)), height=bar_heights, zorder=1)

        plt.xticks(
        range(len(task_test_sets)),
        [','.join(task_classes.values()) for t, task_classes in task_metadata.items()],
        rotation='vertical'
        )

        plt.axhline(avg_task_test_acc, c=[0.4]*3, linestyle=':')
        plt.text(0, avg_task_test_acc+0.002, f'{model_name} (average)', c=[0.4]*3, size=8)

        if prev_accs is not None:
            # plot the previous step's accuracies on top
            # (will show forgetting in red)
            for p, prev_acc_list in enumerate(prev_accs):
                plt.bar(x = range(len(prev_acc_list)), height=prev_acc_list, fc='tab:red', zorder=0, alpha=0.5*((p+1)/len(prev_accs)))

        if baseline_taskwise_accs is not None:
            for t, acc in enumerate(baseline_taskwise_accs):
                plt.plot([t-0.5, t+0.5], [acc, acc], c='black', linestyle='--')

            # show average as well:
            baseline_avg = np.mean(baseline_taskwise_accs)
            plt.axhline(baseline_avg, c=[0.6]*3, linestyle=':')
            plt.text(0, baseline_avg+0.002, 'baseline average', c=[0.6]*3, size=8)

        plt.ylim([0, 1])
        #plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure to wandb
        file_path = os.path.join(results_dir, f'taskwise_accuracy_task_{task_id}.png')
        plt.savefig(file_path)
        img = Image.open(file_path)
        wandb.log({f'taskwise accuracy': wandb.Image(img), 'task': task_id})

        plt.close()

    return task_test_accs

# Evaluate the model on the test sets of all tasks
def test_evaluate_metrics(multitask_model: nn.Module, 
                  selected_test_sets,  
                  task_test_sets, 
                  prev_accs = None,
                  show_taskwise_accuracy=True, 
                  baseline_taskwise_accs = None, 
                  model_name: str='', 
                  verbose=False, 
                  batch_size=16,
                  results_dir="",
                  task_id=0,
                  task_metadata=None,
                  device=None
                 ):
    """
    Evaluates the model on all selected test sets and optionally displays results.
    Args:
        multitask_model (nn.Module): The trained multitask model to evaluate.
        selected_test_sets (list[Dataset]): List of test datasets for each task.
        prev_accs (list[list[float]], optional): Previous accuracies for tracking forgetting.
        show_taskwise_accuracy (bool, optional): If True, plots a bar chart of taskwise accuracies.
        baseline_taskwise_accs (list[float], optional): Baseline accuracies for comparison.
        model_name (str, optional): Name of the model to show in plots. Default is ''.
        verbose (bool, optional): If True, prints detailed evaluation results. Default is False.
    Returns:
        list[float]: Taskwise accuracies for the selected test sets.
    """
    metrics = {}
    
    print(f'{model_name} evaluation on test set of all tasks:'.capitalize())

    task_test_losses = []
    task_test_accs = []
    task_test_times = []

    # Iterate over each task's test dataset
    for t, test_data in enumerate(selected_test_sets):
        # Create a DataLoader for the current task's test dataset
        test_loader = utils.data.DataLoader(test_data,
                                       batch_size=batch_size,
                                       shuffle=True)

        # Evaluate the model on the current task
        _, task_test_acc, time = evaluate_model_timed(multitask_model, test_loader, device=device)

        print(f'{task_metadata[t]}: {task_test_acc:.2%} in {time:.2f} seconds')

        task_test_accs.append(task_test_acc)
        task_test_times.append(time)

    # print(task_test_times)
    metrics['task_test_accs'] = task_test_accs        
    AA = np.mean(task_test_accs)
    metrics['AA'] = AA
    
    if t > 0:
        FM, BWT = compute_FM_BWT(task_test_accs, prev_accs)
        metrics['FM'] = FM
        metrics['BWT'] = BWT
    else:
        FM = 0
        BWT = 0
        metrics['FM'] = FM
        metrics['BWT'] = BWT
    
    Num_params = sum(p.numel() for p in multitask_model.parameters())
    Time_inf = np.mean(task_test_times)
    metrics['Num_params'] = Num_params
    metrics['Time_inf'] = Time_inf
    
    print(f'\n +++ AA: {AA:.2%}, FM: {FM:.2%}, BWT: {BWT:.2%}, Num_params: {Num_params}, Time_inf: {Time_inf:.2f} +++ ')

    # Plot taskwise accuracy if enabled
    if show_taskwise_accuracy:
        bar_heights = task_test_accs + [0]*(len(task_test_sets) - len(selected_test_sets))
        # display bar plot with accuracy on each evaluation task
        plt.bar(x = range(len(task_test_sets)), height=bar_heights, zorder=1)

        plt.xticks(
        range(len(task_test_sets)),
        [','.join(task_classes.values()) for t, task_classes in task_metadata.items()],
        rotation='vertical'
        )

        plt.axhline(AA, c=[0.4]*3, linestyle=':')
        plt.text(0, AA+0.002, f'{model_name} (average)', c=[0.4]*3, size=8)

        if prev_accs is not None:
            # plot the previous step's accuracies on top
            # (will show forgetting in red)
            for p, prev_acc_list in enumerate(prev_accs):
                plt.bar(x = range(len(prev_acc_list)), height=prev_acc_list, fc='tab:red', zorder=0, alpha=0.5*((p+1)/len(prev_accs)))

        if baseline_taskwise_accs is not None:
            for t, acc in enumerate(baseline_taskwise_accs):
                plt.plot([t-0.5, t+0.5], [acc, acc], c='black', linestyle='--')

            # show average as well:
            baseline_avg = np.mean(baseline_taskwise_accs)
            plt.axhline(baseline_avg, c=[0.6]*3, linestyle=':')
            plt.text(0, baseline_avg+0.002, 'baseline average', c=[0.6]*3, size=8)

        plt.ylim([0, 1])
        #plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure to wandb
        file_path = os.path.join(results_dir, f'taskwise_accuracy_task_{task_id}.png')
        plt.savefig(file_path)
        img = Image.open(file_path)
        wandb.log({f'taskwise accuracy': wandb.Image(img), 'task': task_id})

        plt.close()

    return metrics


def setup_dataset(dataset_name, data_dir='./data', num_tasks=10, val_frac=0.1, test_frac=0.1, batch_size=256):
    """
    Sets up dataset, dataloaders, and metadata for training and testing.

    Args:
        dataset_name (str): Name of the dataset ('Split-CIFAR100', 'TinyImagenet', 'Split-MNIST').
        data_dir (str): Directory where the dataset is stored.
        num_tasks (int): Number of tasks to split the dataset into.
        val_frac (float): Fraction of the data to use for validation.
        test_frac (float): Fraction of the data to use for testing.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        dict: A dictionary containing dataloaders and metadata for training and testing.
    """
    # Initialization
    timestep_tasks = {}

    task_test_sets = []
    task_metadata = {}

    # Dataset-specific settings
    if dataset_name == 'Split-MNIST':
        dataset_train = datasets.MNIST(root=data_dir, train=True, download=True)
        dataset_test = datasets.MNIST(root=data_dir, train=False, download=True)
        
        num_classes = 10
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), # Convert to 3-channel grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        task_classes_per_task = num_classes // num_tasks
        timestep_task_classes = {
            t: list(range(t * task_classes_per_task, (t + 1) * task_classes_per_task))
            for t in range(num_tasks)
        }

    elif dataset_name == 'Split-CIFAR100':
        dataset_train = datasets.CIFAR100(root=data_dir, train=True, download=True)
        dataset_test = datasets.CIFAR100(root=data_dir, train=False, download=True)
        num_classes = 100
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        task_classes_per_task = num_classes // num_tasks
        timestep_task_classes = {
            t: list(range(t * task_classes_per_task, (t + 1) * task_classes_per_task))
            for t in range(num_tasks)
        }

    elif dataset_name == 'TinyImageNet':
        dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200', 'train'))
        dataset_test = datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200', 'val'))
        num_classes = 200
        preprocess = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        task_classes_per_task = num_classes // num_tasks
        timestep_task_classes = {
            t: list(range(t * task_classes_per_task, (t + 1) * task_classes_per_task))
            for t in range(num_tasks)
        }


    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_images_per_class = {}
    for class_idx in tqdm(range(num_classes)):
        indices = [i for i, label in enumerate(dataset_train.targets) if label == class_idx]
        train_images_per_class[class_idx] = indices  # Store indices instead of images
        # If you need images elsewhere, consider storing them separately or processing them on-the-fly
        
    # Process tasks
    for t, task_classes in timestep_task_classes.items():
        if dataset_name == 'Split-MNIST':
            task_indices_train = [i for i, label in enumerate(dataset_train.targets) if label in task_classes]
            task_images_train = [Image.fromarray(np.array(dataset_train.data[i]), mode='L') for i in task_indices_train]
            task_labels_train = [label for i, label in enumerate(dataset_train.targets) if label in task_classes]
            task_indices_test = [i for i, label in enumerate(dataset_test.targets) if label in task_classes]
            task_images_test = [Image.fromarray(np.array(dataset_test.data[i]), mode='L') for i in task_indices_test]
            task_labels_test = [label for i, label in enumerate(dataset_test.targets) if label in task_classes]

        elif dataset_name == 'Split-CIFAR100':
            task_indices_train = [i for i, label in enumerate(dataset_train.targets) if label in task_classes]
            task_images_train = [Image.fromarray(dataset_train.data[i]) for i in task_indices_train]
            task_labels_train = [label for i, label in enumerate(dataset_train.targets) if label in task_classes]
            task_indices_test = [i for i, label in enumerate(dataset_test.targets) if label in task_classes]
            task_images_test = [Image.fromarray(dataset_test.data[i]) for i in task_indices_test]
            task_labels_test = [label for i, label in enumerate(dataset_test.targets) if label in task_classes]

        elif dataset_name == 'TinyImageNet':
            task_indices_train = [i for i, (_, label) in enumerate(dataset_train.samples) if label in task_classes]
            task_images_train = [dataset_train[i][0] for i in task_indices_train]
            task_labels_train = [label for i, (_, label) in enumerate(dataset_train.samples) if label in task_classes]
            task_indices_test = [i for i, (_, label) in enumerate(dataset_test.samples) if label in task_classes]
            task_images_test = [dataset_test[i][0] for i in task_indices_test]
            task_labels_test = [label for i, (_, label) in enumerate(dataset_test.samples) if label in task_classes]

        
        # Map old labels to 0-based labels for the task
        class_to_idx = {orig: idx for idx, orig in enumerate(task_classes)}
        task_labels = [class_to_idx[int(label)] for label in task_labels_train]

        # Map old labels to 0-based labels for the task for the test
        task_labels_test = [class_to_idx[int(label)] for label in task_labels_test]

        # Create tensors
        task_images_train_tensor = torch.stack([preprocess(img) for img in task_images_train])
        task_labels_train_tensor = torch.tensor(task_labels, dtype=torch.long)
        task_ids_train_tensor = torch.full((len(task_labels_train_tensor),), t, dtype=torch.long)

        # TensorDataset
        task_dataset_train = TensorDataset(task_images_train_tensor, task_labels_train_tensor, task_ids_train_tensor)

        # Train/Validation/Test split
        train_size = int((1 - val_frac) * len(task_dataset_train))
        val_size = len(task_dataset_train) - train_size

        
        train_set, val_set = random_split(task_dataset_train, [train_size, val_size])
        
        task_images_test_tensor = torch.stack([preprocess(img) for img in task_images_test])
        task_labels_test_tensor = torch.tensor(task_labels_test, dtype=torch.long)
        task_ids_test_tensor = torch.full((len(task_labels_test_tensor),), t, dtype=torch.long)
        
        test_set = TensorDataset(task_images_test_tensor, task_labels_test_tensor, task_ids_test_tensor)

        # Store datasets and metadata
        timestep_tasks[t] = (train_set, val_set)
        task_test_sets.append(test_set)
        if dataset_name == 'TinyImagenet':
            task_metadata[t] = {
                idx: os.path.basename(dataset_train.classes[orig]) for orig, idx in class_to_idx.items()
            }
        else:
            task_metadata[t] = {
                idx: dataset_train.classes[orig] if hasattr(dataset_train, 'classes') else str(orig)
                for orig, idx in class_to_idx.items()
            }

    # Final datasets
    final_test_data = ConcatDataset(task_test_sets)
    final_test_loader = DataLoader(final_test_data, batch_size=batch_size, shuffle=True)
    print(f"Final test size (containing all tasks): {len(final_test_data)}")
    

    return {
        'timestep_tasks': timestep_tasks,
        'final_test_loader': final_test_loader,
        'task_metadata': task_metadata,
        'task_test_sets': task_test_sets,
        'images_per_class': train_images_per_class,
        'timestep_task_classes': timestep_task_classes
    }


class MinimumSubsetBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, task_classes, images_per_class):
        self.dataset = dataset
        self.batch_size = batch_size
        self.task_classes = task_classes
        self.images_per_class = images_per_class

        # print("Images per class:", self.images_per_class.keys())

        for class_idx in self.task_classes:
            if len(self.images_per_class[class_idx]) == 0:
                raise ValueError(f"No samples found for class {class_idx}.")

        self.class_to_indices = {
            class_idx: self.images_per_class[class_idx].copy()
            for class_idx in self.task_classes
        }

        for class_idx in self.task_classes:
            random.shuffle(self.class_to_indices[class_idx])

    def __iter__(self):
        class_iterators = {class_idx: iter(indices) for class_idx, indices in self.class_to_indices.items()}

        while True:
            batch = []

            try:
                for class_idx in self.task_classes:
                    batch.append(next(class_iterators[class_idx]))
            except StopIteration:
                break

            remaining_batch_size = self.batch_size - len(batch)
            if remaining_batch_size > 0:
                all_class_indices = [idx for class_indices in self.class_to_indices.values() for idx in class_indices]
                available_indices = list(set(all_class_indices) - set(batch))
                if remaining_batch_size > len(available_indices):
                    sampled = available_indices
                else:
                    sampled = random.sample(available_indices, remaining_batch_size)
                batch += sampled

            #print("Batch:", batch)
            yield batch

    def __len__(self):
        min_class_len = min(len(indices) for indices in self.class_to_indices.values())
        return min_class_len
    
    
def setup_dataset_prototype(dataset_name, data_dir='./data', num_tasks=10, val_frac=0.1, test_frac=0.1, batch_size=256):
    """
    Sets up dataset, dataloaders, and metadata for training and testing.

    Args:
        dataset_name (str): Name of the dataset ('Split-CIFAR100', 'TinyImagenet', 'Split-MNIST').
        data_dir (str): Directory where the dataset is stored.
        num_tasks (int): Number of tasks to split the dataset into.
        val_frac (float): Fraction of the data to use for validation.
        test_frac (float): Fraction of the data to use for testing.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        dict: A dictionary containing dataloaders and metadata for training and testing.
    """
    # Initialization
    timestep_tasks = {}

    task_test_sets = []
    task_metadata = {}

    # Dataset-specific settings
    if dataset_name == 'Split-MNIST':
        dataset_train = datasets.MNIST(root=data_dir, train=True, download=True)
        dataset_test = datasets.MNIST(root=data_dir, train=False, download=True)
        
        num_classes = 10
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), # Convert to 3-channel grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        task_classes_per_task = num_classes // num_tasks
        timestep_task_classes = {
            t: list(range(t * task_classes_per_task, (t + 1) * task_classes_per_task))
            for t in range(num_tasks)
        }

    elif dataset_name == 'Split-CIFAR100':
        dataset_train = datasets.CIFAR100(root=data_dir, train=True, download=True)
        dataset_test = datasets.CIFAR100(root=data_dir, train=False, download=True)
        
        
        num_classes = 100
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        task_classes_per_task = num_classes // num_tasks
        timestep_task_classes = {
            t: list(range(t * task_classes_per_task, (t + 1) * task_classes_per_task))
            for t in range(num_tasks)
        }

    elif dataset_name == 'TinyImageNet':
        dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200', 'train'))
        num_classes = 200
        preprocess = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        task_classes_per_task = num_classes // num_tasks
        timestep_task_classes = {
            t: list(range(t * task_classes_per_task, (t + 1) * task_classes_per_task))
            for t in range(num_tasks)
        }

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Build a dictionary of training indices per class
    train_images_per_class = {}
    for class_idx in tqdm(range(num_classes), desc="Collecting training indices per class"):
        if dataset_name in ['Split-MNIST', 'Split-CIFAR100']:
            indices = [i for i, label in enumerate(dataset_train.targets) if label == class_idx]
        elif dataset_name == 'TinyImageNet':
            indices = [i for i, (_, label) in enumerate(dataset_train.samples) if label == class_idx]
        train_images_per_class[class_idx] = indices  # Store indices instead of images

    # Select one prototype image per class
    train_prototype_image_per_class = {}
    prototype_indices = set()
    
    for class_idx in range(num_classes):
        if len(train_images_per_class[class_idx]) == 0:
            raise ValueError(f"No training images found for class {class_idx} in dataset {dataset_name}.")
        prototype_idx = random.choice(train_images_per_class[class_idx])
        prototype_indices.add(prototype_idx)

        # Load and preprocess the prototype image
        if dataset_name == 'Split-MNIST':
            # For MNIST, dataset_train[idx] returns (image, label)
            img, _ = dataset_train[prototype_idx]
            img = preprocess(img)
        elif dataset_name == 'Split-CIFAR100':
            img, _ = dataset_train[prototype_idx]
            img = preprocess(img)
        elif dataset_name == 'TinyImageNet':
            img, _ = dataset_train[prototype_idx]
            img = preprocess(img)
        
        train_prototype_image_per_class[class_idx] = img
        
        # Remove the prototype index from training indices to ensure disjointness
        train_images_per_class[class_idx].remove(prototype_idx)
        
    # Process tasks
    for t, task_classes in tqdm(timestep_task_classes.items(), desc="Processing tasks"):
        if dataset_name == 'Split-MNIST':
            # Exclude prototype indices
            task_indices_train = [
                i for i, label in enumerate(dataset_train.targets)
                if label in task_classes and i not in prototype_indices
            ]
            task_images_train = [Image.fromarray(np.array(dataset_train.data[i]), mode='L') for i in task_indices_train]
            task_labels_train = [label for i, label in enumerate(dataset_train.targets) if label in task_classes and i not in prototype_indices]
            task_indices_test = [i for i, label in enumerate(dataset_test.targets) if label in task_classes]
            task_images_test = [Image.fromarray(np.array(dataset_test.data[i]), mode='L') for i in task_indices_test]
            task_labels_test = [label for i, label in enumerate(dataset_test.targets) if label in task_classes]

        elif dataset_name == 'Split-CIFAR100':
            # Exclude prototype indices
            task_indices_train = [
                i for i, label in enumerate(dataset_train.targets)
                if label in task_classes and i not in prototype_indices
            ]
            task_images_train = [Image.fromarray(dataset_train.data[i]) for i in task_indices_train]
            task_labels_train = [label for i, label in enumerate(dataset_train.targets) if label in task_classes and i not in prototype_indices]
            task_indices_test = [i for i, label in enumerate(dataset_test.targets) if label in task_classes]
            task_images_test = [Image.fromarray(dataset_test.data[i]) for i in task_indices_test]
            task_labels_test = [label for i, label in enumerate(dataset_test.targets) if label in task_classes]

        elif dataset_name == 'TinyImageNet':
            # Exclude prototype indices
            task_indices_train = [
                i for i, (_, label) in enumerate(dataset_train.samples)
                if label in task_classes and i not in prototype_indices
            ]
            task_images_train = [dataset_train[i][0] for i in task_indices_train]
            task_labels_train = [label for i, (_, label) in enumerate(dataset_train.samples) if label in task_classes and i not in prototype_indices]
            task_indices_test = [i for i, (_, label) in enumerate(dataset_test.samples) if label in task_classes]
            task_images_test = [dataset_test[i][0] for i in task_indices_test]
            task_labels_test = [label for i, (_, label) in enumerate(dataset_test.samples) if label in task_classes]

        # Map old labels to 0-based labels for the task
        class_to_idx = {orig: idx for idx, orig in enumerate(task_classes)}
        task_labels = [class_to_idx[int(label)] for label in task_labels_train]
        task_labels_test = [class_to_idx[int(label)] for label in task_labels_test]

        # Create tensors
        task_images_train_tensor = torch.stack([preprocess(img) for img in task_images_train])
        task_labels_train_tensor = torch.tensor(task_labels, dtype=torch.long)
        task_ids_train_tensor = torch.full((len(task_labels_train_tensor),), t, dtype=torch.long)
        
        # TensorDataset for training
        task_dataset_train = TensorDataset(task_images_train_tensor, task_labels_train_tensor, task_ids_train_tensor)
        
        
        # Train/Validation split
        train_size = int((1 - val_frac) * len(task_dataset_train))
        val_size = len(task_dataset_train) - train_size
        train_set, val_set = random_split(task_dataset_train, [train_size, val_size])
        
        
        # Prepare test set
        task_images_test_tensor = torch.stack([preprocess(img) for img in task_images_test])
        task_labels_test_tensor = torch.tensor(task_labels_test, dtype=torch.long)
        task_ids_test_tensor = torch.full((len(task_labels_test_tensor),), t, dtype=torch.long)
        
        # TensorDataset for testing
        test_set = TensorDataset(task_images_test_tensor, task_labels_test_tensor, task_ids_test_tensor)
        
        # Store datasets and metadata
        timestep_tasks[t] = (train_set, val_set)
        task_test_sets.append(test_set)
        if dataset_name == 'TinyImageNet':
            task_metadata[t] = {
                idx: os.path.basename(dataset_train.classes[orig]) for orig, idx in class_to_idx.items()
            }
        else:
            task_metadata[t] = {
                idx: dataset_train.classes[orig] if hasattr(dataset_train, 'classes') else str(orig)
                for orig, idx in class_to_idx.items()
            }

    # Final test data loader
    final_test_data = ConcatDataset(task_test_sets)
    final_test_loader = DataLoader(final_test_data, batch_size=batch_size, shuffle=True)
    print(f"Final test size (containing all tasks): {len(final_test_data)}")
    
    # Create a prototype loader
    prototype_batch_size = num_classes // num_tasks
    prototype_images_tensor = torch.stack([train_prototype_image_per_class[c] for c in range(num_classes)])
    prototype_loader = DataLoader(prototype_images_tensor, batch_size=prototype_batch_size, shuffle=False)
    print(f"Prototype loader size: {len(prototype_images_tensor)}")
    
    # Create a mapping from task_id to prototypes
    task_prototypes = {
        t: torch.stack([train_prototype_image_per_class[c] for c in timestep_task_classes[t]])
        for t in range(num_tasks)
    }
    
    return {
        'timestep_tasks': timestep_tasks,
        'final_test_loader': final_test_loader,
        'task_metadata': task_metadata,
        'task_test_sets': task_test_sets,
        'images_per_class': train_images_per_class,
        'timestep_task_classes': timestep_task_classes,
        'train_prototype_image_per_class': train_prototype_image_per_class,
        'prototype_loader': prototype_loader,
        'task_prototypes': task_prototypes,
        'prototype_indices': prototype_indices
    }




import torch

def temperature_softmax(x, T):
    """Applies temperature-scaled softmax over the channel dimension.
    
    Args:
        x (torch.Tensor): Input tensor (batch, num_classes).
        T (float): Temperature for scaling logits.

    Returns:
        torch.Tensor: Probability distribution of shape (batch, num_classes).
    """
    return torch.softmax(x / T, dim=1)

def KL_divergence(p, q, epsilon=1e-10):
    """Computes the Kullback-Leibler (KL) divergence between two distributions.
    
    Args:
        p (torch.Tensor): First probability distribution (batch, num_classes).
        q (torch.Tensor): Second probability distribution (batch, num_classes).
        epsilon (float): Small constant to avoid log(0) or division by zero.

    Returns:
        torch.Tensor: KL divergence per example in the batch (batch,).
    """
    # Add epsilon to avoid log(0) or division by zero
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    
    # Compute KL divergence
    kl_div = torch.sum(p * torch.log(p / q), dim=-1)
    
    return kl_div

def distillation_output_loss(student_pred, teacher_pred, temperature):
    """Computes the distillation loss between student and teacher model predictions.
    
    Args:
        student_pred (torch.Tensor): Logits from the student model (batch, num_classes).
        teacher_pred (torch.Tensor): Logits from the teacher model (batch, num_classes).
        temperature (float): Temperature for scaling logits.

    Returns:
        torch.Tensor: Distillation loss per example in the batch (batch,).
    """
    # Apply temperature-scaled softmax to student and teacher predictions
    student_soft = temperature_softmax(student_pred, temperature)
    teacher_soft = temperature_softmax(teacher_pred, temperature)

    # Compute KL divergence as distillation loss
    kl_div = KL_divergence(student_soft, teacher_soft)
    #Only print if nan values are present
    if torch.isnan(kl_div).any():
        print(f'KL div shape: {kl_div.shape} || KL div: {kl_div} between student and teacher temperature softmax')

    # Return scaled KL divergence
    return kl_div * (temperature ** 2)

import os
import logging
class logger:
    def __init__(self, results_dir):
        logging.basicConfig(filename=os.path.join(results_dir, 'training.log'), 
                            level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logg = logging.getLogger()
    
    def log(self, message):
        self.logg.info(message)
        print(message)
        
        
        
        
def seed_everything(seed=69):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
        

def _merge(d1: EasyDict, d2: EasyDict):
	res = EasyDict(d1)
	for key, value in d2.items():
		if isinstance(value, EasyDict) and isinstance(res.get(key, None), EasyDict):
			res[key] = _merge(res[key], value)
		else: res[key] = value
	return res

        
def config_load(filename) -> EasyDict:
	if filename.find('/') != -1:
		__CONFIG_ROOTDIR__ = filename.split('/')[-2]
	else: __CONFIG_ROOTDIR__ = ''
	with open(filename, 'r') as f:
		__CONTENT__ = f.read()
	del filename, f
	exec(__CONTENT__)
	res = EasyDict(**locals())
	res.pop('__CONTENT__')
	res.pop('__CONFIG_ROOTDIR__')
	if res.get('_base_', None) is not None:
		bases = EasyDict()
		for basename in list(res._base_):
			base = config_load(os.path.join(__CONFIG_ROOTDIR__, basename))
			bases = _merge(bases, base)
		res = _merge(res, bases)
	return res
        

def setup_optimizer(parameters, lr, l2_reg, optimizer):
    if optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    elif optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg)
    elif optimizer == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)
    
        
def count_optimizer_parameters(optimizer: torch.optim.Optimizer, logger=None) -> None:
    """
    Prints the number of parameters in each parameter group of the optimizer
    and the total number of unique parameters being optimized.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance to inspect.
    """
    total_params = 0
    unique_params = set()
    
    logger.log("\n=== Optimizer Parameter Groups ===")
    for idx, param_group in enumerate(optimizer.param_groups):
        num_params = len(param_group['params'])
        # Calculate the total number of parameters in this group
        num_params_in_group = sum(p.numel() for p in param_group['params'])
        print(f"Parameter Group {idx + 1}: {num_params} parameters, Total Parameters: {num_params_in_group}")
        total_params += num_params_in_group
        # Add to the set of unique parameters to avoid double-counting
        for p in param_group['params']:
            unique_params.add(p)
    
    logger.log(f"Total Optimized Parameters: {total_params} ({sum(p.numel() for p in unique_params)} unique)")
    logger.log("===================================\n")
     
     

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        # Compute differences between adjacent pixels
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])  # Horizontal differences
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])  # Vertical differences
        
        # Sum over all dimensions
        tv_loss = diff_h.mean() + diff_w.mean()
        return tv_loss     
     
     
     
#-----------------------------------------------------------------#
#----------------------utils.py-----------------------------------#
#-----------------------------------------------------------------#
        
        

# Main block to test the setup_dataset_prototype function
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Set random seeds for reproducibility
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)

    # Parameters for the setup
    dataset_name = 'Split-CIFAR100'  # Change as needed: 'Split-MNIST', 'Split-CIFAR100', 'TinyImageNet'
    data_dir = './data'              # Ensure this directory exists or change as needed
    num_tasks = 10
    val_frac = 0.1
    test_frac = 0.1
    batch_size = 256

    # Setup the dataset
    print("Setting up the dataset with prototypes...")
    dataset_info = setup_dataset_prototype(
        dataset_name=dataset_name,
        data_dir=data_dir,
        num_tasks=num_tasks,
        val_frac=val_frac,
        test_frac=test_frac,
        batch_size=batch_size
    )
    print("Dataset setup completed.\n")

    # Accessing the returned dictionary
    timestep_tasks = dataset_info['timestep_tasks']
    final_test_loader = dataset_info['final_test_loader']
    task_metadata = dataset_info['task_metadata']
    task_test_sets = dataset_info['task_test_sets']
    images_per_class = dataset_info['images_per_class']
    timestep_task_classes = dataset_info['timestep_task_classes']
    train_prototypes = dataset_info['train_prototype_image_per_class']
    prototype_loader = dataset_info['prototype_loader']
    task_prototypes = dataset_info['task_prototypes']
    prototype_indices = dataset_info['prototype_indices']

    # Verify prototype_loader
    print("Verifying prototype_loader...")
    for batch_idx, batch in enumerate(prototype_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f" - Batch shape: {batch.shape}")  # Expected: (prototype_batch_size, C, H, W)
        # Optionally, visualize prototypes in the first batch
        if batch_idx == 0:
            num_prototypes = batch.size(0)
            plt.figure(figsize=(num_prototypes * 2, 2))
            for i in range(num_prototypes):
                img = batch[i]
                # Unnormalize the image for visualization
                if dataset_name == 'Split-MNIST':
                    img = img * 0.5 + 0.5  # Since it was normalized with mean=0.5, std=0.5
                    img = img.squeeze().numpy()
                    plt.subplot(1, num_prototypes, i + 1)
                    plt.imshow(img, cmap='gray')
                elif dataset_name in ['Split-CIFAR100', 'TinyImageNet']:
                    # Adjust unnormalization based on dataset
                    if dataset_name == 'Split-CIFAR100':
                        mean = np.array([0.5, 0.5, 0.5])
                        std = np.array([0.5, 0.5, 0.5])
                    elif dataset_name == 'TinyImageNet':
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                    img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
                    img = (img * std + mean).clip(0, 1)  # Unnormalize and clip
                    plt.subplot(1, num_prototypes, i + 1)
                    plt.imshow(img)
                plt.axis('off')
            plt.suptitle("Prototypes Batch 1")
            plt.show()
    print("Prototype_loader verification completed.\n")

    # Verify task_prototypes mapping
    print("Verifying task_prototypes mapping...")
    for t in range(num_tasks):
        prototypes = task_prototypes[t]
        print(f"Task {t}:")
        print(f" - Number of prototypes: {prototypes.shape[0]}")
        print(f" - Prototype shape: {prototypes.shape[1:]}")
        # Optionally, visualize the first prototype of each task
        if t < 1:  # Change or remove this condition to visualize more tasks
            plt.figure(figsize=(2, 2))
            img = prototypes[0]
            if dataset_name == 'Split-MNIST':
                img = img * 0.5 + 0.5
                img = img.squeeze().numpy()
                plt.imshow(img, cmap='gray')
            elif dataset_name in ['Split-CIFAR100', 'TinyImageNet']:
                if dataset_name == 'Split-CIFAR100':
                    mean = np.array([0.5, 0.5, 0.5])
                    std = np.array([0.5, 0.5, 0.5])
                elif dataset_name == 'TinyImageNet':
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
                img = (img * std + mean).clip(0, 1)  # Unnormalize and clip
                plt.imshow(img)
            plt.title(f"Task {t} Prototype")
            plt.axis('off')
            plt.show()
    print("Task_prototypes mapping verification completed.\n")

    # Optionally, verify that prototypes are excluded from training data
    print("Verifying that prototypes are excluded from training data...")
    all_train_indices = set()
    for t, (train_set, val_set) in timestep_tasks.items():
        # Extract the original indices from the train_set
        # Note: random_split creates Subset objects with subset.indices
        subset = train_set
        if isinstance(subset, Subset):
            subset_indices = subset.indices
        else:
            subset_indices = []
        all_train_indices.update(subset_indices)

    # Check that none of the prototype indices are in the training indices
    intersection = all_train_indices.intersection(prototype_indices)
    if len(intersection) == 0:
        print("Success: No prototype indices found in the training data.")
    else:
        print(f"Error: Found {len(intersection)} prototype indices in the training data.")
    print("Verification of prototype exclusion completed.\n")

    # Summary of tasks and prototypes
    print("Summary of tasks and prototypes:")
    for t in range(num_tasks):
        classes = timestep_task_classes[t]
        num_classes_task = len(classes)
        print(f"Task {t}: {num_classes_task} classes")
        # Optionally, list class names or indices
        class_names = [task_metadata[t][idx] for idx in range(num_classes_task)]
        print(f" - Classes: {class_names}\n")

    print("All verifications completed successfully.")
