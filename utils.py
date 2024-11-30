import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils as utils
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
#from tinyimagenet import TinyImageNet
import pandas as pd
import matplotlib as mpl
import wandb
import io
from PIL import Image
import pdb

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
                   device = 'cuda'
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
                  task_metadata=None
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
        task_test_loss, task_test_acc = evaluate_model(multitask_model, test_loader)

        if verbose:
            print(f'{task_metadata[t]}: {task_test_acc:.2%}')
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
        dataset = datasets.MNIST(root=data_dir, train=True, download=True)
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
        dataset = datasets.CIFAR100(root=data_dir, train=True, download=True)
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
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200', 'train'))
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

    # Process tasks
    for t, task_classes in timestep_task_classes.items():
        if dataset_name == 'Split-MNIST':
            task_indices = [i for i, label in enumerate(dataset.targets) if label in task_classes]
            task_images = [Image.fromarray(dataset.data[i].numpy(), mode='L') for i in task_indices]
            task_labels = [label for i, label in enumerate(dataset.targets) if label in task_classes]

        elif dataset_name == 'Split-CIFAR100':
            task_indices = [i for i, label in enumerate(dataset.targets) if label in task_classes]
            task_images = [Image.fromarray(dataset.data[i]) for i in task_indices]
            task_labels = [label for i, label in enumerate(dataset.targets) if label in task_classes]

        elif dataset_name == 'TinyImageNet':
            task_indices = [i for i, (_, label) in enumerate(dataset.samples) if label in task_classes]
            task_images = [dataset[i][0] for i in task_indices]
            task_labels = [label for i, (_, label) in enumerate(dataset.samples) if label in task_classes]

        # Map old labels to 0-based labels for the task
        class_to_idx = {orig: idx for idx, orig in enumerate(task_classes)}
        task_labels = [class_to_idx[int(label)] for label in task_labels]


        # Create tensors
        task_images_tensor = torch.stack([preprocess(img) for img in task_images])
        task_labels_tensor = torch.tensor(task_labels, dtype=torch.long)
        task_ids_tensor = torch.full((len(task_labels_tensor),), t, dtype=torch.long)

        # TensorDataset
        task_dataset = TensorDataset(task_images_tensor, task_labels_tensor, task_ids_tensor)

        # Train/Validation/Test split
        train_size = int((1 - val_frac - test_frac) * len(task_dataset))
        val_size = int(val_frac * len(task_dataset))
        test_size = len(task_dataset) - train_size - val_size
        train_set, val_set, test_set = random_split(task_dataset, [train_size, val_size, test_size])

        # Store datasets and metadata
        timestep_tasks[t] = (train_set, val_set)
        task_test_sets.append(test_set)
        if dataset_name == 'TinyImagenet':
            task_metadata[t] = {
                idx: os.path.basename(dataset.classes[orig]) for orig, idx in class_to_idx.items()
            }
        else:
            task_metadata[t] = {
                idx: dataset.classes[orig] if hasattr(dataset, 'classes') else str(orig)
                for orig, idx in class_to_idx.items()
            }

    # Final datasets
    final_test_data = ConcatDataset(task_test_sets)
    final_test_loader = DataLoader(final_test_data, batch_size=batch_size, shuffle=True)

    return {
        'timestep_tasks': timestep_tasks,
        'final_test_loader': final_test_loader,
        'task_metadata': task_metadata,
        'task_test_sets': task_test_sets
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