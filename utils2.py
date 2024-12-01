'''import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils as utils
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
import pandas as pd
import matplotlib as mpl
import wandb
import io
from PIL import Image

def inspect_batch(images, # batch of images as torch tensors
        labels=None,      # optional vector of ground truth label integers
        predictions=None, # optional vector/matrix of model predictions
        # display parameters:
        class_names=None, # optional list or dict of class idxs to class name strings
        title=None,       # optional title for entire plot
        # figure display/sizing params:
        center_title=True,
        max_to_show=16,
        num_cols = 4,
        scale=1,
        ):
    """accepts a batch of images as a torch tensor or list of tensors,
    and plots them in a grid for manual inspection.
    optionally, you can supply ground truth labels
    and/or model predictions, to display those as well."""

    max_to_show = min([max_to_show, len(images)]) # cap at number of images

    num_rows = int(np.ceil(max_to_show / num_cols))

    # add extra figure height if needed for captions:
    extra_height = (((labels is not None) or (predictions is not None)) * 0.2)

    fig_width = 2 * scale * num_cols
    fig_height = (2+extra_height) * scale * num_rows + ((title is not None) * 0.3)

    fig, axes = plt.subplots(num_rows, num_cols, squeeze=False, figsize=(fig_width, fig_height))
    all_axes = []
    for ax_row in axes:
        all_axes.extend(ax_row)

    if class_names is not None:
        if labels is not None:
            labels = [f'{l}:{class_names[int(l)]}' for l in labels]
        if predictions is not None:
            if len(predictions.shape) == 2:
                # probability distribution or onehot vector, so argmax it:
                predictions = predictions.argmax(dim=1)
            predictions = [f'{p}:{class_names[int(p)]}' for p in predictions]

    for b, ax in enumerate(all_axes):
        if b < max_to_show:
            # rearrange to H*W*C:
            img_p = images[b].permute([1,2,0])
            # un-normalise:
            img = (img_p - img_p.min()) / (img_p.max() - img_p.min())
            # to numpy:
            img = img.cpu().detach().numpy()

            ax.imshow(img, cmap='gray')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

            if labels is not None:
                ax.set_title(f'{labels[b]}', fontsize=10*scale**0.5)
            if predictions is not None:
                ax.set_title(f'pred: {predictions[b]}', fontsize=10*scale**0.5)
            if labels is not None and predictions is not None:
                if labels[b] == predictions[b]:
                    ### matching prediction, mark as correct:
                    mark, color = '✔', 'green'
                else:
                    mark, color = '✘', 'red'

                ax.set_title(f'label:{labels[b]}    \npred:{predictions[b]} {mark}', color=color, fontsize=8*scale**0.5)
        else:
            ax.axis('off')
    if title is not None:
        if center_title:
            x, align = 0.5, 'center'
        else:
            x, align = 0, 'left'
        fig.suptitle(title, fontsize=14*scale**0.5, x=x, horizontalalignment=align)
    fig.tight_layout()
    plt.show()
    plt.close()


# quick function for displaying the classes of a task:
def inspect_task(task_data, title=None):
    num_task_classes = len(task_data.classes)
    task_classes = tuple([str(c) for c in task_data.classes])

    class_image_examples = [[batch[0] for batch in task_data if batch[1]==c][0] for c in range(num_task_classes)]
    inspect_batch(class_image_examples, labels=task_classes, scale=0.7, num_cols=num_task_classes, title=title, center_title=False)

def training_plot(metrics,
      title=None, # optional figure title
      alpha=0.05, # smoothing parameter for train loss
      baselines=None, # optional list, or named dict, of baseline accuracies to compare to
      show_epochs=False,    # display boundary lines between epochs
      show_timesteps=False, # display discontinuities between CL timesteps
      results_dir=""
      ):

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
    """calculates accuracy over a batch as a float
    given predicted logits 'pred' and integer targets 'y'"""
    return (pred.argmax(axis=1) == y).float().mean().item()

def evaluate_model(multitask_model: nn.Module,  # trained model capable of multi-task classification
                   val_loader: utils.data.DataLoader,  # task-specific data to evaluate on
                   loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                   device = 'cuda'
                  ):
    """runs model on entirety of validation loader,
    with specified loss and accuracy functions,
    and returns average loss/acc over all batches"""
    with torch.no_grad():
        batch_val_losses, batch_val_accs = [], []

        for batch in val_loader:
            vx, vy, task_ids = batch
            vx, vy = vx.to(device), vy.to(device)

            idx_sample_class_0 = (vy == 0).nonzero()[0].item()
            idx_sample_class_1 = (vy == 1).nonzero()[0].item()

            input_hypernet = [idx_sample_class_0, idx_sample_class_1]

            input_hypernet_tensor = torch.tensor(input_hypernet, device=device)
            vy_no_hyper = vy[~torch.isin(torch.arange(vy.size(0), device=device), input_hypernet_tensor)]

            vpred = multitask_model(vx, input_hypernet)
            val_loss = loss_fn(vpred, vy_no_hyper)
            val_acc = get_batch_acc(vpred, vy_no_hyper)

            batch_val_losses.append(val_loss.item())
            batch_val_accs.append(val_acc)
    return np.mean(batch_val_losses), np.mean(batch_val_accs)


def test_evaluate(multitask_model: nn.Module, 
                  selected_test_sets,  
                  task_test_sets, 
                  prev_accs = None,
                  show_taskwise_accuracy=True, 
                  baseline_taskwise_accs = None, 
                  model_name: str='', 
                  verbose=False, 
                  batch_size=64,
                  results_dir="",
                  task_id=0
                 ):
    if verbose:
        print(f'{model_name} evaluation on test set of all tasks:'.capitalize())

    task_test_losses = []
    task_test_accs = []

    for t, test_data in enumerate(selected_test_sets):
        test_loader = utils.data.DataLoader(test_data,
                                       batch_size=batch_size,
                                       shuffle=True)

        task_test_loss, task_test_acc = evaluate_model(multitask_model, test_loader)

        if verbose:
            print(f'{test_data.classes}: {task_test_acc:.2%}')
            if baseline_taskwise_accs is not None:
                print(f'(baseline: {baseline_taskwise_accs[t]:.2%})')

        task_test_losses.append(task_test_loss)
        task_test_accs.append(task_test_acc)

    avg_task_test_loss = np.mean(task_test_losses)
    avg_task_test_acc = np.mean(task_test_accs)

    if verbose:
        print(f'\n +++ AVERAGE TASK TEST ACCURACY: {avg_task_test_acc:.2%} +++ ')


    if show_taskwise_accuracy:
        bar_heights = task_test_accs + [0]*(len(task_test_sets) - len(selected_test_sets))
        # display bar plot with accuracy on each evaluation task
        plt.bar(x = range(len(task_test_sets)), height=bar_heights, zorder=1)
        plt.xticks(range(len(task_test_sets)), [','.join(task.classes) for task in task_test_sets], rotation='vertical')
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
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure to wandb
        plt.savefig(results_dir)
        img = Image.open(results_dir)
        wandb.log({f'taskwise accuracy': wandb.Image(img), 'task': task_id})

        plt.close()

    return task_test_accs


from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    # Assuming each sample in batch is a tuple (sequence, label)
    sequences, labels = zip(*batch)
    
    # Pad the sequences to the same length (using pad_sequence)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Convert labels into a tensor (assuming labels are scalars)
    labels = torch.tensor(labels)
    
    return padded_sequences, labels

def build_task_datasets(
    fmnist,
    timestep_task_classes,
    preprocess,
    VAL_FRAC,
    TEST_FRAC,
    BATCH_SIZE,
    inspect_task=None
):
    """
    Builds task datasets for each timestep with train/validation/test splits and combines test datasets for final evaluation.
    
    Args:
        fmnist: Dataset object containing data, labels, and class mappings.
        timestep_task_classes (dict): Mapping of timestep to a list of classes for the task.
        preprocess (callable): Function to preprocess images.
        VAL_FRAC (float): Fraction of data for validation split.
        TEST_FRAC (float): Fraction of data for test split.
        BATCH_SIZE (int): Batch size for DataLoaders.
        inspect_task (callable, optional): Function to inspect task datasets.
    
    Returns:
        dict: Contains the following:
            - 'timestep_tasks': Train/validation datasets for each timestep.
            - 'final_test_loader': DataLoader for the combined test datasets of all tasks.
            - 'joint_train_loader': DataLoader for the combined training datasets of all tasks.
    """
    timestep_tasks = {}
    timestep_loaders = {}
    task_test_sets = []

    for t, task_classes in timestep_task_classes.items():
        # Map original labels to task-specific labels
        task_class_labels = [fmnist.class_to_idx[cl] for cl in task_classes]
        task_datapoint_idxs = [i for i, label in enumerate(fmnist.targets) if label in task_class_labels]
        task_datapoints = [fmnist[idx] for idx in task_datapoint_idxs]

        class_to_idx = {task_classes[i]: i for i in range(len(task_classes))}
        task_images = [preprocess(img) for (img, label) in task_datapoints]
        task_labels = [class_to_idx[fmnist.classes[label]] for (img, label) in task_datapoints]
        task_ids = [t] * len(task_datapoints)

        task_image_tensor = torch.stack(task_images)
        task_label_tensor = torch.tensor(task_labels, dtype=torch.long)
        task_id_tensor = torch.tensor(task_ids, dtype=torch.long)

        task_data = TensorDataset(task_image_tensor, task_label_tensor, task_id_tensor)

        # Train/validation/test split
        train_frac = 1.0 - VAL_FRAC - TEST_FRAC
        task_train, task_val, task_test = random_split(task_data, [int(train_frac * len(task_data)),
                                                                   int(VAL_FRAC * len(task_data)),
                                                                   int(TEST_FRAC * len(task_data))])

        # Set dataset attributes
        for dataset in (task_train, task_val, task_test):
            dataset.classes = task_classes
            dataset.num_classes = len(task_classes)
            dataset.class_to_idx = class_to_idx
            dataset.task_id = t

        # Inspect samples if inspection function is provided
        if inspect_task:
            print(f'Time {t}: Task ID {t}, {len(task_train)} train, {len(task_val)} validation, {len(task_test)} test')
            inspect_task(task_train)

        # Add datasets to task dictionary and test sets
        timestep_tasks[t] = (task_train, task_val)
        task_test_sets.append(task_test)

    # Combine all task test datasets for final evaluation
    final_test_data = ConcatDataset(task_test_sets)
    final_test_loader = DataLoader(final_test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f'Final test set size (containing all tasks): {len(final_test_data)}')

    # Combine all task training datasets for joint training
    joint_train_data = ConcatDataset([train for train, _ in timestep_tasks.values()])
    joint_train_loader = DataLoader(joint_train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f'Joint training set size (containing all tasks): {len(joint_train_data)}')

    return {
        'timestep_tasks': timestep_tasks,
        'final_test_loader': final_test_loader,
        'joint_train_loader': joint_train_loader,
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
        print(message)'''