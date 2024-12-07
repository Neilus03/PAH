import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import torch
from torch import utils
import numpy as np
from hypernetwork import HyperCMTL_seq_simple_2d
from utils import setup_dataset, evaluate_model_2d, distillation_output_loss, get_batch_acc, evaluate_model_2d, test_evaluate_2d, training_plot, evaluate_model_2d
import wandb
import time
import os
from tqdm import tqdm
import logging
from config import config

class Logger:
    def __init__(self, results_dir):
        logging.basicConfig(filename=os.path.join(results_dir, 'training.log'), 
                            level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logg = logging.getLogger()
    
    def log(self, message):
        self.logg.info(message)
        print(message)

os.environ["CUDA_VISIBLE_DEVICES"]="4"

# Device setup
device = torch.device(config['misc']['device'] if torch.cuda.is_available() else "cpu")

# Define the objective function for Optuna optimization
def objective(trial):
    # Sample hyperparameters from Optuna
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)  # Learning rate (log scale)
    temperature = trial.suggest_float('temperature', 0.5, 3.0)  # Temperature for distillation loss
    stability = trial.suggest_float('stability', 0.0, 5.0)  # Stability parameter for soft loss
    task_head_projection_size = trial.suggest_int('task_head_projection_size', 128, 1024, step=128)  # Task head size
    hyper_hidden_features = trial.suggest_int('hyper_hidden_features', 128, 512, step=64)  # Hypernetwork hidden features
    hyper_hidden_layers = trial.suggest_int('hyper_hidden_layers', 2, 6)  # Hypernetwork layers

    # Dataset and training setup
    dataset = "Split-CIFAR100"  # Set your dataset between "Split-MNIST" or "Split-CIFAR100" or "TinyImageNet"
    NUM_TASKS = 5 if dataset=='Split-MNIST' else 10  # Set the number of tasks
    BATCH_SIZE = 128  # Set batch size
    EPOCHS_PER_TIMESTEP = 12  # Set number of epochs per timestep
    VAL_FRAC = 0.1
    TEST_FRAC = 0.1
    l2_reg = 0.0  # L2 regularization (Not used in this experiment yet) !!!
    weight_hard_loss_prototypes = 0.2
    weight_soft_loss_prototypes = 0.05
    backbone = 'resnet50'  # Backbone architecture

    os.makedirs('results', exist_ok=True)
    # num = str(len(os.listdir('results/'))).zfill(3)
    num = time.strftime("%m%d-%H%M%S")
    results_dir = 'results/' + num + '-HyperCMTL_seq_' + dataset
    os.makedirs(results_dir, exist_ok=True)

    # Initialize the logger
    logger = Logger(results_dir)

    # Log initial information
    logger.log('Starting training...')
    logger.log(f'Training hyperparameters: EPOCHS_PER_TIMESTEP={EPOCHS_PER_TIMESTEP}, lr={lr}, l2_reg={l2_reg}, temperature={temperature}, stability={stability}')
    logger.log(f'Training on device: {device}')

    # Setup the dataset
    data = setup_dataset(dataset, data_dir='./data', num_tasks=NUM_TASKS, val_frac=VAL_FRAC, test_frac=TEST_FRAC, batch_size=BATCH_SIZE)
    timestep_tasks = data['timestep_tasks']
    final_test_loader = data['final_test_loader']
    task_test_sets = data['task_test_sets']
    task_metadata = data['task_metadata']

    # Initialize the model
    model = HyperCMTL_seq_simple_2d(
        num_instances=len(task_metadata),
        backbone='resnet50',  # Using ResNet50 backbone
        task_head_projection_size=task_head_projection_size,
        task_head_num_classes=len(task_metadata[0]),
        hyper_hidden_features=hyper_hidden_features,
        hyper_hidden_layers=hyper_hidden_layers,
        device=device,
        std=0.02
    ).to(device)

    # Log the model architecture and configuration
    # logger.log(f'Model architecture: {model}')
    logger.log(f"Model initialized with backbone_config={backbone}, task_head_projection_size={task_head_projection_size}, hyper_hidden_features={hyper_hidden_features}, hyper_hidden_layers={hyper_hidden_layers}")
    
    # Initialize the previous model
    previous_model = None

    # Initialize optimizer and loss function
    opt = torch.optim.AdamW(model.get_optimizer_list())
    loss_fn = torch.nn.CrossEntropyLoss()

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

    with wandb.init(project='HyperCMTL', name=f'HyperCMTL_seq-learned_emb-{dataset}-{backbone}') as run:
        wandb.config.update(config)

        # outer loop over each task, in sequence
        for t, (task_train, task_val) in timestep_tasks.items():
            logger.log(f"Training on task id: {t}  (classification between: {task_metadata[t]})")

            #if verbose:
                #inspect_task(task_train=task_train, task_metadata=task_metadata)

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
                    pred, pred_prototypes = model(x, task_id)
                    y_prototypes = torch.arange(len(task_metadata[int(task_id)]), device=device, dtype=torch.int64)
                    
                    hard_loss = loss_fn(pred, y)
                    prototypes_loss = loss_fn(pred_prototypes, y_prototypes)

                    #if previous model exists, calculate distillation loss
                    soft_loss = torch.tensor(0.0).to(device)
                    if previous_model is not None:
                        for old_task_id in range(t):
                            with torch.no_grad():
                        
                                old_pred, old_pred_prot = previous_model(x, old_task_id)
                            new_prev_pred, new_prev_pred_prot = model(x, old_task_id)
                            soft_loss += distillation_output_loss(new_prev_pred, old_pred, temperature).mean().to(device)
                            soft_loss += distillation_output_loss(new_prev_pred_prot, old_pred_prot, temperature).mean().to(device) * weight_soft_loss_prototypes

                    total_loss = hard_loss + stability * soft_loss + prototypes_loss * weight_hard_loss_prototypes
                    
                    total_loss.backward()
                    opt.step()

                    accuracy_batch = get_batch_acc(pred, y)
                    
                    wandb.log({'hard_loss': hard_loss.item(), 'soft_loss': (soft_loss*stability).item(), 
                            'train_loss': total_loss.item(), 'prototype_loss': prototypes_loss.item(),
                            'epoch': e, 'task_id': t, 'batch_idx': batch_idx, 'train_accuracy': accuracy_batch})

                    # track loss and accuracy:
                    epoch_train_losses.append(hard_loss.item())
                    epoch_train_accs.append(accuracy_batch)
                    epoch_soft_losses.append(soft_loss.item() if isinstance(soft_loss, torch.Tensor) else soft_loss)
                    metrics['steps_trained'] += 1

                    if show_progress:
                        # show loss/acc of this batch in progress bar:
                        progress_bar.set_description((f'E{e} batch loss:{hard_loss:.2f}, batch acc:{epoch_train_accs[-1]:>5.1%}'))

                # evaluate after each epoch on the current task's validation set:
                avg_val_loss, avg_val_acc, avg_val_loss_prot, avg_val_acc_prot = evaluate_model_2d(model, val_loader, loss_fn, device = device, task_metadata = task_metadata, task_id=t, wandb_run = run.id)

                wandb.log({'val_loss': avg_val_loss, 'val_accuracy': avg_val_acc, 'epoch': e, 'task_id': t, 'val_prot_loss': avg_val_loss_prot, 'val_prot_accuracy': avg_val_acc_prot})

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
            test_accs, test_accs_prot = test_evaluate_2d(
                                    multitask_model=model, 
                                    selected_test_sets=task_test_sets[:t+1],  
                                    task_test_sets=task_test_sets, 
                                    prev_accs = prev_test_accs,
                                    prev_accs_prot = prev_test_accs_prot,
                                    show_taskwise_accuracy=True, 
                                    baseline_taskwise_accs = None, 
                                    model_name= 'HyperCMTL_seq + LwF', 
                                    verbose=True, 
                                    batch_size=BATCH_SIZE,
                                    results_dir=results_dir,
                                    task_id=t,
                                    task_metadata=task_metadata,
                                    wandb_run = run.id
                                    )
            
            wandb.log({'mean_test_acc': np.mean(test_accs), 'task_id': t, 'mean_test_acc_prot': np.mean(test_accs_prot)})

            prev_test_accs.append(test_accs)
            prev_test_accs_prot.append(test_accs_prot)

            #store the current model as the previous model
            previous_model = model.deepcopy()
            #torch.cuda.empty_cache()

        final_avg_test_acc = np.mean(test_accs)
        logger.log(f'Final average test accuracy: {final_avg_test_acc:.2%}')
        wandb.log({'val_accuracy': final_avg_test_acc, 'epoch': e, 'task_id': t})
        wandb.summary['final_avg_test_acc'] = final_avg_test_acc

    return final_avg_test_acc  # Return the objective value to Optuna

# Create the Optuna study
study = optuna.create_study(direction='maximize')  # Maximize the validation accuracy
study.optimize(objective, n_trials=30)  # Number of trials can be adjusted

# save figures and plot for the optuna study
plot_optimization_history(study).write_image('results/optuna_study.png')
plot_param_importances(study).write_image('results/optuna_params.png')

# save the .npy file with the optimal hyperparameters found in the study 
np.save('results/optuna_hyperparameters.npy', study.best_trial.params)
study.best_trial.params["lr"]

# Print the best hyperparameters
print(f'Best trial: {study.best_trial.params}')
print(f'Best validation accuracy: {study.best_value}')
