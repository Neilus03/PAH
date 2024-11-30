import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset

def build_task_datasets(fmnist, timestep_task_classes, preprocess, VAL_FRAC,
                        TEST_FRAC, BATCH_SIZE, inspect_task=None):
    
    TEST_FRAC = 0.1
    VAL_FRAC = 0.1
    # HYPER_TRAIN_FRAC = 0.1
    MAIN_TRAIN_FRAC = 0.8
    
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
        main_task_train, task_val, task_test = random_split(task_data, [int(len(task_data) * MAIN_TRAIN_FRAC), 
                                                                                        #   int(len(task_data) * HYPER_TRAIN_FRAC), 
                                                                                          int(len(task_data) * VAL_FRAC), 
                                                                                          int(len(task_data) * TEST_FRAC)])

        # Set dataset attributes
        for dataset in (main_task_train, task_val, task_test):
            dataset.classes = task_classes
            dataset.num_classes = len(task_classes)
            dataset.class_to_idx = class_to_idx
            dataset.task_id = t

        print(f'Time {t}: Task ID {t}, {len(main_task_train)} main train, {len(task_val)} val, {len(task_test)} test')

        # Add datasets to task dictionary and test sets
        timestep_tasks[t] = (main_task_train, task_val)
        task_test_sets.append(task_test)

    # Combine all task test datasets for final evaluation
    final_test_data = ConcatDataset(task_test_sets)
    final_test_loader = DataLoader(final_test_data, batch_size=BATCH_SIZE, shuffle=True)
    print(f'Final test set size (containing all tasks): {len(final_test_data)}')

    return {
        'timestep_tasks': timestep_tasks,
        'final_test_loader': final_test_loader,
        'task_test_sets': task_test_sets
    }

