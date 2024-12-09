import os
import subprocess
from time import sleep
import threading

# Define your datasets, freeze options, models, and devices
all_datasets = ["Split-MNIST", "Split-CIFAR100", "TinyImageNet"]
freeze = ["False", "True"]
models = ["EWC", "SI", "LwF"]
devices = ["0", "1", "2", "3", "4", "5"]

# Mapping of models to their corresponding training and config files
training_files = {
    "EWC": "train_baseline_EWC.py",
    "SI": "train_baseline_SI.py",
    "LwF": "train_baseline_LwF.py",
    "Hyper": "train_hyper.py",
    "Hyper_prot": "train_hyper_prot.py",
    "Hyper2d_i": "train_hyper2d.py",
    "Hyper2d": "train_hyper2d.py"
}

config_files = {
    "EWC": "baseline_ewc.py",
    "SI": "baseline_si.py",
    "LwF": "baseline_lwf.py",
    "Hyper": "hyper.py",
    "Hyper_prot": "hyper_prot.py",
    "Hyper2d_i": "hyper2d_i.py",
    "Hyper2d": "hyper2d.py"
}

# Set to keep track of busy GPUs
busy_gpus = set()
gpu_lock = threading.Lock()

def gpu_is_free(device):
    """
    Check if a GPU is free by querying its compute processes.
    """
    cmd = f"nvidia-smi -i {device} --query-compute-apps=pid --format=csv,noheader"
    try:
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        return len(result) == 0
    except subprocess.CalledProcessError as e:
        print(f"Error checking GPU {device}: {e}")
        return False

def wait_for_free_gpu():
    """
    Wait until a GPU becomes free and return its ID.
    """
    while True:
        for dev in devices:
            with gpu_lock:
                if dev in busy_gpus:
                    continue
            print(f"Checking GPU {dev}...")
            if gpu_is_free(dev):
                print(f"GPU {dev} is free.")
                with gpu_lock:
                    busy_gpus.add(dev)
                return dev
        print("No free GPUs found. Waiting for 60 seconds...")
        sleep(60)

def monitor_gpus():
    """
    Continuously monitor the status of busy GPUs and update the busy_gpus set.
    """
    while True:
        to_remove = set()
        with gpu_lock:
            current_busy_gpus = list(busy_gpus)
        for dev in current_busy_gpus:
            if gpu_is_free(dev):
                print(f"GPU {dev} is now free.")
                to_remove.add(dev)
        if to_remove:
            with gpu_lock:
                busy_gpus.difference_update(to_remove)
        sleep(30)

# Start the GPU monitoring thread
monitor_thread = threading.Thread(target=monitor_gpus, daemon=True)
monitor_thread.start()

# Iterate over all combinations of freeze options, datasets, and models
for fr in freeze:
    for dataset in all_datasets:
        changes = {
            "NAME-DATASET": dataset,
            "FREEZE_BKBN": fr,
            "NUM_TASKS_VAR": "10" if dataset == "Split-CIFAR100" else "20"
        }

        for model in models:
            # Wait for a free GPU and get its ID
            selected_gpu = wait_for_free_gpu()
            print(f"Starting new job with model={model}, dataset={dataset}, freeze={fr} on GPU={selected_gpu}")

            # Read and modify the config file
            config_path = f"configs/{config_files[model]}"
            with open(config_path, "r") as f:
                content = f.read()
                for key, value in changes.items():
                    content = content.replace(key, value)

            # Write the modified config to a temporary file
            temp_config = f"configs/{config_files[model]}-temp.py"
            with open(temp_config, "w") as f:
                f.write(content)

            # Construct the command to run the training script
            cmd = (
                f"CUDA_VISIBLE_DEVICES={selected_gpu} "
                f"nohup python {training_files[model]} {temp_config} "
                f"> {temp_config}.log 2>&1 &"
            )
            print(f"Executing command: {cmd}")

            # Launch the training script in the background
            subprocess.Popen(cmd, shell=True)

            # Brief pause to allow the system to recognize the new job
            sleep(15)

            # Remove the temporary config file
            os.remove(temp_config)

print("All jobs have been submitted.")
