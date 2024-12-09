import os
import wandb


changes = {"NAME-DATASET": "Split-CIFAR100",
            "FROOZE_BKBN": "False"}

train = ["LwF", "SI", "EWC"]
devices = [0, 1, 2]

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

# execute the training script with its corresponding configuration file
for model, dev in zip(train, devices):
    with open(f"configs/{config_files[model]}", "r") as f:
        content = f.read()
        for key, value in changes.items():
            content = content.replace(key, value)
    
    with open(f"configs/{config_files[model]}-temp.py", "w") as f:
        f.write(content)
        
    os.system(f"sbatch ztrain {training_files[model]} configs/{config_files[model]}-temp.py")
    
from time import sleep
sleep(10)
for model in train:
    os.remove(f"configs/{config_files[model]}-temp.py")
    