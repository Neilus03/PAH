import os
from time import sleep
import wandb

all_datasets = ["Split-CIFAR100"]
freeze = ["False", "True"]
models = ["Hyper", "Hyper_prot", "Hyper2d_i", "Hyper2d"]

all_datasets = ["Split-CIFAR100"]
freeze = ["False"]
models = ["Hyper2d"]

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

# do squeue -u mpilligua and check how many jobs are running
ret = int(os.popen("squeue -u mpilligua | wc -l").read())
for fr in freeze:
    for dataset in all_datasets:
        changes = {"NAME-DATASET": dataset, # Split-MNIST, Split-CIFAR100, TinyImageNet
                    "FROOZE_BKBN": fr}
        
        for model in models:
            ret = int(os.popen("squeue -u mpilligua | wc -l").read())
            while ret > 6:
                ret = int(os.popen("squeue -u mpilligua | wc -l").read())
                print(f"Jobs running: {ret}", end="\r")
                os.system("sleep 100")

            print(f"Jobs running: {ret} - Starting new job with {model} and {dataset} and {fr}")
            with open(f"configs/{config_files[model]}", "r") as f:
                content = f.read()
                for key, value in changes.items():
                    content = content.replace(key, value)
            
            with open(f"configs/{config_files[model]}-temp.py", "w") as f:
                f.write(content)
                
            print(f"sbatch ztrain {training_files[model]} configs/{config_files[model]}-temp.py")
            os.system(f"sbatch ztrain {training_files[model]} configs/{config_files[model]}-temp.py")
            
            sleep(10)
            os.remove(f"configs/{config_files[model]}-temp.py")
    

# # train = ["LwF", "SI", "EWC"]
# devices = [0, 1, 2]


# # execute the training script with its corresponding configuration file
# for model, dev in zip(models, devices):
#     with open(f"configs/{config_files[model]}", "r") as f:
#         content = f.read()
#         for key, value in changes.items():
#             content = content.replace(key, value)
    
#     with open(f"configs/{config_files[model]}-temp.py", "w") as f:
#         f.write(content)
        
#     os.system(f"sbatch ztrain {training_files[model]} configs/{config_files[model]}-temp.py")
    
# from time import sleep
# sleep(10)
# for model in train:
#     os.remove(f"configs/{config_files[model]}-temp.py")
    