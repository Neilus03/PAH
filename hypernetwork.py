import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from copy import deepcopy

import numpy as np
# import random
# import numpy as np

# from collections import OrderedDict
from torchmeta.modules import MetaSequential, MetaLinear

from metamodules import FCBlock, BatchLinear, HyperNetwork, get_subdict, HyperNetwork_seq
from torchmeta.modules import MetaModule
from copy import deepcopy

from backbones import ResNet50, MobileNetV2, EfficientNetB0
import random

torch.manual_seed(69)
np.random.seed(69)
random.seed(69)
torch.cuda.manual_seed_all(69)
torch.cuda.manual_seed(69)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

backbone_dict = {
    'resnet50': ResNet50,
    'mobilenetv2': MobileNetV2,
    'efficientnetb0': EfficientNetB0
}

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

class HyperCMTL(nn.Module):
    """
    Hypernetwork-based Conditional Multi-Task Learning (HyperCMTL) model.

    This model combines a convolutional backbone, a task-specific head, and a hypernetwork
    to dynamically generate parameters for task-specific learning. It is designed for
    applications requiring task conditioning, such as meta-learning or multi-task learning.

    Args:
        num_instances (int): Number of task instances to support (e.g., number of tasks).
        device (str, optional): Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
        std (float, optional): Standard deviation for initializing the task embeddings. Default is 0.01.

    Attributes:
        num_instances (int): Number of task instances.
        device (torch.device): Device for computation.
        std (float): Standard deviation for embedding initialization.
        backbone (ConvBackbone): Convolutional network for feature extraction.
        task_head (TaskHead): Fully connected network for task-specific classification.
        hypernet (HyperNetwork): Hypernetwork to generate parameters for the task head.
        hyper_emb (nn.Embedding): Task-specific embeddings used as input to the hypernetwork.
    """
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device=device,
                 channels=1,
                 img_size=[32, 32],
                 std=0.01):
        super().__init__()

        self.num_instances = num_instances
        self.backbone_name = backbone
        self.task_head_projection_size = task_head_projection_size
        self.task_head_num_classes = task_head_num_classes
        self.hyper_hidden_features = hyper_hidden_features
        self.hyper_hidden_layers = hyper_hidden_layers
        self.device = device
        self.channels = channels
        self.img_size = img_size
        self.std = std

        # Backbone
        '''self.backbone = ConvBackbone(layers=backbone_layers,
                                     input_size=(channels, img_size[0], img_size[1]),
                                     device=device)
        '''
        if backbone in backbone_dict:
            self.backbone = backbone_dict[self.backbone_name](device=device, pretrained=True)
        else: 
            raise ValueError(f"Backbone {backbone} is not supported.")
        

        # Task head
        self.task_head = TaskHead(input_size=self.backbone.num_features,
                                  projection_size=task_head_projection_size,
                                  num_classes=task_head_num_classes,
                                  dropout=0.5,
                                  device=device)

        # Hypernetwork
        self.backbone_emb_size = self.backbone.num_features
        self.hn_in = 64  # Input size for hypernetwork embedding
        self.hypernet = HyperNetwork(hyper_in_features=self.hn_in,
                                     hyper_hidden_layers=hyper_hidden_layers,
                                     hyper_hidden_features=hyper_hidden_features,
                                     hypo_module=self.task_head,
                                     activation='relu')



        self.hyper_emb = nn.Embedding(self.num_instances, self.hn_in)
        nn.init.normal_(self.hyper_emb.weight, mean=0, std=std)
        
    def get_params(self, task_idx):
        z = self.hyper_emb(torch.LongTensor([task_idx]).to(self.device))
        return self.hypernet(z)


    def forward(self, support_set, task_idx, **kwargs):
        params = self.get_params(task_idx)
        # # print("after get params", params)
        backbone_out = self.backbone(support_set)
        task_head_out = self.task_head(backbone_out, params=params)
        
        return task_head_out.squeeze(0)
    
    def deepcopy(self):
        new_model = HyperCMTL(num_instances=self.num_instances,
                    backbone=self.backbone_name,
                    task_head_projection_size=self.task_head_projection_size,
                    task_head_num_classes=self.task_head_num_classes,
                    hyper_hidden_features=self.hyper_hidden_features,
                    hyper_hidden_layers=self.hyper_hidden_layers,
                    device=self.device,
                    channels=self.channels,
                    img_size=self.img_size, 
                    std=self.std)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device=self.device)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.hyper_emb.parameters(), 'lr': 1e-3})
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        # print("optimizer_list", optimizer_list)
        return optimizer_list



class HyperCMTL_all(nn.Module):
    """
    Hypernetwork-based Conditional Multi-Task Learning (HyperCMTL) model.

    This model combines a convolutional backbone, a task-specific head, and a hypernetwork
    to dynamically generate parameters for task-specific learning. It is designed for
    applications requiring task conditioning, such as meta-learning or multi-task learning.

    Args:
        num_instances (int): Number of task instances to support (e.g., number of tasks).
        device (str, optional): Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
        std (float, optional): Standard deviation for initializing the task embeddings. Default is 0.01.

    Attributes:
        num_instances (int): Number of task instances.
        device (torch.device): Device for computation.
        std (float): Standard deviation for embedding initialization.
        backbone (ConvBackbone): Convolutional network for feature extraction.
        task_head (TaskHead): Fully connected network for task-specific classification.
        hypernet (HyperNetwork): Hypernetwork to generate parameters for the task head.
        hyper_emb (nn.Embedding): Task-specific embeddings used as input to the hypernetwork.
    """
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device=device,
                 channels=1,
                 img_size=[32, 32],
                 std=0.01):
        super().__init__()

        self.num_instances = num_instances
        self.backbone_name = backbone
        self.task_head_projection_size = task_head_projection_size
        self.task_head_num_classes = task_head_num_classes
        self.hyper_hidden_features = hyper_hidden_features
        self.hyper_hidden_layers = hyper_hidden_layers
        self.device = device
        self.channels = channels
        self.img_size = img_size
        self.std = std

        # Backbone
        '''self.backbone = ConvBackbone(layers=backbone_layers,
                                     input_size=(channels, img_size[0], img_size[1]),
                                     device=device)
        '''
        # if backbone in backbone_dict:
        #     self.backbone = backbone_dict[self.backbone_name](device=device, pretrained=True)
        # else: 
        #     raise ValueError(f"Backbone {backbone} is not supported.")
        
        self.backbone = Backbone(self.backbone_name, device=device, pretrained=True)

        # Task head
        self.task_head = TaskHead(input_size=self.backbone.num_features,
                                  projection_size=task_head_projection_size,
                                  num_classes=task_head_num_classes,
                                  dropout=0.5,
                                  device=device)

        # Hypernetwork
        self.backbone_emb_size = self.backbone.num_features
        self.hn_in = 64  # Input size for hypernetwork embedding
        self.hypernet_head = HyperNetwork(hyper_in_features=self.hn_in,
                                     hyper_hidden_layers=hyper_hidden_layers,
                                     hyper_hidden_features=hyper_hidden_features,
                                     hypo_module=self.task_head,
                                     activation='relu')
        
        self.hypernet_backbone = HyperNetwork(hyper_in_features=self.hn_in,
                                        hyper_hidden_layers=hyper_hidden_layers,
                                        hyper_hidden_features=hyper_hidden_features,
                                        hypo_module=self.backbone,
                                        activation='relu')

        self.hyper_emb = nn.Embedding(self.num_instances, self.hn_in)
        nn.init.normal_(self.hyper_emb.weight, mean=0, std=std)
        
        self.hyper_emb_backbone = nn.Embedding(self.num_instances, self.hn_in)
        nn.init.normal_(self.hyper_emb_backbone.weight, mean=0, std=std)

    def get_params_head(self, task_idx):
        z = self.hyper_emb_head(torch.LongTensor([task_idx]).to(self.device))
        return self.hypernet_head(z)
    
    def get_params_backbone(self, task_idx):
        z = self.hyper_emb_backbone(torch.LongTensor([task_idx]).to(self.device))
        return self.hypernet_backbone(z)

    def forward(self, support_set, task_idx, **kwargs):
        params_backbone = self.get_params_backbone(task_idx)
        backbone_out = self.backbone(support_set, params=params_backbone)
        
        params_head = self.get_params_head(task_idx)
        task_head_out = self.task_head(backbone_out, params=params_head)
        return task_head_out.squeeze(0)
    
    def deepcopy(self):
        new_model = HyperCMTL(num_instances=self.num_instances,
                    backbone=self.backbone_name,
                    task_head_projection_size=self.task_head_projection_size,
                    task_head_num_classes=self.task_head_num_classes,
                    hyper_hidden_features=self.hyper_hidden_features,
                    hyper_hidden_layers=self.hyper_hidden_layers,
                    device=self.device,
                    channels=self.channels,
                    img_size=self.img_size, 
                    std=self.std)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device=self.device)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.hyper_emb_head.parameters(), 'lr': 1e-3})
        optimizer_list.append({'params': self.hyper_emb_backbone.parameters(), 'lr': 1e-3})

        optimizer_list.extend(self.hypernet_head.get_optimizer_list())
        optimizer_list.extend(self.hypernet_backbone.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        return optimizer_list




class HyperCMTL_seq(nn.Module):
    """
    Hypernetwork-based Conditional Multi-Task Learning (HyperCMTL) model.

    This model combines a convolutional backbone, a task-specific head, and a hypernetwork
    to dynamically generate parameters for task-specific learning. It is designed for
    applications requiring task conditioning, such as meta-learning or multi-task learning.

    Args:
        num_instances (int): Number of task instances to support (e.g., number of tasks).
        device (str, optional): Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
        std (float, optional): Standard deviation for initializing the task embeddings. Default is 0.01.

    Attributes:
        num_instances (int): Number of task instances.
        device (torch.device): Device for computation.
        std (float): Standard deviation for embedding initialization.
        backbone (ConvBackbone): Convolutional network for feature extraction.
        task_head (TaskHead): Fully connected network for task-specific classification.
        hypernet (HyperNetwork): Hypernetwork to generate parameters for the task head.
        hyper_emb (nn.Embedding): Task-specific embeddings used as input to the hypernetwork.
    """
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device=device,
                 channels=1,
                 img_size=[32, 32],
                 std=0.01):
        super().__init__()

        self.num_instances = num_instances
        self.backbone_name = backbone
        self.task_head_projection_size = task_head_projection_size
        self.task_head_num_classes = task_head_num_classes
        self.hyper_hidden_features = hyper_hidden_features
        self.hyper_hidden_layers = hyper_hidden_layers
        self.device = device
        self.channels = channels
        self.img_size = img_size
        self.std = std

        # Backbone
        '''self.backbone = ConvBackbone(layers=backbone_layers,
                                     input_size=(channels, img_size[0], img_size[1]),
                                     device=device)
        '''
        if backbone in backbone_dict:
            self.backbone = backbone_dict[self.backbone_name](device=device, pretrained=True)
        else: 
            raise ValueError(f"Backbone {backbone} is not supported.")
        

        # Task head
        self.task_head = TaskHead(input_size=self.backbone.num_features,
                                  projection_size=task_head_projection_size,
                                  num_classes=task_head_num_classes,
                                  dropout=0.5,
                                  device=device)

        # Hypernetwork
        self.backbone_emb_size = self.backbone.num_features
        self.hyper_emb = nn.Embedding(self.num_instances, 256)
        nn.init.normal_(self.hyper_emb.weight, mean=0, std=std)
        
        
        self.hn_in = 256*2  # Input size for hypernetwork embedding
        self.hypernet = HyperNetwork_seq(hyper_in_features=self.hn_in,
                                     hyper_hidden_layers=hyper_hidden_layers,
                                     hyper_hidden_features=hyper_hidden_features,
                                     hypo_module=self.task_head,
                                     activation='relu')


        self.reduce_backbone = nn.Linear(2048, 256)
        
    def get_params(self, task_idx, backbone_out):
        z = self.hyper_emb(torch.LongTensor([task_idx]).to(self.device))

        # z = z.repeat(backbone_out.size(0), 1)
        # # print("z", z.size())

        backbone_out = torch.mean(self.reduce_backbone(backbone_out), dim=0).unsqueeze(0)
        input_hyp = torch.cat((z, backbone_out), dim=1)
        # # print("input_hyp", input_hyp.size())
        return self.hypernet(input_hyp)


    def forward(self, support_set, task_idx, **kwargs):
        # # print("after get params", params)
        backbone_out = self.backbone(support_set)
        # # print("backbone_out", backbone_out.size())
        params = self.get_params(task_idx, backbone_out)
        # # print("backbone_out", backbone_out.size())
        task_head_out = self.task_head(backbone_out, params=params)
        
        # # print("task_head_out", task_head_out.size())
        return task_head_out.squeeze(0)
    
    def deepcopy(self):
        new_model = HyperCMTL_seq(num_instances=self.num_instances,
                    backbone=self.backbone_name,
                    task_head_projection_size=self.task_head_projection_size,
                    task_head_num_classes=self.task_head_num_classes,
                    hyper_hidden_features=self.hyper_hidden_features,
                    hyper_hidden_layers=self.hyper_hidden_layers,
                    device=self.device,
                    channels=self.channels,
                    img_size=self.img_size, 
                    std=self.std)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device=self.device)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.hyper_emb.parameters(), 'lr': 1e-3})
        optimizer_list.append({'params': self.reduce_backbone.parameters(), 'lr': 1e-3})
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        # print("optimizer_list", optimizer_list)
        return optimizer_list



class HyperCMTL_seq_simple(nn.Module):
    """
    Hypernetwork-based Conditional Multi-Task Learning (HyperCMTL) model.

    This model combines a convolutional backbone, a task-specific head, and a hypernetwork
    to dynamically generate parameters for task-specific learning. It is designed for
    applications requiring task conditioning, such as meta-learning or multi-task learning.

    Args:
        num_instances (int): Number of task instances to support (e.g., number of tasks).
        device (str, optional): Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
        std (float, optional): Standard deviation for initializing the task embeddings. Default is 0.01.

    Attributes:
        num_instances (int): Number of task instances.
        device (torch.device): Device for computation.
        std (float): Standard deviation for embedding initialization.
        backbone (ConvBackbone): Convolutional network for feature extraction.
        task_head (TaskHead): Fully connected network for task-specific classification.
        hypernet (HyperNetwork): Hypernetwork to generate parameters for the task head.
        hyper_emb (nn.Embedding): Task-specific embeddings used as input to the hypernetwork.
    """
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device=device,
                 channels=1,
                 img_size=[32, 32],
                 std=0.01):
        super().__init__()

        self.num_instances = num_instances
        self.backbone_name = backbone
        self.task_head_projection_size = task_head_projection_size
        self.task_head_num_classes = task_head_num_classes
        self.hyper_hidden_features = hyper_hidden_features
        self.hyper_hidden_layers = hyper_hidden_layers
        self.device = device
        self.channels = channels
        self.img_size = img_size
        self.std = std

        # Backbone
        '''self.backbone = ConvBackbone(layers=backbone_layers,
                                     input_size=(channels, img_size[0], img_size[1]),
                                     device=device)
        '''
        if backbone in backbone_dict:
            self.backbone = backbone_dict[self.backbone_name](device=device, pretrained=True)
        else: 
            raise ValueError(f"Backbone {backbone} is not supported.")
        

        # freeze the backbone 
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Task head
        self.task_head = TaskHead_simple(input_size=self.backbone.num_features,
                                  projection_size=task_head_projection_size,
                                  num_classes=task_head_num_classes,
                                  dropout=0.5,
                                  device=device)

        # Hypernetwork
        self.backbone_emb_size = self.backbone.num_features
        self.hyper_emb = nn.Embedding(self.num_instances, 4096)
        nn.init.normal_(self.hyper_emb.weight, mean=0, std=std)
        

        
        self.hn_in = 4096
        self.hypernet = HyperNetwork_seq(hyper_in_features=self.hn_in,
                                     hyper_hidden_layers=6,
                                     hyper_hidden_features=1024,
                                     hypo_module=self.task_head,
                                     activation='relu')



        # self.reduce_backbone = nn.Linear(2048, 256)
        
    def get_params(self, task_idx, backbone_out):
        z = self.hyper_emb(torch.LongTensor([task_idx]).to(self.device))

        # z = z.repeat(backbone_out.size(0), 1)
        # # print("z", z.size())

        # backbone_out = torch.mean(self.reduce_backbone(backbone_out), dim=0).unsqueeze(0)
        # backbone_out = torch.mean(backbone_out, dim=0).unsqueeze(0)
        # input_hyp = torch.cat((z, backbone_out), dim=1)
        # # print("input_hyp", input_hyp.size())
        return self.hypernet(z)


    def forward(self, support_set, task_idx, **kwargs):
        # # print("after get params", params)
        backbone_out = self.backbone(support_set)
        # # print("backbone_out", backbone_out.size())
        params = self.get_params(task_idx, backbone_out)
        # # print("backbone_out", backbone_out.size())
        task_head_out = self.task_head(backbone_out, params=params)
        
        # # print("task_head_out", task_head_out.size())
        return task_head_out.squeeze(0)
    
    def deepcopy(self):
        new_model = HyperCMTL_seq_simple(num_instances=self.num_instances,
                    backbone=self.backbone_name,
                    task_head_projection_size=self.task_head_projection_size,
                    task_head_num_classes=self.task_head_num_classes,
                    hyper_hidden_features=self.hyper_hidden_features,
                    hyper_hidden_layers=self.hyper_hidden_layers,
                    device=self.device,
                    channels=self.channels,
                    img_size=self.img_size, 
                    std=self.std)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device=self.device)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.hyper_emb.parameters(), 'lr': 1e-3})
        # optimizer_list.append({'params': self.reduce_backbone.parameters(), 'lr': 1e-3})
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        # print("optimizer_list", optimizer_list)
        return optimizer_list



class HyperCMTL_seq_simple_2d(nn.Module):
    """
    Hypernetwork-based Conditional Multi-Task Learning (HyperCMTL) model.

    This model combines a convolutional backbone, a task-specific head, and a hypernetwork
    to dynamically generate parameters for task-specific learning. It is designed for
    applications requiring task conditioning, such as meta-learning or multi-task learning.

    Args:
        num_instances (int): Number of task instances to support (e.g., number of tasks).
        device (str, optional): Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
        std (float, optional): Standard deviation for initializing the task embeddings. Default is 0.01.

    Attributes:
        num_instances (int): Number of task instances.
        device (torch.device): Device for computation.
        std (float): Standard deviation for embedding initialization.
        backbone (ConvBackbone): Convolutional network for feature extraction.
        task_head (TaskHead): Fully connected network for task-specific classification.
        hypernet (HyperNetwork): Hypernetwork to generate parameters for the task head.
        hyper_emb (nn.Embedding): Task-specific embeddings used as input to the hypernetwork.
    """
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device=device,
                 channels=1,
                 img_size=[32, 32],
                 std=0.02):
        super().__init__()

        self.num_instances = num_instances
        self.backbone_name = backbone
        self.task_head_projection_size = task_head_projection_size
        self.task_head_num_classes = task_head_num_classes
        self.hyper_hidden_features = hyper_hidden_features
        self.hyper_hidden_layers = hyper_hidden_layers
        self.device = device
        self.channels = channels
        self.img_size = img_size
        self.std = std

        # Backbone
        '''self.backbone = ConvBackbone(layers=backbone_layers,
                                     input_size=(channels, img_size[0], img_size[1]),
                                     device=device)
        '''
        if backbone in backbone_dict:
            self.backbone = backbone_dict[self.backbone_name](device=device, pretrained=True)
        else: 
            raise ValueError(f"Backbone {backbone} is not supported.")
        

        # freeze the backbone 
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Task head
        self.task_head = TaskHead_simple(input_size=self.backbone.num_features,
                                  projection_size=task_head_projection_size,
                                  num_classes=task_head_num_classes,
                                  dropout=0.5,
                                  device=device)

        
        self.backbone_copy = deepcopy(self.backbone)
        self.task_head_copy = self.task_head.deepcopy()

        self.num_channels_prototype = 1
        self.size_emb = 20*20*self.num_channels_prototype*self.task_head_num_classes

        # Hypernetwork
        self.backbone_emb_size = self.backbone.num_features
        self.hyper_emb = nn.Embedding(self.num_instances, self.size_emb)
        nn.init.normal_(self.hyper_emb.weight, mean=0, std=std)
        
        self.hypernet = HyperNetwork_seq(hyper_in_features=self.size_emb,
                                     hyper_hidden_layers=6,
                                     hyper_hidden_features=1024,
                                     hypo_module=self.task_head,
                                     activation='relu')



        # self.reduce_backbone = nn.Linear(2048, 256)
        
    def get_params(self, task_idx):
        z = self.hyper_emb(torch.LongTensor([task_idx]).to(self.device))
        return self.hypernet(z), z

    def forward(self, support_set, task_idx, **kwargs):
        params, z = self.get_params(task_idx)
        
        backbone_out = self.backbone(support_set)
        task_head_out = self.task_head(backbone_out, params=params)
        
        z_2d = z.view(self.task_head_num_classes, self.num_channels_prototype, 20, 20)
        self.learned_prototyes = z_2d

        z_2d = z_2d.repeat(1, 3, 1, 1)
        with torch.no_grad():
            z_out = self.backbone(z_2d)
            z_out = self.task_head(z_out, params=params)

        return task_head_out.squeeze(0), z_out.squeeze(0)
    
    def deepcopy(self):
        new_model = HyperCMTL_seq_simple_2d(num_instances=self.num_instances,
                    backbone=self.backbone_name,
                    task_head_projection_size=self.task_head_projection_size,
                    task_head_num_classes=self.task_head_num_classes,
                    hyper_hidden_features=self.hyper_hidden_features,
                    hyper_hidden_layers=self.hyper_hidden_layers,
                    device=self.device,
                    channels=self.channels,
                    img_size=self.img_size, 
                    std=self.std)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device=self.device)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.hyper_emb.parameters(), 'lr': 1e-3})
        # optimizer_list.append({'params': self.reduce_backbone.parameters(), 'lr': 1e-3})
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        # print("optimizer_list", optimizer_list)
        return optimizer_list

    def get_prototypes(self, task_idx = None):
        if task_idx is None: 
            return self.learned_prototyes
        else: 
            z = self.hyper_emb(torch.LongTensor([task_idx]).to(self.device))
            z_2d = z.view(self.task_head_num_classes, self.num_channels_prototype, 20, 20)
            return z_2d


    def initialize_embeddings(self, embeddings, task_idx=None):
        if task_idx is None:
            self.hyper_emb.weight.data = embeddings
            return self.hyper_emb.weight.data
        
        current_weights = self.hyper_emb.weight.data
        current_weights[task_idx] = embeddings
        self.hyper_emb.weight.data = current_weights
        return self.hyper_emb.weight.data





import matplotlib.pyplot as plt
class HyperCMTL_seq_simple_2d_color(nn.Module):
    """
    Hypernetwork-based Conditional Multi-Task Learning (HyperCMTL) model.

    This model combines a convolutional backbone, a task-specific head, and a hypernetwork
    to dynamically generate parameters for task-specific learning. It is designed for
    applications requiring task conditioning, such as meta-learning or multi-task learning.

    Args:
        num_instances (int): Number of task instances to support (e.g., number of tasks).
        device (str, optional): Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
        std (float, optional): Standard deviation for initializing the task embeddings. Default is 0.01.

    Attributes:
        num_instances (int): Number of task instances.
        device (torch.device): Device for computation.
        std (float): Standard deviation for embedding initialization.
        backbone (ConvBackbone): Convolutional network for feature extraction.
        task_head (TaskHead): Fully connected network for task-specific classification.
        hypernet (HyperNetwork): Hypernetwork to generate parameters for the task head.
        hyper_emb (nn.Embedding): Task-specific embeddings used as input to the hypernetwork.
    """
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device=device,
                 channels=1,
                 img_size=[32, 32],
                 std=0.01):
        super().__init__()

        self.num_instances = num_instances
        self.backbone_name = backbone
        self.task_head_projection_size = task_head_projection_size
        self.task_head_num_classes = task_head_num_classes
        self.hyper_hidden_features = hyper_hidden_features
        self.hyper_hidden_layers = hyper_hidden_layers
        self.device = device
        self.channels = channels
        self.img_size = img_size
        self.std = std

        # Backbone
        '''self.backbone = ConvBackbone(layers=backbone_layers,
                                     input_size=(channels, img_size[0], img_size[1]),
                                     device=device)
        '''
        if backbone in backbone_dict:
            self.backbone = backbone_dict[self.backbone_name](device=device, pretrained=True)
        else: 
            raise ValueError(f"Backbone {backbone} is not supported.")
        

        # freeze the backbone 
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Task head
        self.task_head = TaskHead_simple(input_size=self.backbone.num_features,
                                  projection_size=task_head_projection_size,
                                  num_classes=task_head_num_classes,
                                  dropout=0.5,
                                  device=device)

        
        self.backbone_copy = deepcopy(self.backbone)
        self.task_head_copy = self.task_head.deepcopy()

        self.height_prototype = 11

        self.size_emb = self.height_prototype*self.height_prototype*self.task_head_num_classes*3

        # Hypernetwork
        self.backbone_emb_size = self.backbone.num_features
        self.hyper_emb = nn.Embedding(self.num_instances, self.size_emb)
        nn.init.normal_(self.hyper_emb.weight, mean=0.5, std=0)
        
        self.hypernet = HyperNetwork_seq(hyper_in_features=self.size_emb,
                                     hyper_hidden_layers=6,
                                     hyper_hidden_features=1024,
                                     hypo_module=self.task_head,
                                     activation='relu')



        # self.reduce_backbone = nn.Linear(2048, 256)
        
    def get_params(self, task_idx):
        z = self.hyper_emb(torch.LongTensor([task_idx]).to(self.device))
        return self.hypernet(z), z

    def forward(self, support_set, task_idx, **kwargs):
        params, z = self.get_params(task_idx)
        
        backbone_out = self.backbone(support_set)
        task_head_out = self.task_head(backbone_out, params=params)
        
        z_2d = z.view(self.task_head_num_classes, 3, self.height_prototype, self.height_prototype)
        self.learned_prototyes = z_2d

        # plt.imshow(z_2d[0].cpu().detach().numpy().reshape(3, 20, 20).transpose(1, 2, 0) * 255)
        # plt.savefig("z_2d.png")
        # z_2d = z_2d.repeat(1, 3, 1, 1)
        z_out = self.backbone_copy(z_2d)
        z_out = self.task_head_copy(z_out, params=params)

        return task_head_out.squeeze(0), z_out.squeeze(0)
    
    def deepcopy(self):
        new_model = HyperCMTL_seq_simple_2d_color(num_instances=self.num_instances,
                    backbone=self.backbone_name,
                    task_head_projection_size=self.task_head_projection_size,
                    task_head_num_classes=self.task_head_num_classes,
                    hyper_hidden_features=self.hyper_hidden_features,
                    hyper_hidden_layers=self.hyper_hidden_layers,
                    device=self.device,
                    channels=self.channels,
                    img_size=self.img_size, 
                    std=self.std)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device=self.device)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.hyper_emb.parameters(), 'lr': 1e-3})
        # optimizer_list.append({'params': self.reduce_backbone.parameters(), 'lr': 1e-3})
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        # print("optimizer_list", optimizer_list)
        return optimizer_list

    def get_prototypes(self, task_idx = None):
        if task_idx is None: 
            return self.learned_prototyes
        else: 
            z = self.hyper_emb(torch.LongTensor([task_idx]).to(self.device))
            z_2d = z.view(self.task_head_num_classes, 3, self.height_prototype, self.height_prototype)
            return z_2d


class HyperCMTL_prototype(nn.Module):
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device=device,
                 channels=1,
                 img_size=[32, 32],
                 std=0.01):
        super().__init__()

        self.num_instances = num_instances
        self.backbone_name = backbone
        self.task_head_projection_size = task_head_projection_size
        self.task_head_num_classes = task_head_num_classes
        self.hyper_hidden_features = hyper_hidden_features
        self.hyper_hidden_layers = hyper_hidden_layers
        self.device = device
        self.channels = channels
        self.img_size = img_size
        self.std = std


        if backbone in backbone_dict:
            self.backbone = backbone_dict[self.backbone_name](device=device, pretrained=True)
        else: 
            raise ValueError(f"Backbone {backbone} is not supported.")
        
        # Task head
        self.task_head = TaskHead(input_size=self.backbone.num_features,
                                  projection_size=task_head_projection_size,
                                  num_classes=task_head_num_classes,
                                  dropout=0.5,
                                  device=device)

        # Hypernetwork
        self.backbone_emb_size = self.backbone.num_features
        self.hn_in = 64  # Input size for hypernetwork embedding
        self.hypernet = HyperNetwork(hyper_in_features=self.hn_in*2,
                                     hyper_hidden_layers=hyper_hidden_layers,
                                     hyper_hidden_features=hyper_hidden_features,
                                     hypo_module=self.task_head,
                                     activation='relu')

        # self.hyper_emb = nn.Embedding(self.num_instances, self.hn_in)
    
        self.hyper_emb = nn.Sequential(
            nn.Linear(self.backbone_emb_size*self.task_head_num_classes, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.hn_in),
            nn.ReLU()
        )
        self.learnt_emb = nn.Embedding(self.num_instances, self.hn_in)
        nn.init.normal_(self.learnt_emb.weight, mean=0, std=std)
        # nn.init.normal_(self.hyper_emb[0].weight, mean=0, std=std)
                                 
    def get_params(self, prototype_out, task_idx):
        learnt_emb = self.learnt_emb(torch.LongTensor([task_idx]).to(self.device))
        # # print("prototype_out", prototype_out.size())
        input_hyper_reduced = self.hyper_emb(prototype_out.flatten().unsqueeze(0))

        # # print("input_hyper_reduced", input_hyper_reduced.size())
        task_emb = torch.concat((input_hyper_reduced, learnt_emb), dim=1)
        out = self.hypernet(task_emb)
        return out 
    

    def forward(self, support_set, prototypes_idx, task_id, **kwargs):
        backbone_out = self.backbone(support_set)
        prototypes_out = backbone_out[prototypes_idx, :]
        backbone_out_no_prototypes = backbone_out[~torch.isin(torch.arange(backbone_out.size(0)), prototypes_idx)]
        
        # prototype_emb = backbone_out[task_idx, :]
        # task_idx_tensor = torch.tensor(task_idx)
        # others_emb = backbone_out[~torch.isin(torch.arange(backbone_out.size(0)), task_idx_tensor)]
        
        # input_task_head = backbone_out[~torch.isin(torch.arange(backbone_out.size(0)), task_idx_tensor)]
        
        params = self.get_params(prototypes_out, task_id)
        task_head_out = self.task_head(backbone_out_no_prototypes, params=params)
        
        return task_head_out.squeeze(0)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.hyper_emb.parameters(), 'lr': 1e-3})
        optimizer_list.append({'params': self.learnt_emb.parameters(), 'lr': 1e-3})
        
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        # print("optimizer_list", optimizer_list)
        return optimizer_list
    
    def deepcopy(self, device=device):
        new_model = HyperCMTL_prototype(num_instances=self.num_instances,
                    backbone=self.backbone_name,
                    task_head_projection_size=self.task_head_projection_size,
                    task_head_num_classes=self.task_head_num_classes,
                    hyper_hidden_features=self.hyper_hidden_features,
                    hyper_hidden_layers=self.hyper_hidden_layers,
                    device=self.device,
                    channels=self.channels,
                    img_size=self.img_size, 
                    std=self.std).to(device)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device)

class HyperCMTL_prototype_attention_old(HyperCMTL):
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device=device,
                 channels=1,
                 img_size=[32, 32],
                 std=0.01):
        
        super().__init__(num_instances=num_instances,
                            backbone=backbone,
                            task_head_projection_size=task_head_projection_size,
                            task_head_num_classes=task_head_num_classes,
                            hyper_hidden_features=hyper_hidden_features,
                            hyper_hidden_layers=hyper_hidden_layers,
                            device=device,
                            channels=channels,
                            img_size=img_size,
                            std=std)

        # freeze the backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.prototype_size = 128
        self.hn_in = 256
        self.backbone_emb_size = self.backbone.num_features

        self.prototype_mlp = nn.Sequential(
            nn.Linear(self.backbone_emb_size*self.task_head_num_classes, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.prototype_size),
            nn.ReLU(),
            nn.Linear(self.prototype_size, self.backbone_emb_size)
        )
        
        self.attended_output_mlp = nn.Sequential(
            nn.Linear(self.backbone_emb_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.hn_in),
            nn.ReLU()
        ) 
        
        self.hypernet = HyperNetwork(hyper_in_features=self.hn_in,
                                        hyper_hidden_layers=hyper_hidden_layers,
                                        hyper_hidden_features=hyper_hidden_features,
                                        hypo_module=self.task_head,
                                        activation='relu')

    def get_params(self, prototype_out, backbone_out):
        flattened_backbone_out = prototype_out.flatten().unsqueeze(0)
        attention_weights = self.prototype_mlp(flattened_backbone_out)
        
        attended_output = torch.sum(attention_weights * backbone_out, dim=0)
        reduced_attended_output = self.attended_output_mlp(attended_output)
        
        out = self.hypernet(reduced_attended_output)
        return out 
    
    def forward(self, support_set, prototypes, **kwargs):
        backbone_out = self.backbone(support_set)
        # # print(prototypes.size())
        prototype_emb = self.backbone(prototypes)
        
        # prototype_emb = backbone_out[task_idx, :]
        # task_idx_tensor = torch.tensor(task_idx)
        # others_emb = backbone_out[~torch.isin(torch.arange(backbone_out.size(0)), task_idx_tensor)]
        
        # input_task_head = backbone_out[~torch.isin(torch.arange(backbone_out.size(0)), task_idx_tensor)]
        
        params = self.get_params(prototype_emb, backbone_out)
        task_head_out = self.task_head(backbone_out, params=params)
        
        return task_head_out.squeeze(0)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.prototype_mlp.parameters(), 'lr': 1e-3})
        optimizer_list.append({'params': self.attended_output_mlp.parameters(), 'lr': 1e-3})
        
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        # print("optimizer_list", optimizer_list)
        return optimizer_list

    def deepcopy(self, device=device):
        new_model = HyperCMTL_prototype_attention_old(num_instances=self.num_instances,
                    backbone=self.backbone_name,
                    task_head_projection_size=self.task_head_projection_size,
                    task_head_num_classes=self.task_head_num_classes,
                    hyper_hidden_features=self.hyper_hidden_features,
                    hyper_hidden_layers=self.hyper_hidden_layers,
                    device=self.device,
                    channels=self.channels,
                    img_size=self.img_size, 
                    std=self.std).to(device)
        new_model.load_state_dict(self.state_dict())
        return new_model

class HyperCMTL_prototype_attention(HyperCMTL):
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device=device,
                 channels=1,
                 img_size=[32, 32],
                 std=0.01):
        
        super().__init__(num_instances=num_instances,
                            backbone=backbone,
                            task_head_projection_size=task_head_projection_size,
                            task_head_num_classes=task_head_num_classes,
                            hyper_hidden_features=hyper_hidden_features,
                            hyper_hidden_layers=hyper_hidden_layers,
                            device=device,
                            channels=channels,
                            img_size=img_size,
                            std=std)

        # freeze the backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.prototype_size = 128
        self.hn_in = 256
        self.backbone_emb_size = self.backbone.num_features
        self.head_size = self.hn_in

        # self.query = nn.Linear(self.backbone_emb_size, self.hn_in)
        self.key = nn.Linear(self.backbone_emb_size, self.hn_in)
        self.value = nn.Linear(self.backbone_emb_size, self.hn_in)


        self.prototype_mlp = nn.Sequential(
            nn.Linear(self.backbone_emb_size*self.task_head_num_classes, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.hn_in),
            nn.ReLU(),
        )
        
        # self.attended_output_mlp = nn.Sequential(
        #     nn.Linear(self.backbone_emb_size, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, self.hn_in),
        #     nn.ReLU()
        # ) 
        
        self.hypernet = HyperNetwork(hyper_in_features=self.hn_in,
                                        hyper_hidden_layers=hyper_hidden_layers,
                                        hyper_hidden_features=hyper_hidden_features,
                                        hypo_module=self.task_head,
                                        activation='relu')

    def get_params(self, prototype_out, backbone_out):
        #flatten prototype_out
        prototype_out = prototype_out.flatten().unsqueeze(0)

        #query, key, value matrices
        Q = self.prototype_mlp(prototype_out)
        # print("Q", Q.size())
        K = self.key(backbone_out)
        # print("K", K.size())
        V = self.value(backbone_out)
        
        #scaled dot-product attention dot(Q, K) / sqrt(d_k)
        attention = Q @ K.transpose(-2, -1) / np.sqrt(self.head_size)
        
        #softmax function
        attention_map = torch.softmax(attention, dim=-1) #getting attention map
        
        #dot product of "softmaxed" attention and value matrices
        attention = attention_map @ V
        
        out = self.hypernet(attention)

        return out 
    
    def forward(self, support_set, prototypes, **kwargs):
        backbone_out = self.backbone(support_set)
        # # print(prototypes.size())
        prototype_emb = self.backbone(prototypes)
        # print("prototype_emb", prototype_emb.size())
        
        # prototype_emb = backbone_out[task_idx, :]
        # task_idx_tensor = torch.tensor(task_idx)
        # others_emb = backbone_out[~torch.isin(torch.arange(backbone_out.size(0)), task_idx_tensor)]
        
        # input_task_head = backbone_out[~torch.isin(torch.arange(backbone_out.size(0)), task_idx_tensor)]
        
        params = self.get_params(prototype_emb, backbone_out)
        task_head_out = self.task_head(backbone_out, params=params)
        
        return task_head_out.squeeze(0)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.prototype_mlp.parameters(), 'lr': 1e-3})
        # optimizer_list.append({'params': self.query.parameters(), 'lr': 1e-3})
        optimizer_list.append({'params': self.key.parameters(), 'lr': 1e-3})
        optimizer_list.append({'params': self.value.parameters(), 'lr': 1e-3})
        
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        # print("optimizer_list", optimizer_list)
        return optimizer_list

    def deepcopy(self, device=device):
        new_model = HyperCMTL_prototype_attention(num_instances=self.num_instances,
                    backbone=self.backbone_name,
                    task_head_projection_size=self.task_head_projection_size,
                    task_head_num_classes=self.task_head_num_classes,
                    hyper_hidden_features=self.hyper_hidden_features,
                    hyper_hidden_layers=self.hyper_hidden_layers,
                    device=self.device,
                    channels=self.channels,
                    img_size=self.img_size, 
                    std=self.std).to(device)
        new_model.load_state_dict(self.state_dict())
        return new_model






class HyperCMTL_seq_prototype_simple(nn.Module):
    """
    Hypernetwork-based Conditional Multi-Task Learning (HyperCMTL) model.

    This model combines a convolutional backbone, a task-specific head, and a hypernetwork
    to dynamically generate parameters for task-specific learning. It is designed for
    applications requiring task conditioning, such as meta-learning or multi-task learning.

    Args:
        num_instances (int): Number of task instances to support (e.g., number of tasks).
        device (str, optional): Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
        std (float, optional): Standard deviation for initializing the task embeddings. Default is 0.01.

    Attributes:
        num_instances (int): Number of task instances.
        device (torch.device): Device for computation.
        std (float): Standard deviation for embedding initialization.
        backbone (ConvBackbone): Convolutional network for feature extraction.
        task_head (TaskHead): Fully connected network for task-specific classification.
        hypernet (HyperNetwork): Hypernetwork to generate parameters for the task head.
        hyper_emb (nn.Embedding): Task-specific embeddings used as input to the hypernetwork.
    """
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 freeze_backbone=False,
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device=device,
                 channels=1,
                 img_size=[32, 32],
                 std=0.01):
        super().__init__()

        self.num_instances = num_instances
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.task_head_projection_size = task_head_projection_size
        self.task_head_num_classes = task_head_num_classes
        self.hyper_hidden_features = hyper_hidden_features
        self.hyper_hidden_layers = hyper_hidden_layers
        self.device = device
        self.channels = channels
        self.img_size = img_size
        self.std = std

        # Backbone
        '''self.backbone = ConvBackbone(layers=backbone_layers,
                                     input_size=(channels, img_size[0], img_size[1]),
                                     device=device)
        '''
        if backbone in backbone_dict:
            self.backbone = backbone_dict[self.backbone_name](device=device, pretrained=True)
        else: 
            raise ValueError(f"Backbone {backbone} is not supported.")
        
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        #Copy the backbone to be used for prototype extraction
        self.backbone_prototype_frozen = deepcopy(self.backbone)
        
        # freeze the backbone for prototype extraction
        for param in self.backbone_prototype_frozen.parameters():
           param.requires_grad = False
            


        # Task head
        self.task_head = TaskHead_simple(input_size=self.backbone.num_features,
                                  projection_size=task_head_projection_size,
                                  num_classes=task_head_num_classes,
                                  dropout=0.5,
                                  device=device)

        # Hypernetwork
        self.backbone_emb_size = self.backbone.num_features
        self.hyper_emb = nn.Embedding(self.num_instances, 1024)
        nn.init.normal_(self.hyper_emb.weight, mean=0, std=std)
        
    
        
        
        self.hyper_emb_prototype = nn.Linear(self.backbone_emb_size, 4096)
        nn.init.normal_(self.hyper_emb_prototype.weight, mean=0, std=std)
        
        self.hn_in = 5120 #4096
        self.hypernet = HyperNetwork_seq(hyper_in_features=self.hn_in,
                                     hyper_hidden_layers=6,
                                     hyper_hidden_features=1024,
                                     hypo_module=self.task_head,
                                     activation='relu')

            
    def get_params(self, task_idx, prototypes_backbone_out):
        # print("prototypes_backbone_out", prototypes_backbone_out.size())
        # print('task_idx', task_idx.size())
        
        # Process task index through embedding
        z = self.hyper_emb(torch.LongTensor([task_idx]).to(self.device))
        # print("z", z.size())
        
        # Process prototypes through the linear layer
        # prototypes_backbone_out shape: [num_prototypes, backbone_emb_size]
        z_prototypes = self.hyper_emb_prototype(prototypes_backbone_out)  # Shape: [num_prototypes, 4096]
        # print("z_prototypes", z_prototypes.size())
        
        # Aggregate prototype embeddings (e.g., mean)
        z_prototypes = z_prototypes.mean(dim=0).unsqueeze(0)  # Shape: [1, 4096]
        #print("z_prototypes after mean", z_prototypes.size())
        
        # Concatenate task embedding with prototype embedding
        input_hyp = torch.cat((z, z_prototypes), dim=1)  # Shape: [1, 8192]
        #print("input_hyp", input_hyp.size())
        
        return self.hypernet(input_hyp)
    
    
    def forward(self, support_set, prototypes, task_idx, **kwargs):
        # print("after get params", params)
        backbone_out = self.backbone(support_set)
        
        prototypes_backbone_out = self.backbone_prototype_frozen(prototypes)
        
        # print("backbone_out", backbone_out.size())
        params = self.get_params(task_idx, prototypes_backbone_out)
        
        
        # print("backbone_out", backbone_out.size())
        task_head_out = self.task_head(backbone_out, params=params)
        
        # print("task_head_out", task_head_out.size())
        return task_head_out.squeeze(0)
    
    def deepcopy(self):
        new_model = HyperCMTL_seq_prototype_simple(num_instances=self.num_instances,
                    backbone=self.backbone_name,
                    task_head_projection_size=self.task_head_projection_size,
                    task_head_num_classes=self.task_head_num_classes,
                    hyper_hidden_features=self.hyper_hidden_features,
                    hyper_hidden_layers=self.hyper_hidden_layers,
                    device=self.device,
                    channels=self.channels,
                    img_size=self.img_size, 
                    std=self.std)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device=self.device)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.hyper_emb.parameters(), 'lr': 1e-3})
        optimizer_list.append({'params': self.hyper_emb_prototype.parameters(), 'lr': 1e-3})
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        # print("optimizer_list", optimizer_list)
        return optimizer_list



class TaskHead(MetaModule):
    def __init__(self, input_size: int, # number of features in the backbone's output
                 projection_size: int,  # number of neurons in the hidden layer
                 num_classes: int,      # number of output neurons
                 dropout: float=0.,     # optional dropout rate to apply
                 device=device):
        super().__init__()

        self.projection = BatchLinear(input_size, projection_size)
        self.classifier = BatchLinear(projection_size, num_classes)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.relu = nn.ReLU()

        self.device = device
        self.to(device)

    def forward(self, x, params):
        # assume x is already unactivated feature logits,
        # e.g. from resnet backbone
        # # print("inside taskhead forward", params)
        # # print("after get_subdict", get_subdict(params, 'projection'))
        x = self.projection(self.relu(self.dropout(x)), params=get_subdict(params, 'projection'))
        x = self.classifier(self.relu(self.dropout(x)), params=get_subdict(params, 'classifier'))

        return x
    
    def get_optimizer_list(self):
        optimizer_list = [{'params': self.parameters(), 'lr': 1e-3}]
        return optimizer_list



class TaskHead_simple(MetaModule):
    def __init__(self, input_size: int, # number of features in the backbone's output
                 projection_size: int,  # number of neurons in the hidden layer
                 num_classes: int,      # number of output neurons
                 dropout: float=0.,     # optional dropout rate to apply
                 device=device):
        super().__init__()

        self.input_size = input_size
        self.projection_size = projection_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self.classifier = BatchLinear(input_size, num_classes, bias=False)

        self.device = device
        self.to(device)

    def forward(self, x, params):
        return self.classifier(x, params=get_subdict(params, 'classifier'))
    
    def get_optimizer_list(self):
        optimizer_list = [{'params': self.classifier.parameters(), 'lr': 1e-3}]
        return optimizer_list

    def deepcopy(self):
        new_model = TaskHead_simple(input_size=self.input_size,
                                    projection_size=self.projection_size,
                                    num_classes=self.num_classes,
                                    dropout=self.dropout,
                                    device=self.device)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device=self.device)

from collections import OrderedDict
import re
import warnings
class Backbone(nn.Module):
    def __init__(self, backbone_name, device=device, pretrained=True):
        super().__init__()
        self._children_modules_parameters_cache = dict()

        self.device = device
        self.pretrained = pretrained
        self.backbone_name = backbone_name
        self.net = MetaResNet50Backbone(device=device, pretrained=pretrained)

        self.num_features = self.net.num_features

    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            # if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None

        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names

            else:
                key_escape = re.escape(key)
                key_re = re.compile(r'^{0}\.(.+)'.format(key_escape))

                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r'\1', k) for k in all_names if key_re.match(k) is not None]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn('Module `{0}` has no parameter corresponding to the '
                          'submodule named `{1}` in the dictionary `params` '
                          'provided as an argument to `forward()`. Using the '
                          'default parameters for this submodule. The list of '
                          'the parameters in `params`: [{2}].'.format(
                          self.__class__.__name__, key, ', '.join(all_names)),
                          stacklevel=2)
            return None

        return OrderedDict([(name, params[f'{key}.{name}']) for name in names])
    
    def forward(self, x, params=None):
        return self.net(x)
    
    def get_optimizer_list(self):
        optimizer_list = [{'params': self.net.parameters(), 'lr': 1e-3}]
        return optimizer_list


    




