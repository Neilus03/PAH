import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

import numpy as np
# import random
# import numpy as np

# from collections import OrderedDict
from torchmeta.modules import MetaSequential, MetaLinear

from metamodules import FCBlock, BatchLinear, HyperNetwork, get_subdict
from torchmeta.modules import MetaModule

from backbones import ResNet50, MobileNetV2, EfficientNetB0

backbone_dict = {
    'resnet50': ResNet50,
    'mobilenetv2': MobileNetV2,
    'efficientnetb0': EfficientNetB0
}

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
                 device='cuda',
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
        # print("after get params", params)
        backbone_out = self.backbone(support_set)
        task_head_out = self.task_head(backbone_out, params=params)
        
        return task_head_out.squeeze(0)
    
    def deepcopy(self, device='cuda'):
        new_model = HyperCMTL(num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device='cuda',
                 channels=1,
                 img_size=[32, 32],
                 std=0.01).to(device)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        optimizer_list.append({'params': self.hyper_emb.parameters(), 'lr': 1e-3})
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        print("optimizer_list", optimizer_list)
        return optimizer_list

class HyperCMTL_prototype(HyperCMTL):
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device='cuda',
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

        # self.hyper_emb = nn.Linear(self.backbone_emb_size, self.hn_in//self.task_head_num_classes)
        # nn.init.normal_(self.hyper_emb.weight, mean=0, std=std)
    
        self.hyper_emb = nn.Sequential(
            nn.Linear(self.backbone_emb_size*self.task_head_num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, self.hn_in),
            nn.ReLU()
        )
        # nn.init.normal_(self.hyper_emb[0].weight, mean=0, std=std)

    def get_params(self, prototype_out):
        print("prototype_out", prototype_out.size())
        input_hyper_reduced = self.hyper_emb(prototype_out.flatten().unsqueeze(0))

        print("input_hyper_reduced", input_hyper_reduced.size())

        out = self.hypernet(input_hyper_reduced)
        return out 
    

    def forward(self, support_set, prototypes, **kwargs):
        backbone_out = self.backbone(support_set)
        # print(prototypes.size())
        prototype_emb = self.backbone(prototypes)
        
        # prototype_emb = backbone_out[task_idx, :]
        # task_idx_tensor = torch.tensor(task_idx)
        # others_emb = backbone_out[~torch.isin(torch.arange(backbone_out.size(0)), task_idx_tensor)]
        
        # input_task_head = backbone_out[~torch.isin(torch.arange(backbone_out.size(0)), task_idx_tensor)]
        
        params = self.get_params(prototype_emb)
        task_head_out = self.task_head(backbone_out, params=params)
        
        return task_head_out.squeeze(0)
    
    def deepcopy(self, device='cuda'):
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
        return new_model.to(device)

class HyperCMTL_prototype_attention_old(HyperCMTL):
    def __init__(self,
                 num_instances=1,
                 backbone='resnet50',  # Backbone architecture
                 task_head_projection_size=64,             # Task head hidden layer size
                 task_head_num_classes=2,                  # Task head output size
                 hyper_hidden_features=256,                # Hypernetwork hidden layer size
                 hyper_hidden_layers=2,                    # Hypernetwork number of layers
                 device='cuda',
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
        # print(prototypes.size())
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
        print("optimizer_list", optimizer_list)
        return optimizer_list

    def deepcopy(self, device='cuda'):
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
                 device='cuda',
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
        print("Q", Q.size())
        K = self.key(backbone_out)
        print("K", K.size())
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
        # print(prototypes.size())
        prototype_emb = self.backbone(prototypes)
        print("prototype_emb", prototype_emb.size())
        
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
        print("optimizer_list", optimizer_list)
        return optimizer_list

    def deepcopy(self, device='cuda'):
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


class TaskHead(MetaModule):
    def __init__(self, input_size: int, # number of features in the backbone's output
                 projection_size: int,  # number of neurons in the hidden layer
                 num_classes: int,      # number of output neurons
                 dropout: float=0.,     # optional dropout rate to apply
                 device="cuda"):
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
        # print("inside taskhead forward", params)
        # print("after get_subdict", get_subdict(params, 'projection'))
        x = self.projection(self.relu(self.dropout(x)), params=get_subdict(params, 'projection'))
        x = self.classifier(self.relu(self.dropout(x)), params=get_subdict(params, 'classifier'))

        return x
    
    def get_optimizer_list(self):
        optimizer_list = [{'params': self.parameters(), 'lr': 1e-3}]
        return optimizer_list


'''class MultitaskModel(nn.Module):
    def __init__(self, backbone: nn.Module,
                 device="cuda"):
        super().__init__()

        self.backbone = backbone

        # a dict mapping task IDs to the classification heads for those tasks:
        self.task_heads = nn.ModuleDict()
        # we must use a nn.ModuleDict instead of a base python dict,
        # to ensure that the modules inside are properly registered in self.parameters() etc.

        self.relu = nn.ReLU()
        self.device = device
        self.to(device)

    def forward(self,
                x: torch.Tensor,
                task_id: int):

        task_id = str(int(task_id))
        # nn.ModuleDict requires string keys for some reason,
        # so we have to be sure to cast the task_id from tensor(2) to 2 to '2'

        assert task_id in self.task_heads, f"no head exists for task id {task_id}"

        # select which classifier head to use:
        chosen_head = self.task_heads[task_id]

        # activated features from backbone:
        x = self.relu(self.backbone(x))
        # task-specific prediction:
        x = chosen_head(x)

        return x

    def add_task(self,
                 task_id: int,
                 head: nn.Module):
        """accepts an integer task_id and a classification head
        associated to that task.
        adds the head to this model's collection of task heads."""
        self.task_heads[str(task_id)] = head

    @property
    def num_task_heads(self):
        return len(self.task_heads)
    
    '''