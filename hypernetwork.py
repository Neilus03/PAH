import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# import random
# import numpy as np

# from collections import OrderedDict
from torchmeta.modules import MetaSequential, MetaLinear

from metamodules import FCBlock, BatchLinear, HyperNetwork, get_subdict
from torchmeta.modules import MetaModule

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
                 #backbone_layers=[32, 64, 128, 256, 512],  # Backbone architecture
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
        #self.backbone_layers = backbone_layers
        self.task_head_projection_size = task_head_projection_size
        self.task_head_num_classes = task_head_num_classes
        self.hyper_hidden_features = hyper_hidden_features
        self.hyper_hidden_layers = hyper_hidden_layers
        self.device = device
        self.channels = channels
        self.std = std

        # Backbone
        '''self.backbone = ConvBackbone(layers=backbone_layers,
                                     input_size=(channels, img_size[0], img_size[1]),
                                     device=device)
        '''
        self.backbone = ConvBackbone(pretrained=True, device=device)

        # Task head
        self.task_head = TaskHead(input_size=self.backbone.num_features,
                                  projection_size=task_head_projection_size,
                                  num_classes=task_head_num_classes,
                                  dropout=0.5,
                                  device=device)

        # Hypernetwork
        hn_in = 64  # Input size for hypernetwork embedding
        self.hypernet = HyperNetwork(hyper_in_features=hn_in,
                                     hyper_hidden_layers=hyper_hidden_layers,
                                     hyper_hidden_features=hyper_hidden_features,
                                     hypo_module=self.task_head,
                                     activation='relu')

        self.hyper_emb = nn.Embedding(self.num_instances, hn_in)
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
        new_model = HyperCMTL(
            num_instances=self.num_instances,
            #backbone_layers=self.backbone_layers,
            task_head_projection_size=self.task_head_projection_size,
            task_head_num_classes=self.task_head_num_classes,
            hyper_hidden_features=self.hyper_hidden_features,
            hyper_hidden_layers=self.hyper_hidden_layers,
            device=device,
            channels=self.channels,
            std=0.01
        ).to(device)
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

'''
class ConvBackbone(nn.Module):
    def __init__(self,
                 layers=[32, 64, 128, 256, 512], # list of conv layer num_kernels
                 input_size=(1,32,32), # for grayscale images
                 device="cuda",
                ):
        super().__init__()

        in_channels, in_h, in_w = input_size

        # build the sequential stack of conv layers:
        conv_layers = []
        prev_layer_size = in_channels
        for layer_size in layers:
            conv_layers.append(nn.Conv2d(prev_layer_size, layer_size, (3,3), padding='same'))
            conv_layers.append(nn.ReLU())
            prev_layer_size = layer_size
        self.conv_stack = torch.nn.Sequential(*conv_layers)

        # global average pooling to reduce to single dimension:
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # number of output features:
        self.num_features = prev_layer_size

        self.relu = nn.ReLU()

        self.device = device
        self.to(device)

    def forward(self, x):
        # pass through conv layers:
        x = self.conv_stack(x)

        # downsample and reshape to vector:
        x = self.pool(x)
        x = x.view(-1, self.num_features)

        # return feature vector:
        return x
    
    def get_optimizer_list(self):
        optimizer_list = [{'params': self.parameters(), 'lr': 1e-3}]
        return optimizer_list
'''

class ConvBackbone(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained ResNet-50
        resnet = resnet50(pretrained=pretrained)
        print("resnet:", resnet)
        #Freeze the first few layers
        for name, param in resnet.named_parameters():
            if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
            
            
        
        # Remove the fully connected layer and retain only the convolutional backbone
        self.feature_extractor = nn.Sequential(
            *(list(resnet.children())[:-2])  # Removes FC and avg pooling
        )
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (512 for ResNet-50)
        self.num_features = resnet.fc.in_features

        self.device = device
        self.to(device)

    def forward(self, x):
        # Pass through ResNet backbone
        x = self.feature_extractor(x)
        
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]


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


class MultitaskModel(nn.Module):
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
    
    