import torch
import torch.nn as nn
import torch.nn.functional as F

# import random
# import numpy as np

# from collections import OrderedDict
from torchmeta.modules import MetaSequential, MetaLinear

from TSR.metamodules2 import FCBlock, BatchLinear, HyperNetwork, get_subdict
from torchmeta.modules import MetaModule

class HyperCMTL(nn.Module):
    def __init__(self,
                 num_instances=1, 
                 device='cuda',
                 std=0.01,
                 ):
        super().__init__()

        self.num_instances = num_instances
        self.device = device

        self.std = std

        # video decomposition network
        self.backbone = ConvBackbone(layers=[32, 64, 128, 256, 512], # list of conv layer num_kernels
                                input_size=(1,32,32), # for grayscale images
                                device=device)

        self.task_head = TaskHead(input_size=self.backbone.num_features,
                                    projection_size=64,
                                    num_classes=2,
                                    dropout=0.5,
                                    device=device)

        ## hypernetwork
        hn_in = 64

        self.hypernet = HyperNetwork(hyper_in_features=hn_in,
                                     hyper_hidden_layers=2,
                                     hyper_hidden_features=256,
                                     hypo_module=self.task_head, 
                                     activation='relu')

        self.hyper_emb = nn.Embedding(self.num_instances, hn_in)
        nn.init.normal_(self.hyper_emb.weight, mean=0, std=std)

    def get_params(self, task_idx):
        # print(torch.LongTensor([task_idx]).to(self.device))
        z = self.hyper_emb(torch.LongTensor([task_idx]).to(self.device))
        # print("z", z)
        out = self.hypernet(z)
        # print("out", out)
        return out

    def forward(self, support_set, task_idx, **kwargs):
        params = self.get_params(task_idx)
        
        backbone_out = self.backbone(support_set)
        task_head_out = self.task_head(backbone_out, params=params)
        
        return task_head_out.squeeze(0)
    
    def deepcopy(self, device='cuda'):
        new_model = HyperCMTL(num_instances=self.num_instances, device=self.device, std=self.std)
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


class HyperCMTL2(nn.Module): # changing the input to the hypernet
    def __init__(self,
                 num_instances=1, 
                 device='cuda',
                 std=0.01,
                 ):
        super().__init__()

        self.num_instances = num_instances
        self.device = device

        self.std = std

        # video decomposition network
        self.backbone = ConvBackbone(layers=[32, 64, 128, 256, 512], # list of conv layer num_kernels
                                input_size=(1,32,32), # for grayscale images
                                device=device)

        self.task_head = TaskHead(input_size=self.backbone.num_features,
                                    projection_size=64,
                                    num_classes=2,
                                    dropout=0.5,
                                    device=device)

        ## hypernetwork
        hn_in = 64
        backbone_emb_size = self.backbone.num_features
        batch_size = 256

        self.hypernet = HyperNetwork(hyper_in_features=hn_in,
                                     hyper_hidden_layers=2,
                                     hyper_hidden_features=256,
                                     hypo_module=self.task_head, 
                                     activation='relu')

        # self.reduce_batch = nn.Linear(batch_size, 1)
        # nn.init.normal_(self.reduce_batch.weight, mean=0, std=std)
        
        # self.attention = nn.Linear(backbone_emb_size, 1)
        # nn.init.normal_(self.attention.weight, mean=0, std=std)
        
        self.hyper_emb = nn.Linear(backbone_emb_size, hn_in//2)
        nn.init.normal_(self.hyper_emb.weight, mean=0, std=std)

    def get_params(self, backbone_out):
        input_hyper_reduced = self.hyper_emb(backbone_out)
        input_hyper_reduced = input_hyper_reduced.flatten()
        
        # padding = torch.zeros(256 - backbone_out.shape[0], backbone_out.shape[1]).to(self.device)
        # if backbone_out.shape[0] != 256:
        #     backbone_out = torch.cat([backbone_out, padding], dim=0)
            
        # attention_vector = self.attention(backbone_out)
        # attention_vector[padding.shape[0]:] = -1e9
        # attention_vector = F.softmax(attention_vector, dim=0)
        # backbone_out = torch.sum(attention_vector * backbone_out, dim=0)
        # z = self.hyper_emb(backbone_out)
        
        out = self.hypernet(input_hyper_reduced)
        return out

    def forward(self, support_set, task_idx, **kwargs):
        backbone_out = self.backbone(support_set)
        
        input_hyper = backbone_out[task_idx, :]
        
        task_idx_tensor = torch.tensor(task_idx)
        input_task_head = backbone_out[~torch.isin(torch.arange(backbone_out.size(0)), task_idx_tensor)]
        
        params = self.get_params(input_hyper)
        task_head_out = self.task_head(input_task_head, params=params)
        
        return task_head_out.squeeze(0)
    
    def deepcopy(self, device='cuda'):
        new_model = HyperCMTL2(num_instances=self.num_instances, device=self.device, std=self.std)
        new_model.load_state_dict(self.state_dict())
        return new_model.to(device)
    
    def get_optimizer_list(self):
        # networks = [self.backbone, self.task_head, self.hypernet, self.hyper_emb]
        optimizer_list = []
        # optimizer_list.append({'params': self.hyper_emb.parameters(), 'lr': 1e-3})
        # optimizer_list.append({'params': self.reduce_batch.parameters(), 'lr': 1e-3})
        optimizer_list.append({'params': self.hyper_emb.parameters(), 'lr': 1e-3})
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.backbone.get_optimizer_list())
        optimizer_list.extend(self.task_head.get_optimizer_list())
        # print("optimizer_list", optimizer_list)
        return optimizer_list



class ConvBackbone(nn.Module):
    def __init__(self,
                 layers=[32, 64, 128, 256, 512], # list of conv layer num_kernels
                 input_size=(1,32,32), # for grayscale images
                 device='cuda',
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


class TaskHead(MetaModule):
    def __init__(self, input_size: int, # number of features in the backbone's output
                 projection_size: int,  # number of neurons in the hidden layer
                 num_classes: int,      # number of output neurons
                 dropout: float=0.,     # optional dropout rate to apply
                 device='cuda'):
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
                 device='cuda'):
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
    
    