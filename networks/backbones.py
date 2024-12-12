import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, mobilenet_v2, resnet18, alexnet
import timm  # For EfficientNet and other models
from timm import create_model  # For ViT and other models from the "timm" library


class ResNet50(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained ResNet-50
        resnet = resnet50(pretrained=pretrained)
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
        # print(x.shape)
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, device="cuda"):
        super().__init__()

        # Load pretrained ResNet-50
        resnet = resnet18(pretrained=pretrained)
        # Remove the fully connected layer and retain only the convolutional backbone
        self.feature_extractor = nn.Sequential(
            *(list(resnet.children())[:-2])  # Removes FC and avg pooling
        )
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (512 for ResNet-50)
        self.num_features = resnet.fc.in_features
        #print(self.num_features)
        self.device = device
        self.to(device)

    def forward(self, x):
        # Pass through ResNet backbone
        x = self.feature_extractor(x)
        # #print(x.shape)
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]   

class ReducedResNet18(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load the standard ResNet-18 model
        resnet = resnet18(pretrained=pretrained)

        # Modify the convolutional layers to reduce feature maps
        resnet.conv1 = nn.Conv2d(3, 64 // 3, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.bn1 = nn.BatchNorm2d(64 // 3)

        # Adjust the number of feature maps in each block
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            self._reduce_layer(resnet.layer1, 64 // 3, 64 // 3),
            self._reduce_layer(resnet.layer2, 64 // 3, 128 // 3, stride=2),
            self._reduce_layer(resnet.layer3, 128 // 3, 256 // 3, stride=2),
            self._reduce_layer(resnet.layer4, 256 // 3, 512 // 3, stride=2),
        )

        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Set the number of output features
        self.num_features = 512 // 3
        self.device = device
        self.to(device)

    def _reduce_layer(self, layer, in_channels, out_channels, stride=1):
        modified_blocks = []
        for block in layer:
            # Modify the convolutional layers in the block
            block.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            block.bn1 = nn.BatchNorm2d(out_channels)
            block.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            block.bn2 = nn.BatchNorm2d(out_channels)
            
            # Adjust the downsample layer to match dimensions
            if block.downsample is not None or in_channels != out_channels or stride != 1:
                block.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            
            modified_blocks.append(block)
            in_channels = out_channels  # Update in_channels for the next block
        return nn.Sequential(*modified_blocks)

    def forward(self, x):
        # print("x.shape", x.shape)  # Check input shape
        x = self.feature_extractor(x)
        x = self.pool(x)
        # print("x.shape after pool", x.shape)  # Check shape after pooling
        x = x.view(x.size(0), -1)
        # print("x.shape after flatten", x.shape)  # Check final shape
        return x

    def get_optimizer_list(self):
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]




class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained MobileNetV2
        mobilenet = mobilenet_v2(pretrained=pretrained)
        # Remove the fully connected layer and retain only the convolutional backbone
        self.feature_extractor = mobilenet.features
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (1280 for MobileNetV2)
        self.num_features = mobilenet.classifier[1].in_features
        #print(self.num_features)

        self.device = device
        self.to(device)

    def forward(self, x):
        # Pass through MobileNetV2 backbone
        x = self.feature_extractor(x)
        
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]

class EfficientNetB0(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Use the timm library to load EfficientNetB0
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = self.model.classifier.in_features
        #print(self.num_features)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_optimizer_list(self):
        return [{'params': self.model.parameters(), 'lr': 1e-4}]


class AlexNet(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained AlexNet
        alexnet_ = alexnet(pretrained=pretrained)
        # Remove the fully connected layer and retain only the convolutional backbone
        self.feature_extractor = nn.Sequential(
            *(list(alexnet_.children())[:-2])  # Removes FC and avg pooling
        )
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (256 for AlexNet)
        self.num_features = alexnet_.classifier[6].in_features
        #print(self.num_features)

        self.device = device
        self.to(device)

    def forward(self, x):
        # Pass through AlexNet backbone
        x = self.feature_extractor(x)
        #print(x.shape)
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]

class ViT(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained Vision Transformer
        vit = create_model('vit_base_patch16_224', pretrained=pretrained)
        #print(vit)
        # Remove the fully connected layer and retain only the transformer backbone
        self.feature_extractor = vit.forward_features
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (768 for ViT base model)
        self.num_features = vit.head.in_features
        #print(self.num_features)

        # Store the device and move the model to the correct device
        self.device = device
        self.to(device)  # Make sure the model is on the correct device

    def forward(self, x):
        # Ensure the input is on the correct device (same as the model)
        x = x.to(self.device)  # Ensure the input is on the same device as the model

        # Pass through ViT backbone
        x = self.feature_extractor(x)
        
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]




# Function to initialize the backbone
def get_backbone(name, pretrained=True, device="cuda"):
    if name == "resnet50":
        print("ResNet50", ResNet50(pretrained, device))
        return ResNet50(pretrained, device)
    if name == "resnet18":
        print("ResNet18", ResNet18(pretrained, device))
        return ResNet18(pretrained, device)
    elif name == "mobilenetv2":
        print("MobileNetV2", MobileNetV2(pretrained, device))
        return MobileNetV2(pretrained, device)
    elif name == "efficientnetb0":
        print("EfficientNetB0", EfficientNetB0(pretrained, device))
        return EfficientNetB0(pretrained, device)
    elif name == "vit":
        print("ViT", ViT(pretrained, device))
        return ViT(pretrained, device)
    elif name == "alexnet":
        print("AlexNet", AlexNet(pretrained, device))
        return AlexNet(pretrained, device)
    elif name == "resnet18_reduced":
        print("ReducedResNet18", ReducedResNet18(pretrained, device))
        return ReducedResNet18(pretrained, device)
    else:
        raise ValueError(f"Backbone {name} is not supported.")

if __name__ == "__main__":
    name = "resnet18"
    backbone = get_backbone(name, pretrained=True, device="cuda")
    #print amount of parameters
    print(sum(p.numel() for p in backbone.parameters() if p.requires_grad))




# resnet
    # def _forward_impl(self, x: Tensor) -> Tensor:
    #     # See note [TorchScript super()]
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)

    #     return x


# (backbone): ResNet50(
#     (feature_extractor): Sequential(
#       (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#       (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#       (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#       (4): Sequential(
#         (0): Bottleneck(
#           (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#           (downsample): Sequential(
#             (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )
#         (1): Bottleneck(
#           (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#         (2): Bottleneck(
#           (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#       )
#       (5): Sequential(
#         (0): Bottleneck(
#           (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#           (downsample): Sequential(
#             (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#             (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )
#         (1): Bottleneck(
#           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#         (2): Bottleneck(
#           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#         (3): Bottleneck(
#           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#       )
#       (6): Sequential(
#         (0): Bottleneck(
#           (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#           (downsample): Sequential(
#             (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#             (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )
#         (1): Bottleneck(
#           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#         (2): Bottleneck(
#           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#         (3): Bottleneck(
#           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#         (4): Bottleneck(
#           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#         (5): Bottleneck(
#           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#       )
#       (7): Sequential(
#         (0): Bottleneck(
#           (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#           (downsample): Sequential(
#             (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#             (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )
#         (1): Bottleneck(
#           (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#         (2): Bottleneck(
#           (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (relu): ReLU(inplace=True)
#         )
#       )
#     )
#     (pool): AdaptiveAvgPool2d(output_size=(1, 1))
#   )

# import re
# from collections import OrderedDict
# import torch
# import torch.nn as nn
# from torchmeta.modules import MetaModule, MetaSequential
# from torchmeta.modules import MetaConv2d, MetaBatchNorm2d
# from typing import Callable, Optional, OrderedDict as OrderedDictType

# def get_subdict(dictionary: OrderedDictType[str, torch.Tensor], key: Optional[str] = None) -> OrderedDictType[str, torch.Tensor]:
#     """
#     Extracts a subdictionary based on a given key prefix.
    
#     Args:
#         dictionary (OrderedDict): The dictionary to extract from.
#         key (str, optional): The prefix to filter keys. If None or empty, returns the entire dictionary.

#     Returns:
#         OrderedDict: A subdictionary containing only the keys that start with the specified prefix,
#         with the prefix removed from the keys.
    
#     Example:
#         dictionary = {'layer1.weight': ..., 'layer2.bias': ...}
#         get_subdict(dictionary, 'layer1') -> {'weight': ...}
#     """
#     if dictionary is None:
#         return None
#     if (key is None) or (key == ''):
#         return dictionary
#     key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
#     return OrderedDictType(
#         (key_re.sub(r'\1', k), value) 
#         for (k, value) in dictionary.items() 
#         if key_re.match(k) is not None
#     )

# class MetaBottleneck(MetaModule):
#     expansion: int = 4

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = MetaBatchNorm2d
#         width = int(planes * (base_width / 64.0)) * groups

#         # Define the three convolutional layers
#         self.conv1 = MetaConv2d(inplanes, width, kernel_size=1, bias=False)
#         self.bn1 = norm_layer(width)
#         self.conv2 = MetaConv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, dilation=dilation, bias=False)
#         self.bn2 = norm_layer(width)
#         self.conv3 = MetaConv2d(width, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x: torch.Tensor, params: Optional[OrderedDictType[str, torch.Tensor]] = None) -> torch.Tensor:
#         if params is None:
#             params = OrderedDictType(self.named_parameters())

#         identity = x

#         # First convolutional layer
#         out = self.conv1(x, params=get_subdict(params, 'conv1'))
#         out = self.bn1(out, params=get_subdict(params, 'bn1'))
#         out = self.relu(out)

#         # Second convolutional layer
#         out = self.conv2(out, params=get_subdict(params, 'conv2'))
#         out = self.bn2(out, params=get_subdict(params, 'bn2'))
#         out = self.relu(out)

#         # Third convolutional layer
#         out = self.conv3(out, params=get_subdict(params, 'conv3'))
#         out = self.bn3(out, params=get_subdict(params, 'bn3'))

#         # Downsample if necessary
#         if self.downsample is not None:
#             identity = self.downsample(x, params=get_subdict(params, 'downsample'))

#         # Add residual connection
#         out += identity
#         out = self.relu(out)

#         return out

# class MetaResNet50Backbone(MetaModule):
#     def __init__(self, pretrained: bool = False, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = MetaBatchNorm2d

#         # Initialize the feature extractor as a MetaSequential with ordered layers
#         self.feature_extractor = MetaSequential(OrderedDict([
#             ('0', MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
#             ('1', norm_layer(64)),
#             ('2', nn.ReLU(inplace=True)),
#             ('3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('4', self._make_layer(64, 64, blocks=3, norm_layer=norm_layer)),
#             ('5', self._make_layer(256, 128, blocks=4, stride=2, norm_layer=norm_layer)),
#             ('6', self._make_layer(512, 256, blocks=6, stride=2, norm_layer=norm_layer)),
#             ('7', self._make_layer(1024, 512, blocks=3, stride=2, norm_layer=norm_layer)),
#         ]))

#         # Define the pooling layer
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))

#         if pretrained:
#             self._load_pretrained_weights()

#     def _make_layer(
#         self,
#         inplanes: int,
#         planes: int,
#         blocks: int,
#         stride: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> MetaSequential:
#         if norm_layer is None:
#             norm_layer = MetaBatchNorm2d

#         downsample = None
#         if stride != 1 or inplanes != planes * MetaBottleneck.expansion:
#             # Define downsample as a MetaSequential with conv and bn
#             downsample = MetaSequential(OrderedDict([
#                 ('0', MetaConv2d(inplanes, planes * MetaBottleneck.expansion, kernel_size=1, stride=stride, bias=False)),
#                 ('1', norm_layer(planes * MetaBottleneck.expansion)),
#             ]))

#         layers = []
#         # First block with downsampling
#         layers.append(MetaBottleneck(inplanes, planes, stride, downsample, norm_layer=norm_layer))
#         inplanes = planes * MetaBottleneck.expansion
#         # Remaining blocks
#         for _ in range(1, blocks):
#             layers.append(MetaBottleneck(inplanes, planes, norm_layer=norm_layer))

#         return MetaSequential(*layers)

#     def forward(self, x: torch.Tensor, params: Optional[OrderedDictType[str, torch.Tensor]] = None) -> torch.Tensor:
#         if params is None:
#             params = OrderedDictType(self.named_parameters())

#         # Pass through feature extractor with corresponding parameters
#         x = self.feature_extractor(x, params=get_subdict(params, 'feature_extractor'))
#         # Apply pooling
#         x = self.pool(x)
#         return x



# if __name__ == "__main__":
#     # Set device (change to "cuda" if you want to use GPU)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Create a dummy input tensor (e.g., 8 images of size 224x224)
#     dummy_input = torch.randn(8, 3, 224, 224).to(device)  # Ensure the input is on the correct device


#     # Initialize the backbone (e.g., ResNet-50)
#     backbone = get_backbone("resnet50", pretrained=True, device=device)
#     print(f"Using backbone: {backbone.__class__.__name__}")

#     # Perform a forward pass with dummy input
#     output = backbone(dummy_input)
#     print("Output shape:", output.shape)

#     # Check with other backbones as well
#     # For example, using MobileNetV2:
#     backbone = get_backbone("mobilenetv2", pretrained=True, device=device)
#     output = backbone(dummy_input)
#     print(f"Output shape with {backbone.__class__.__name__}:", output.shape)
    
#     # You can similarly test with EfficientNetB0 and ViT
#     backbone = get_backbone("efficientnetb0", pretrained=True, device=device)
#     output = backbone(dummy_input)
#     print(f"Output shape with {backbone.__class__.__name__}:", output.shape)

#     #VIT NO ME ESTÁ FUNCIONANDO POR UN ERROR EN DE torch.CUDA.TENSOR VS TORCH.TENSOR
#     #backbone = get_backbone("vit", pretrained=True, device=device)
#     #output = backbone(dummy_input)
#     #print(f"Output shape with {backbone.__class__.__name__}:", output.shape)

